#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <limits>
#include <sstream>
#include <cstdlib>

namespace py = pybind11;

static py::object& get_global_reader() {
    static py::object global_reader;
    return global_reader;
}

enum DataType : uint8_t {
    TYPE_NUMPY = 0,
    TYPE_MSGPACK = 1
};

struct ChunkTaskHeader {
    uint64_t chunk_id;
    uint64_t num_indices;
};

// ShmResultHeader에 ndim 필드를 추가하고, shape 정보는 header 이후에 연속으로 기록함
struct ShmResultHeader {
    uint64_t chunk_id;
    uint64_t shm_name_len;
    uint64_t data_size;
    uint8_t data_type;
    uint8_t ndim; // numpy 배열의 차원 수 (variable length shape 정보가 header 다음에 기록됨)
};

struct MMapContext {
    void* addr;
    size_t size;
};

static void capsule_destructor(void* p) {
    auto ctx = reinterpret_cast<MMapContext*>(p);
    if (!ctx) return;
    munmap(ctx->addr, ctx->size);
    delete ctx;
}

std::string serialize_python_object(const py::object& obj) {
    py::module pickle = py::module::import("pickle");
    py::object dumps = pickle.attr("dumps");
    py::bytes data = dumps(obj, py::arg("protocol") = 4);
    return std::string(data);
}

py::object deserialize_python_object(const char* data, size_t size) {
    py::module pickle = py::module::import("pickle");
    py::object loads = pickle.attr("loads");
    py::bytes bytes_obj(data, size);
    return loads(bytes_obj);
}

class FastDataLoader {
public:
    struct Worker {
        pid_t pid;
        int task_fd;
        int result_fd;
    };
    
    FastDataLoader(py::object reader, size_t dataset_len, size_t batch_size,
                   size_t num_workers, bool shuffle, bool drop_last,
                   bool persistent_workers = true, size_t prefetch_count = 5)
        : reader_(reader), dataset_len_(dataset_len), batch_size_(batch_size),
          num_workers_(num_workers), shuffle_(shuffle), drop_last_(drop_last),
          persistent_workers_(persistent_workers), current_index_(0), epoch_count_(0),
          prefetch_count_(prefetch_count)
    {
        indices_.resize(dataset_len_);
        std::iota(indices_.begin(), indices_.end(), 0);
        if (shuffle_ && dataset_len_ > 1) {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
        get_global_reader() = reader_;
        if (persistent_workers_ && num_workers_ > 0)
            spawn_workers();

        if (pipe(prefetch_pipe_) < 0)
            throw std::runtime_error("pipe 생성 실패");
        pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("fork 실패");
        } else if (pid == 0) {
            close(prefetch_pipe_[0]);
            prefetch_loop();
            _exit(0);
        } else {
            prefetch_pid_ = pid;
            close(prefetch_pipe_[1]);
        }
    }

    ~FastDataLoader() {
        if (prefetch_pid_ > 0) {
            kill(prefetch_pid_, SIGTERM);
            waitpid(prefetch_pid_, nullptr, 0);
        }
        if (persistent_workers_)
            shutdown_workers();
    }

    py::object operator()() {
        uint32_t batch_size_bytes;
        ssize_t n = read(prefetch_pipe_[0], &batch_size_bytes, sizeof(batch_size_bytes));
        if (n != sizeof(batch_size_bytes))
            return py::list();
        std::vector<char> buffer(batch_size_bytes);
        size_t offset = 0;
        while (offset < batch_size_bytes) {
            ssize_t r = read(prefetch_pipe_[0], buffer.data() + offset, batch_size_bytes - offset);
            if (r <= 0)
                break;
            offset += r;
        }
        return deserialize_python_object(buffer.data(), buffer.size());
    }

private:
    py::object reader_;
    size_t dataset_len_, batch_size_, num_workers_;
    bool shuffle_, drop_last_, persistent_workers_;
    size_t current_index_, epoch_count_;
    std::vector<size_t> indices_;
    std::vector<Worker> workers_;
    int prefetch_pipe_[2];
    pid_t prefetch_pid_;
    size_t prefetch_count_;

    void end_epoch() {
        epoch_count_++;
        current_index_ = 0;
        if (shuffle_ && dataset_len_ > 1) {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
    }

    void spawn_workers() {
        py::gil_scoped_release release;
        workers_.resize(num_workers_);
        for (size_t i = 0; i < num_workers_; i++) {
            int task_pipe[2], result_pipe[2];
            if (pipe(task_pipe) < 0 || pipe(result_pipe) < 0)
                throw std::runtime_error("pipe 실패");
            pid_t pid = fork();
            if (pid < 0)
                throw std::runtime_error("fork 실패");
            if (pid == 0) {
                close(task_pipe[1]);
                close(result_pipe[0]);
                worker_loop(task_pipe[0], result_pipe[1]);
                _exit(0);
            } else {
                close(task_pipe[0]);
                close(result_pipe[1]);
                workers_[i] = Worker{ pid, task_pipe[1], result_pipe[0] };
            }
        }
    }

    void shutdown_workers() {
        for (auto& w : workers_) {
            ChunkTaskHeader hdr = { 0, std::numeric_limits<uint64_t>::max() };
            ::write(w.task_fd, &hdr, sizeof(hdr));
            close(w.task_fd);
            close(w.result_fd);
            waitpid(w.pid, nullptr, 0);
        }
        workers_.clear();
    }

    // load_chunked_in_workers: 최종 배치 데이터를 dict 형태로 반환함
    // 만약 샘플이 numpy array인 경우, header에 기록된 ndim과 shape 정보를 이용하여 배열을 복원함.
    py::dict load_chunked_in_workers(const std::vector<size_t>& batch_indices) {
        py::gil_scoped_acquire gil;
        // 중간 버퍼: 키별로 파트별 데이터를 저장할 dict (키: vector of py::object)
        std::unordered_map<std::string, std::vector<py::object>> merge_buffer;
        // 만약 결과가 단일 값("data")인 경우도 저장할 리스트
        std::vector<py::object> default_buffer;

        size_t total = batch_indices.size();
        size_t chunks = workers_.size();
        size_t chunk_size = (total + chunks - 1) / chunks;
        std::vector<pollfd> pfds(chunks);

        for (size_t i = 0; i < chunks; ++i)
            pfds[i] = { workers_[i].result_fd, POLLIN, 0 };

        for (size_t i = 0; i < chunks; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total);
            if (start >= end)
                continue;
            ChunkTaskHeader hdr = { i, end - start };
            ::write(workers_[i].task_fd, &hdr, sizeof(hdr));
            ::write(workers_[i].task_fd, batch_indices.data() + start, (end - start) * sizeof(uint64_t));
        }

        size_t completed = 0;
        while (completed < chunks) {
            poll(pfds.data(), pfds.size(), -1);
            for (size_t i = 0; i < chunks; ++i) {
                if (pfds[i].revents & POLLIN) {
                    // 고정 길이 헤더 읽기
                    ShmResultHeader rh;
                    ssize_t ret = ::read(pfds[i].fd, &rh, sizeof(rh));
                    if(ret != sizeof(rh))
                        throw std::runtime_error("Failed to read header");

                    // shape 정보 읽기 (numpy인 경우)
                    int ndim = static_cast<int>(rh.ndim);
                    std::vector<int64_t> shape_vec(ndim);
                    ret = ::read(pfds[i].fd, shape_vec.data(), ndim * sizeof(int64_t));
                    if(ret != ndim * static_cast<ssize_t>(sizeof(int64_t)))
                        throw std::runtime_error("Failed to read shape info");

                    // shm 이름 읽기
                    std::string shm_name(rh.shm_name_len, '\0');
                    ::read(pfds[i].fd, &shm_name[0], rh.shm_name_len);

                    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
                    shm_unlink(shm_name.c_str());
                    void* addr = mmap(nullptr, rh.data_size, PROT_READ, MAP_SHARED, fd, 0);
                    close(fd);
                    auto* ctx = new MMapContext{ addr, rh.data_size };
                    py::capsule base(ctx, &capsule_destructor);

                    py::object sample;
                    if (rh.data_type == TYPE_NUMPY) {
                        std::vector<py::ssize_t> shape(ndim);
                        for (int j = 0; j < ndim; ++j)
                            shape[j] = static_cast<py::ssize_t>(shape_vec[j]);
                        std::vector<py::ssize_t> strides(ndim);
                        if(ndim > 0) {
                            strides[ndim-1] = sizeof(float);
                            for (int j = ndim - 2; j >= 0; --j)
                                strides[j] = shape[j+1] * strides[j+1];
                        }
                        py::array arr(py::buffer_info(addr, sizeof(float),
                                                    py::format_descriptor<float>::format(),
                                                    ndim, shape, strides), base);
                        sample = arr;
                    } else {
                        sample = deserialize_python_object(static_cast<char*>(addr), rh.data_size);
                    }

                    // 결과 버퍼에 저장: dict인 경우와 단일(default_key="data") 경우 분리
                    if (py::isinstance<py::dict>(sample)) {
                        py::dict sample_dict = sample.cast<py::dict>();
                        for (auto item : sample_dict) {
                            py::object key_obj = py::reinterpret_borrow<py::object>(item.first);
                            py::object value_obj = py::reinterpret_borrow<py::object>(item.second);
                            std::string key_str = py::str(key_obj).cast<std::string>();
                            merge_buffer[key_str].push_back(value_obj);
                        }
                    } else {
                        default_buffer.push_back(sample);
                    }
                    completed++;
                }
            }
        }

        // 최종 결과 dict 구성: 각 key별로 한 번의 concatenate 실행
        py::dict out;
        py::module numpy = py::module::import("numpy");
        // dict인 경우
        for (auto& pair : merge_buffer) {
            const std::string& key = pair.first;
            const std::vector<py::object>& objs = pair.second;
            if (objs.size() == 1) {
                out[py::str(key)] = objs[0];
            } else {
                // 우선 파이썬 리스트로 변환
                py::list arr_list;
                for (const auto& obj : objs)
                    arr_list.append(obj);
                out[py::str(key)] = numpy.attr("concatenate")(arr_list, py::arg("axis")=0);
            }
        }
        // 단일(default_key) 결과의 경우
        if (!default_buffer.empty()) {
            if (default_buffer.size() == 1)
                out[py::str("data")] = default_buffer[0];
            else {
                py::list arr_list;
                for (const auto& obj : default_buffer)
                    arr_list.append(obj);
                out[py::str("data")] = numpy.attr("concatenate")(arr_list, py::arg("axis")=0);
            }
        }
        return out;
    }


    void prefetch_loop() {
        while (true) {
            if (current_index_ >= dataset_len_)
                end_epoch();
            std::vector<size_t> batch_indices(
                indices_.begin() + current_index_,
                indices_.begin() + std::min(current_index_ + batch_size_, dataset_len_)
            );
            current_index_ += batch_size_;
            py::object batch = load_chunked_in_workers(batch_indices);
            std::string serialized = serialize_python_object(batch);
            uint32_t size32 = serialized.size();
            write(prefetch_pipe_[1], &size32, sizeof(size32));
            write(prefetch_pipe_[1], serialized.data(), serialized.size());
        }
    }

    void worker_loop(int task_fd, int result_fd) {
        while (true) {
            ChunkTaskHeader hdr;
            ssize_t nr = ::read(task_fd, &hdr, sizeof(hdr));
            if (nr == 0 || hdr.num_indices == std::numeric_limits<uint64_t>::max())
                break;
            std::vector<uint64_t> idxbuf(hdr.num_indices);
            ::read(task_fd, idxbuf.data(), hdr.num_indices * sizeof(uint64_t));
            {
                py::gil_scoped_acquire gil;
                py::object rdr = get_global_reader();
                py::object sample = rdr(idxbuf[0]);
                bool is_numpy = py::isinstance<py::array>(sample);
                DataType dtype = is_numpy ? TYPE_NUMPY : TYPE_MSGPACK;
                std::string data;
                size_t total_bytes = 0;
                uint8_t ndim = 0;
                std::vector<int64_t> shape_vec;
                if (is_numpy) {
                    py::array arr = sample.cast<py::array>();
                    auto info = arr.request();
                    ndim = info.ndim;
                    shape_vec.resize(ndim);
                    for (int i = 0; i < ndim; ++i)
                        shape_vec[i] = info.shape[i];
                    total_bytes = info.size * info.itemsize;
                    data.assign(static_cast<char*>(info.ptr), total_bytes);
                } else {
                    data = serialize_python_object(sample);
                    total_bytes = data.size();
                }
                char shm_name[64];
                snprintf(shm_name, 64, "/myshm_%d_%llu", getpid(), (unsigned long long)random());
                int fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
                ftruncate(fd, total_bytes);
                void* addr = mmap(nullptr, total_bytes, PROT_WRITE, MAP_SHARED, fd, 0);
                std::memcpy(addr, data.data(), total_bytes);
                munmap(addr, total_bytes);
                close(fd);
                // worker_loop에서는 ShmResultHeader와 함께 shape 정보를 기록
                ShmResultHeader result_hdr = { hdr.chunk_id, strlen(shm_name), total_bytes, dtype, ndim };
                ::write(result_fd, &result_hdr, sizeof(result_hdr));
                if (is_numpy) {
                    ::write(result_fd, shape_vec.data(), ndim * sizeof(int64_t));
                }
                ::write(result_fd, shm_name, result_hdr.shm_name_len);
            }
        }
        close(task_fd);
        close(result_fd);
    }
};

PYBIND11_MODULE(FastDataLoader, m) {
    m.doc() = "C++ DataLoader with POSIX SHM + Pickle Protocol for all Python objects (multiprocess prefetching 적용)";
    py::class_<FastDataLoader>(m, "FastDataLoader")
        .def(py::init<py::object, size_t, size_t, size_t, bool, bool, bool, size_t>(),
             py::arg("reader"),
             py::arg("dataset_len"),
             py::arg("batch_size"),
             py::arg("num_workers"),
             py::arg("shuffle"),
             py::arg("drop_last"),
             py::arg("persistent_workers"),
             py::arg("prefetch_count") = 5)
        .def("__call__", &FastDataLoader::operator(), "Fetch next batch from prefetch pipe");
}
