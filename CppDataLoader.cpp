// CppDataLoader 전체 코드 - multiprocess prefetching 적용 버전
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

struct ShmResultHeader {
    uint64_t chunk_id;
    uint64_t shm_name_len;
    uint64_t data_size;
    uint8_t data_type;
    int64_t shape0, shape1, shape2, shape3;
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

// numpy.ndarray이면 tolist()로 변환하도록 default lambda 지정
std::string serialize_python_object(const py::object& obj) {
    py::module msgpack = py::module::import("msgpack");
    py::object packb = msgpack.attr("packb");
    // 만약 obj가 numpy array이면 tolist()로 변환하여 직렬화
    py::bytes data = packb(obj, 
                             py::arg("use_bin_type") = true,
                             py::arg("default") = py::cpp_function([](py::object o) {
                                 if (py::isinstance<py::array>(o))
                                     return o.attr("tolist")();
                                 throw std::runtime_error("Cannot serialize object");
                             }));
    return std::string(data);
}

py::object deserialize_python_object(const char* data, size_t size) {
    py::module msgpack = py::module::import("msgpack");
    py::object unpackb = msgpack.attr("unpackb");
    py::bytes bytes_obj = py::bytes(data, size);
    return unpackb(bytes_obj, py::arg("raw") = false);
}

class CppDataLoader {
public:
    // 기존 worker 관련 코드는 그대로 유지
    struct Worker {
        pid_t pid;
        int task_fd;
        int result_fd;
    };

    // prefetch_count 옵션 (미리 로드할 배치 개수; multiprocess 방식에서는 파이프 버퍼로 관리)
    CppDataLoader(py::object reader, size_t dataset_len, size_t batch_size,
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

        // prefetch 파이프 생성 (부모: prefetch_read_fd, 자식: prefetch_write_fd)
        if (pipe(prefetch_pipe_) < 0)
            throw std::runtime_error("pipe 생성 실패");
        pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("fork 실패");
        } else if (pid == 0) {
            // 자식 프로세스: prefetch 프로세스
            // 부모와의 통신을 위해 읽기 fd 닫기
            close(prefetch_pipe_[0]);
            prefetch_loop();
            _exit(0);
        } else {
            // 부모 프로세스: prefetch 프로세스의 pid 저장, 읽기 fd 유지
            prefetch_pid_ = pid;
            close(prefetch_pipe_[1]);
        }
    }

    ~CppDataLoader() {
        // prefetch 프로세스 종료 요청
        if (prefetch_pid_ > 0) {
            kill(prefetch_pid_, SIGTERM);
            waitpid(prefetch_pid_, nullptr, 0);
        }
        if (persistent_workers_)
            shutdown_workers();
    }

    // __call__에서는 prefetch 파이프에서 직렬화된 배치를 읽어 msgpack으로 역직렬화한 후 반환합니다.
    py::object operator()() {
        // 먼저 배치의 크기를 읽습니다 (4바이트 정수)
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

    // prefetch 프로세스와 통신하기 위한 파이프: prefetch_pipe_[0]는 부모 읽기용, [1]는 자식 쓰기용
    int prefetch_pipe_[2];
    pid_t prefetch_pid_;
    size_t prefetch_count_; // 사용되지 않지만 옵션으로 남겨둠

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
                workers_[i] = Worker{pid, task_pipe[1], result_pipe[0]};
            }
        }
    }

    void shutdown_workers() {
        for (auto& w : workers_) {
            ChunkTaskHeader hdr = {0, std::numeric_limits<uint64_t>::max()};
            ::write(w.task_fd, &hdr, sizeof(hdr));
            close(w.task_fd);
            close(w.result_fd);
            waitpid(w.pid, nullptr, 0);
        }
        workers_.clear();
    }

    py::list load_in_main(const std::vector<size_t>& batch_indices) {
        py::gil_scoped_acquire gil;
        py::list out;
        py::object rdr = get_global_reader();
        for (auto i : batch_indices)
            out.append(rdr(i));
        return out;
    }

    py::list load_chunked_in_workers(const std::vector<size_t>& batch_indices) {
        py::gil_scoped_acquire gil;
        py::list out;
        size_t total = batch_indices.size();
        size_t chunks = workers_.size();
        size_t chunk_size = (total + chunks - 1) / chunks;
        std::vector<pollfd> pfds(chunks);
        for (size_t i = 0; i < chunks; ++i)
            pfds[i] = {workers_[i].result_fd, POLLIN, 0};
        for (size_t i = 0; i < chunks; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total);
            if (start >= end)
                continue;
            ChunkTaskHeader hdr = {i, end - start};
            ::write(workers_[i].task_fd, &hdr, sizeof(hdr));
            ::write(workers_[i].task_fd, batch_indices.data() + start, (end - start) * sizeof(uint64_t));
        }
        size_t completed = 0;
        while (completed < chunks) {
            poll(pfds.data(), pfds.size(), -1);
            for (size_t i = 0; i < chunks; ++i) {
                if (pfds[i].revents & POLLIN) {
                    ShmResultHeader rh;
                    ::read(pfds[i].fd, &rh, sizeof(rh));
                    std::string shm_name(rh.shm_name_len, '\0');
                    ::read(pfds[i].fd, &shm_name[0], rh.shm_name_len);
                    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
                    shm_unlink(shm_name.c_str());
                    void* addr = mmap(nullptr, rh.data_size, PROT_READ, MAP_SHARED, fd, 0);
                    close(fd);
                    auto* ctx = new MMapContext{addr, rh.data_size};
                    py::capsule base(ctx, &capsule_destructor);
                    if (rh.data_type == TYPE_NUMPY) {
                        std::vector<py::ssize_t> shape = {rh.shape0, rh.shape1, rh.shape2, rh.shape3};
                        std::vector<py::ssize_t> strides = {
                            rh.shape1 * rh.shape2 * rh.shape3 * sizeof(float),
                            rh.shape2 * rh.shape3 * sizeof(float),
                            rh.shape3 * sizeof(float),
                            sizeof(float)
                        };
                        py::array arr(py::buffer_info(addr, sizeof(float),
                                                      py::format_descriptor<float>::format(),
                                                      4, shape, strides), base);
                        out.append(arr);
                    } else {
                        py::object obj = deserialize_python_object(static_cast<char*>(addr), rh.data_size);
                        out.append(obj);
                    }
                    completed++;
                }
            }
        }
        return out;
    }

    // prefetch_loop: 별도 프로세스에서 배치를 미리 로드하여 파이프에 직렬화된 데이터를 씁니다.
    void prefetch_loop() {
        // prefetch 프로세스 내에서는 Python 인터프리터가 별도로 실행되므로 GIL 문제 없이 호출 가능
        while (true) {
            // 만약 데이터셋 끝에 도달하면 epoch 종료 처리
            if (current_index_ >= dataset_len_)
                end_epoch();
            std::vector<size_t> batch_indices(
                indices_.begin() + current_index_,
                indices_.begin() + std::min(current_index_ + batch_size_, dataset_len_)
            );
            current_index_ += batch_size_;
            py::object batch;
            if (num_workers_ == 0)
                batch = load_in_main(batch_indices);
            else
                batch = load_chunked_in_workers(batch_indices);
            // dict 배치라면 key별로 배치화 처리
            py::list batch_list = batch.cast<py::list>();
            if (batch_list.size() > 0 && py::isinstance<py::dict>(batch_list[0])) {
                py::dict batched;
                py::dict first = batch_list[0].cast<py::dict>();
                py::object np = py::module::import("numpy");
                for (auto item : first) {
                    py::object key = py::reinterpret_borrow<py::object>(item.first);
                    py::list values;
                    for (auto sample : batch_list) {
                        py::dict sample_dict = sample.cast<py::dict>();
                        values.append(sample_dict[key]);
                    }
                    if (values.size() > 0 && py::isinstance<py::array>(values[0]))
                        batched[key] = np.attr("stack")(values);
                    else
                        batched[key] = np.attr("array")(values);
                }
                batch = batched;
            }
            // 직렬화: prefetch 프로세스에서는 배치를 msgpack으로 직렬화하여 파이프에 기록합니다.
            std::string serialized = serialize_python_object(batch);
            uint32_t size32 = serialized.size();
            // 먼저 직렬화된 데이터 크기를 4바이트 정수로 기록하고,
            write(prefetch_pipe_[1], &size32, sizeof(size32));
            // 그 다음 직렬화된 데이터를 기록합니다.
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
                int64_t shape0 = 0, shape1 = 0, shape2 = 0, shape3 = 0;
                if (is_numpy) {
                    py::array arr = sample.cast<py::array>();
                    auto info = arr.request();
                    shape0 = info.shape[0];
                    shape1 = info.ndim > 1 ? info.shape[1] : 1;
                    shape2 = info.ndim > 2 ? info.shape[2] : 1;
                    shape3 = info.ndim > 3 ? info.shape[3] : 1;
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
                ShmResultHeader result_hdr = {hdr.chunk_id, strlen(shm_name), total_bytes,
                                              dtype, shape0, shape1, shape2, shape3};
                ::write(result_fd, &result_hdr, sizeof(result_hdr));
                ::write(result_fd, shm_name, result_hdr.shm_name_len);
            }
        }
        close(task_fd);
        close(result_fd);
    }
};

PYBIND11_MODULE(CppDataLoader, m) {
    m.doc() = "C++ DataLoader with POSIX SHM + Msgpack Protocol for all Python objects (multiprocess prefetching 적용)";
    py::class_<CppDataLoader>(m, "CppDataLoader")
        .def(py::init<py::object, size_t, size_t, size_t, bool, bool, bool, size_t>(),
             py::arg("reader"),
             py::arg("dataset_len"),
             py::arg("batch_size"),
             py::arg("num_workers"),
             py::arg("shuffle"),
             py::arg("drop_last"),
             py::arg("persistent_workers"),
             py::arg("prefetch_count") = 5,
             "Initialize CppDataLoader with dataset reader and settings")
        .def("__call__", &CppDataLoader::operator(), "Fetch next batch from prefetch pipe");
}
