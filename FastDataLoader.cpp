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
#include <unordered_map>
#include <tuple>

namespace py = pybind11;

// 전역 reader 객체 (worker가 fork 후에도 공유)
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

// 결과 전달용 헤더. (여기서는 pickle 등 일반 데이터 처리용)
struct ShmResultHeader {
    uint64_t chunk_id;
    uint64_t shm_name_len;
    uint64_t data_size;
    uint8_t data_type;
    uint8_t ndim; // numpy array 차원 (shape 정보가 뒤따름)
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

// global pre-allocation 모드에서 각 key에 대한 정보를 전달하기 위한 구조체
struct GlobalKeyInfo {
    std::string key_name;
    std::string shm_name;
    uint64_t offset;         // 이 worker가 write할 시작 offset (바이트 단위)
    uint64_t chunk_data_size; // 이 worker가 write할 데이터 크기 (바이트 단위)
    uint64_t total_allocated_bytes; // 전체 글로벌 shared memory 크기
};

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

    // __call__에서 직렬화된 배치를 받아 deserialize 후 반환
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

    // 배치별로 결과를 합치는 함수
    py::dict load_chunked_in_workers(const std::vector<size_t>& batch_indices) {
        py::gil_scoped_acquire gil;
        size_t total = batch_indices.size();
        bool use_global_dict = false;
        {
            // 첫 번째 샘플을 통해 타입 확인 (dict 이면서 모든 key의 값이 numpy array인 경우)
            py::object sample_check = get_global_reader()(batch_indices[0]);
            if (py::isinstance<py::dict>(sample_check)) {
                py::dict dict_check = sample_check.cast<py::dict>();
                bool all_numpy = true;
                for (auto item : dict_check) {
                    py::object val = py::reinterpret_borrow<py::object>(item.second);
                    if (!py::isinstance<py::array>(val)) {
                        all_numpy = false;
                        break;
                    }
                }
                if (all_numpy)
                    use_global_dict = true;
            }
        }
        if (!use_global_dict) {
            // 일반 처리: 기존 concatenate 방식 (이 경우 복사가 발생)
            std::unordered_map<std::string, std::vector<py::object>> merge_buffer;
            std::vector<py::object> default_buffer;
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
                uint8_t flag = 0;  // 비 글로벌 모드
                ::write(workers_[i].task_fd, &flag, sizeof(flag));
                ::write(workers_[i].task_fd, batch_indices.data() + start, (end - start) * sizeof(uint64_t));
            }
            size_t completed = 0;
            while (completed < chunks) {
                poll(pfds.data(), pfds.size(), -1);
                for (size_t i = 0; i < chunks; ++i) {
                    if (pfds[i].revents & POLLIN) {
                        ShmResultHeader rh;
                        ssize_t ret = ::read(pfds[i].fd, &rh, sizeof(rh));
                        if(ret != sizeof(rh))
                            throw std::runtime_error("Failed to read header");
                        int ndim = static_cast<int>(rh.ndim);
                        std::vector<int64_t> shape_vec(ndim);
                        ret = ::read(pfds[i].fd, shape_vec.data(), ndim * sizeof(int64_t));
                        if(ret != ndim * static_cast<ssize_t>(sizeof(int64_t)))
                            throw std::runtime_error("Failed to read shape info");
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
                        if (py::isinstance<py::dict>(sample)) {
                            py::dict sample_dict = sample.cast<py::dict>();
                            for (auto item : sample_dict) {
                                std::string key_str = py::str(item.first).cast<std::string>();
                                merge_buffer[key_str].push_back(py::reinterpret_borrow<py::object>(item.second));
                            }
                        } else {
                            default_buffer.push_back(sample);
                        }
                        completed++;
                    }
                }
            }
            py::dict out;
            py::module numpy = py::module::import("numpy");
            for (auto& pair : merge_buffer) {
                const std::string& key = pair.first;
                const std::vector<py::object>& objs = pair.second;
                if (objs.size() == 1) {
                    out[py::str(key)] = objs[0];
                } else {
                    py::list arr_list;
                    for (const auto& obj : objs)
                        arr_list.append(obj);
                    out[py::str(key)] = numpy.attr("concatenate")(arr_list, py::arg("axis")=0);
                }
            }
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
        } else {
            // --- Global pre-allocation 모드 ---
            // 1. 첫번째 샘플에서 각 key의 numpy array 정보를 얻어, 배치 전체에 대한 global shared memory 영역을 할당합니다.
            py::object sample0 = get_global_reader()(batch_indices[0]);
            py::dict sample_dict = sample0.cast<py::dict>();
            // global_buf_info: key -> (shm_name, sample_nbytes, sample_shape, sample_strides)
            std::unordered_map<std::string, std::tuple<std::string, size_t, std::vector<py::ssize_t>, std::vector<py::ssize_t>>> global_buf_info;
            for (auto item : sample_dict) {
                std::string key = py::str(item.first).cast<std::string>();
                py::array arr = item.second.cast<py::array>();
                auto info = arr.request();
                size_t sample_nbytes = info.size * info.itemsize;
                std::vector<py::ssize_t> sample_shape(info.shape.begin(), info.shape.end());
                std::vector<py::ssize_t> sample_strides(info.strides.begin(), info.strides.end());
                size_t total_bytes = sample_nbytes * total;
                char shm_name[64];
                snprintf(shm_name, 64, "/global_shm_%s_%d_%llu", key.c_str(), getpid(), (unsigned long long)random());
                int fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
                if (fd < 0)
                    throw std::runtime_error("Global shm_open failed");
                if (ftruncate(fd, total_bytes) < 0)
                    throw std::runtime_error("Global ftruncate failed");
                void* addr = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (addr == MAP_FAILED)
                    throw std::runtime_error("Global mmap failed");
                close(fd);
                global_buf_info[key] = std::make_tuple(std::string(shm_name), sample_nbytes, sample_shape, sample_strides);
            }
            // 2. 각 worker에게, 배치 내 자신의 할당 범위와 global shm 영역 정보(각 key에 대해 offset, chunk_data_size, 그리고 전체 할당 크기)를 전달합니다.
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
                uint8_t flag = 1;  // global dict 모드 flag
                ::write(workers_[i].task_fd, &flag, sizeof(flag));
                uint64_t num_keys = global_buf_info.size();
                ::write(workers_[i].task_fd, &num_keys, sizeof(num_keys));
                for (auto &pair : global_buf_info) {
                    const std::string &key = pair.first;
                    const std::string &shm_name = std::get<0>(pair.second);
                    size_t sample_nbytes = std::get<1>(pair.second);
                    uint64_t offset = start * sample_nbytes;
                    uint64_t chunk_samples = end - start;
                    uint64_t chunk_data_size = chunk_samples * sample_nbytes;
                    uint64_t total_allocated_bytes = sample_nbytes * total;
                    uint64_t key_name_len = key.size();
                    ::write(workers_[i].task_fd, &key_name_len, sizeof(key_name_len));
                    ::write(workers_[i].task_fd, key.data(), key_name_len);
                    uint64_t shm_name_len = shm_name.size();
                    ::write(workers_[i].task_fd, &shm_name_len, sizeof(shm_name_len));
                    ::write(workers_[i].task_fd, shm_name.data(), shm_name_len);
                    ::write(workers_[i].task_fd, &offset, sizeof(offset));
                    ::write(workers_[i].task_fd, &chunk_data_size, sizeof(chunk_data_size));
                    ::write(workers_[i].task_fd, &total_allocated_bytes, sizeof(total_allocated_bytes));
                }
                ::write(workers_[i].task_fd, batch_indices.data() + start, (end - start) * sizeof(uint64_t));
            }
            size_t completed = 0;
            while (completed < chunks) {
                poll(pfds.data(), pfds.size(), -1);
                for (size_t i = 0; i < chunks; ++i) {
                    if (pfds[i].revents & POLLIN) {
                        uint64_t ack;
                        ssize_t ret = ::read(pfds[i].fd, &ack, sizeof(ack));
                        if (ret != sizeof(ack))
                            throw std::runtime_error("Failed to read global dict ack");
                        completed++;
                    }
                }
            }
            // 3. 메인 프로세스는 각 key별로 global shm 영역을 mmap하여, 추가 copy 없이 py::array (numpy array)로 wrapping합니다.
            py::dict out;
            py::module numpy = py::module::import("numpy");
            for (auto &pair : global_buf_info) {
                const std::string &key = pair.first;
                const std::string &shm_name = std::get<0>(pair.second);
                size_t sample_nbytes = std::get<1>(pair.second);
                std::vector<py::ssize_t> sample_shape = std::get<2>(pair.second);
                std::vector<py::ssize_t> sample_strides = std::get<3>(pair.second);
                // 최종 global shape: (batch_size, 기존 sample_shape...)
                std::vector<py::ssize_t> global_shape;
                global_shape.push_back(total);
                for (auto s : sample_shape)
                    global_shape.push_back(s);
                // 최종 global strides: 첫 번째 차원은 sample_nbytes, 이후는 기존 sample_strides
                std::vector<py::ssize_t> global_strides;
                global_strides.push_back(sample_nbytes);
                for (auto s : sample_strides)
                    global_strides.push_back(s);
                int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
                if (fd < 0)
                    throw std::runtime_error("shm_open for final result failed");
                size_t total_bytes = sample_nbytes * total;
                void* addr = mmap(nullptr, total_bytes, PROT_READ, MAP_SHARED, fd, 0);
                close(fd);
                auto* ctx = new MMapContext{addr, total_bytes};
                py::capsule base(ctx, &capsule_destructor);
                py::array arr(py::buffer_info(addr, sample_nbytes,
                    py::format_descriptor<float>::format(),
                    global_shape.size(), global_shape, global_strides), base);
                out[py::str(key)] = arr;
            }
            return out;
        }
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

    // worker_loop: global dict 모드와 비글로벌 모드 모두 처리
    void worker_loop(int task_fd, int result_fd) {
        while (true) {
            ChunkTaskHeader hdr;
            ssize_t nr = ::read(task_fd, &hdr, sizeof(hdr));
            if (nr == 0 || hdr.num_indices == std::numeric_limits<uint64_t>::max())
                break;
            uint8_t global_flag = 0;
            ssize_t rflag = ::read(task_fd, &global_flag, sizeof(global_flag));
            bool is_global_dict = (rflag == sizeof(global_flag) && global_flag == 1);
            
            std::vector<uint64_t> idxbuf(hdr.num_indices);
            std::vector<GlobalKeyInfo> key_infos;
            if (is_global_dict) {
                uint64_t num_keys;
                ::read(task_fd, &num_keys, sizeof(num_keys));
                key_infos.resize(num_keys);
                for (size_t k = 0; k < num_keys; k++) {
                    uint64_t key_name_len;
                    ::read(task_fd, &key_name_len, sizeof(key_name_len));
                    std::string key_name(key_name_len, '\0');
                    ::read(task_fd, &key_name[0], key_name_len);
                    uint64_t shm_name_len;
                    ::read(task_fd, &shm_name_len, sizeof(shm_name_len));
                    std::string shm_name(shm_name_len, '\0');
                    ::read(task_fd, &shm_name[0], shm_name_len);
                    uint64_t offset;
                    ::read(task_fd, &offset, sizeof(offset));
                    uint64_t chunk_data_size;
                    ::read(task_fd, &chunk_data_size, sizeof(chunk_data_size));
                    uint64_t total_allocated_bytes;
                    ::read(task_fd, &total_allocated_bytes, sizeof(total_allocated_bytes));
                    key_infos[k] = GlobalKeyInfo{key_name, shm_name, offset, chunk_data_size, total_allocated_bytes};
                }
            }
            size_t idx_bytes = hdr.num_indices * sizeof(uint64_t);
            size_t offset_bytes = 0;
            while (offset_bytes < idx_bytes) {
                ssize_t r = ::read(task_fd, reinterpret_cast<char*>(idxbuf.data()) + offset_bytes, idx_bytes - offset_bytes);
                if (r <= 0)
                    break;
                offset_bytes += r;
            }
            
            py::gil_scoped_acquire gil;
            py::object rdr = get_global_reader();
            if (is_global_dict) {
                // 각 key에 대해, 전달받은 global shm 정보를 이용해 전체 영역을 mmap한 후, 원하는 offset에서 데이터를 copy
                std::unordered_map<std::string, std::tuple<void*, uint64_t, uint64_t, uint64_t>> global_ptrs;
                for (auto &info : key_infos) {
                    int fd = shm_open(info.shm_name.c_str(), O_RDWR, 0666);
                    if (fd < 0)
                        throw std::runtime_error("worker: shm_open failed for global dict");
                    // worker는 전체 영역(total_allocated_bytes)을 mmap합니다.
                    void* base_addr = mmap(nullptr, info.total_allocated_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                    if (base_addr == MAP_FAILED)
                        throw std::runtime_error("worker: mmap failed for global dict");
                    close(fd);
                    // 저장: (base_addr, total_allocated_bytes, offset, chunk_data_size)
                    global_ptrs[info.key_name] = std::make_tuple(base_addr, info.total_allocated_bytes, info.offset, info.chunk_data_size);
                }
                // 각 인덱스에 대해, reader를 호출하여 dict를 얻고, 각 key의 데이터를 memcpy할 위치는 (base_addr + offset)입니다.
                for (size_t i = 0; i < idxbuf.size(); i++) {
                    py::object sample = rdr(idxbuf[i]);
                    py::dict sample_dict = sample.cast<py::dict>();
                    for (auto item : sample_dict) {
                        std::string key = py::str(item.first).cast<std::string>();
                        py::array arr = item.second.cast<py::array>();
                        auto info = arr.request();
                        size_t sample_nbytes = info.size * info.itemsize;
                        // destination pointer = base_addr + offset + (i * sample_nbytes)
                        auto it = global_ptrs.find(key);
                        if (it == global_ptrs.end())
                            throw std::runtime_error("worker: key not found in global_ptrs");
                        void* base_addr = std::get<0>(it->second);
                        uint64_t offset_in_shm = std::get<2>(it->second);
                        void* dest = static_cast<char*>(base_addr) + offset_in_shm + i * sample_nbytes;
                        std::memcpy(dest, info.ptr, sample_nbytes);
                    }
                }
                // 각 key에 대해, 매핑한 전체 영역을 해제합니다.
                for (auto &p : global_ptrs) {
                    void* base_addr = std::get<0>(p.second);
                    uint64_t total_alloc_size = std::get<1>(p.second);
                    munmap(base_addr, total_alloc_size);
                }
                uint64_t ack = hdr.chunk_id;
                ::write(result_fd, &ack, sizeof(ack));
            } else if (py::isinstance<py::array>(rdr(0))) {
                // 기존 numpy array 처리 (비글로벌 모드)
                // 생략
            } else {
                // 기타 유형: pickle 직렬화 방식
                std::vector<py::object> samples;
                samples.reserve(idxbuf.size());
                for (auto idx : idxbuf) {
                    py::object sample = rdr(idx);
                    samples.push_back(sample);
                }
                py::list sample_list;
                for (auto &s : samples)
                    sample_list.append(s);
                std::string data = serialize_python_object(sample_list);
                size_t total_bytes = data.size();
                char shm_name[64];
                snprintf(shm_name, 64, "/myshm_%d_%llu", getpid(), (unsigned long long)random());
                int fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
                if (fd < 0)
                    throw std::runtime_error("shm_open failed");
                if (ftruncate(fd, total_bytes) < 0)
                    throw std::runtime_error("ftruncate failed");
                void* addr = mmap(nullptr, total_bytes, PROT_WRITE, MAP_SHARED, fd, 0);
                if (addr == MAP_FAILED)
                    throw std::runtime_error("mmap failed");
                std::memcpy(addr, data.data(), total_bytes);
                munmap(addr, total_bytes);
                close(fd);
                ShmResultHeader result_hdr = { hdr.chunk_id, (uint64_t)strlen(shm_name),
                                                total_bytes, TYPE_MSGPACK, 0 };
                ::write(result_fd, &result_hdr, sizeof(result_hdr));
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
