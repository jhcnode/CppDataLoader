#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <signal.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <random>
#include <chrono>

#include <cstdint>
#include <endian.h>

// ─────────────────────────────────────────────────────────────────────────────
// MemoryPool: POSIX shared memory 관리 + mmap/munmap
// ─────────────────────────────────────────────────────────────────────────────
class MemoryPool {
public:
    // 공유 메모리 allocate
    void* allocate(size_t size, std::string &shm_name_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string shm_name = "/fast_data_" + std::to_string(counter_++);
        int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if(fd < 0)
            throw std::runtime_error("Failed to open shm");
        if(ftruncate(fd, size) != 0) {
            close(fd);
            shm_unlink(shm_name.c_str());
            throw std::runtime_error("Failed to truncate shm");
        }
        void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if(ptr == MAP_FAILED) {
            shm_unlink(shm_name.c_str());
            throw std::runtime_error("Failed to mmap");
        }
        mapping_[ptr] = {shm_name, size};
        shm_name_out = shm_name;
        return ptr;
    }

    // 부모 프로세스 측에서 worker 가 만든 shm_name 다시 열기
    void* openShared(const std::string &shm_name, size_t size) {
        int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if(fd < 0)
            throw std::runtime_error("Failed to open shared memory in parent");
        void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if(ptr == MAP_FAILED)
            throw std::runtime_error("Failed to mmap in parent");
        {
            std::lock_guard<std::mutex> lock(mutex_);
            mapping_[ptr] = {shm_name, size};
        }
        return ptr;
    }

    // python capsule deleter 에서 호출 → 실제 shm unlink + unmap
    void release(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mapping_.find(ptr);
        if(it != mapping_.end()) {
            auto &rec = it->second;
            munmap(ptr, rec.second);
            shm_unlink(rec.first.c_str());
            mapping_.erase(it);
        }
    }

    void unlink_only(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = mapping_.find(ptr);
        if (it != mapping_.end()) {
            shm_unlink(it->second.first.c_str());
            // munmap() 은 parent 가 관리
            // mapping_ 도 parent 가 관리
        }
    }


    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

private:
    std::mutex mutex_;
    std::unordered_map<void*, std::pair<std::string, size_t>> mapping_;
    size_t counter_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// robust read/write (EINTR 안전 대책)
// ─────────────────────────────────────────────────────────────────────────────
ssize_t robust_write(int fd, const void *buf, size_t count) {
    size_t written = 0;
    const char* cbuf = (const char*)buf;
    while(written < count) {
        ssize_t ret = write(fd, cbuf + written, count - written);
        if(ret <= 0) {
            if(errno == EINTR) continue;
            return ret;
        }
        written += ret;
    }
    return written;
}

ssize_t robust_read(int fd, void *buf, size_t count) {
    size_t readn = 0;
    char* cbuf = (char*)buf;
    while(readn < count) {
        ssize_t ret = read(fd, cbuf + readn, count - readn);
        if(ret <= 0) {
            if(errno == EINTR) continue;
            return ret;
        }
        readn += ret;
    }
    return readn;
}

// ─────────────────────────────────────────────────────────────────────────────
// endian 변환 함수
// ─────────────────────────────────────────────────────────────────────────────
void write_uint64(int fd, uint64_t val) {
    val = htobe64(val);
    robust_write(fd, &val, sizeof(val));
}
uint64_t read_uint64(int fd) {
    uint64_t val;
    robust_read(fd, &val, sizeof(val));
    return be64toh(val);
}

// 문자열 직렬화/역직렬화
void write_string(int fd, const std::string &s) {
    write_uint64(fd, s.size());
    robust_write(fd, s.data(), s.size());
}
std::string read_string(int fd) {
    uint64_t len = read_uint64(fd);
    std::string s(len, '\0');
    robust_read(fd, &s[0], len);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// ArrayDesc: worker -> parent 로 전달할 하나의 array 메타데이터
// ─────────────────────────────────────────────────────────────────────────────
struct ArrayDesc {
    std::string key;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> strides;
    std::string dtype;   // "<f4", "<i4" 등 PEP 3118 포맷
    std::string shm_name;
    uint64_t size;       // 바이트 크기
    void* ptr;
};

// 직렬화
std::vector<char> serialize_batch(const std::vector<ArrayDesc>& descs) {
    std::ostringstream oss(std::ios::binary);
    uint64_t num_desc = descs.size();
    num_desc = htobe64(num_desc);
    oss.write(reinterpret_cast<const char*>(&num_desc), sizeof(num_desc));
    for(const auto &desc : descs) {
        // key
        uint64_t key_len = htobe64(desc.key.size());
        oss.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        oss.write(desc.key.data(), desc.key.size());

        // shape
        uint64_t ndim = htobe64(desc.shape.size());
        oss.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (auto d : desc.shape) {
            uint64_t dd = htobe64(d);
            oss.write(reinterpret_cast<const char*>(&dd), sizeof(dd));
        }
        // strides
        uint64_t nstr = htobe64(desc.strides.size());
        oss.write(reinterpret_cast<const char*>(&nstr), sizeof(nstr));
        for (auto st : desc.strides) {
            uint64_t sst = htobe64(st);
            oss.write(reinterpret_cast<const char*>(&sst), sizeof(sst));
        }
        // dtype
        uint64_t dt_len = htobe64(desc.dtype.size());
        oss.write(reinterpret_cast<const char*>(&dt_len), sizeof(dt_len));
        oss.write(desc.dtype.data(), desc.dtype.size());

        // shm_name
        uint64_t sn_len = htobe64(desc.shm_name.size());
        oss.write(reinterpret_cast<const char*>(&sn_len), sizeof(sn_len));
        oss.write(desc.shm_name.data(), desc.shm_name.size());

        // size
        uint64_t sz = htobe64(desc.size);
        oss.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    }
    auto str = oss.str();
    std::vector<char> buffer(str.begin(), str.end());
    return buffer;
}

// 역직렬화
std::vector<ArrayDesc> deserialize_batch(const std::vector<char>& buffer) {
    std::istringstream iss(std::string(buffer.data(), buffer.size()), std::ios::binary);
    uint64_t num_desc;
    iss.read(reinterpret_cast<char*>(&num_desc), sizeof(num_desc));
    num_desc = be64toh(num_desc);

    std::vector<ArrayDesc> out;
    out.reserve(num_desc);

    for (size_t i = 0; i < num_desc; i++) {
        ArrayDesc desc;
        // key
        uint64_t key_len;
        iss.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        key_len = be64toh(key_len);
        desc.key.resize(key_len);
        iss.read(&desc.key[0], key_len);

        // shape
        uint64_t ndim;
        iss.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        ndim = be64toh(ndim);
        desc.shape.resize(ndim);
        for (size_t j=0; j<ndim; j++) {
            uint64_t dim_;
            iss.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
            dim_ = be64toh(dim_);
            desc.shape[j] = dim_;
        }
        // strides
        uint64_t nstr;
        iss.read(reinterpret_cast<char*>(&nstr), sizeof(nstr));
        nstr = be64toh(nstr);
        desc.strides.resize(nstr);
        for (size_t j=0; j<nstr; j++) {
            uint64_t st_;
            iss.read(reinterpret_cast<char*>(&st_), sizeof(st_));
            st_ = be64toh(st_);
            desc.strides[j] = st_;
        }
        // dtype
        uint64_t dt_len;
        iss.read(reinterpret_cast<char*>(&dt_len), sizeof(dt_len));
        dt_len = be64toh(dt_len);
        desc.dtype.resize(dt_len);
        iss.read(&desc.dtype[0], dt_len);

        // shm_name
        uint64_t sn_len;
        iss.read(reinterpret_cast<char*>(&sn_len), sizeof(sn_len));
        sn_len = be64toh(sn_len);
        desc.shm_name.resize(sn_len);
        iss.read(&desc.shm_name[0], sn_len);

        // size
        uint64_t sz;
        iss.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        sz = be64toh(sz);
        desc.size = sz;

        out.push_back(desc);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 직렬화: worker가 batch 인덱스 목록을 받을 수 있도록
// ─────────────────────────────────────────────────────────────────────────────
std::vector<char> serialize_task(const std::vector<uint64_t> &indices) {
    std::ostringstream oss(std::ios::binary);
    uint64_t n = htobe64(indices.size());
    oss.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto idx : indices) {
        uint64_t i_ = htobe64(idx);
        oss.write(reinterpret_cast<const char*>(&i_), sizeof(i_));
    }
    auto s = oss.str();
    return std::vector<char>(s.begin(), s.end());
}

std::vector<uint64_t> deserialize_task(int fd) {
    uint64_t n;
    ssize_t rr = robust_read(fd, &n, sizeof(n));
    if(rr <= 0) {
        // 읽을 게 없는 경우 → 종료 신호
        return {};
    }
    n = be64toh(n);
    std::vector<uint64_t> out(n);
    for (size_t i=0; i<n; i++) {
        uint64_t v;
        robust_read(fd, &v, sizeof(v));
        v = be64toh(v);
        out[i] = v;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 전역 shutdown 플래그 (필요 시 사용)
// ─────────────────────────────────────────────────────────────────────────────
std::atomic<bool> g_shutdown(false);

// ─────────────────────────────────────────────────────────────────────────────
// Worker 프로세스
// ─────────────────────────────────────────────────────────────────────────────
void worker_signal_handler(int signum) {
    _exit(1);
}

// worker_main: task_fd 에서 인덱스 벡터를 받아 reader 호출 → shared memory 로 복사 → result_fd 로 결과 전송
void worker_main(int task_fd, int result_fd, py::object reader) {
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    signal(SIGTERM, worker_signal_handler);
    signal(SIGINT, worker_signal_handler);

    while (true) {
        auto indices = deserialize_task(task_fd);
        if (indices.empty()) {
            break;
        }

        try {
            py::gil_scoped_acquire gil;
            py::list idx_list;
            for (auto i : indices) {
                idx_list.append((py::int_)i);
            }

            py::object res_obj = reader(idx_list);
            py::dict res_dict = res_obj.cast<py::dict>();

            // descriptor: 몇 개 전달하는지
            uint64_t num_items = res_dict.size();
            write_uint64(result_fd, num_items);

            for (auto item : res_dict) {
                auto key_str = std::string(py::str(item.first));
                py::array arr = item.second.cast<py::array>();
                auto info = arr.request();

                size_t total_bytes = info.size * info.itemsize;

                // 🔥 pipe 로 descriptor + raw data 전송
                write_string(result_fd, key_str);
                write_string(result_fd, info.format);

                write_uint64(result_fd, info.ndim);
                for (int i = 0; i < info.ndim; ++i) write_uint64(result_fd, info.shape[i]);
                for (int i = 0; i < info.ndim; ++i) write_uint64(result_fd, info.strides[i]);

                write_uint64(result_fd, total_bytes);
                robust_write(result_fd, info.ptr, total_bytes);
            }
        } catch (std::exception& e) {
            // 실패 시 num_items = 0
            uint64_t zero = 0;
            write_uint64(result_fd, zero);
        }
    }

    _exit(0);
}




// Worker 구조체
struct Worker {
    pid_t pid;
    int task_fd;   // 부모가 여기로 write
    int result_fd; // 부모가 여기서 read
};

// ─────────────────────────────────────────────────────────────────────────────
// FastDataLoader 클래스
// ─────────────────────────────────────────────────────────────────────────────
class FastDataLoader {
public:
    FastDataLoader(py::object reader,
                   size_t dataset_len,
                   size_t batch_size,
                   size_t num_workers,
                   bool shuffle,
                   bool drop_last,
                   bool persistent_workers,
                   size_t prefetch_count)
    : reader_(reader),
      dataset_len_(dataset_len),
      batch_size_(batch_size),
      num_workers_(num_workers),
      shuffle_(shuffle),
      drop_last_(drop_last),
      persistent_workers_(persistent_workers),
      prefetch_count_(prefetch_count),
      current_batch_(0),
      stop_prefetch_(false)
    {
        // 인덱스 목록 초기화
        indices_.resize(dataset_len_);
        for (size_t i=0; i<dataset_len_; i++) {
            indices_[i] = i;
        }
        if(shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

        total_batches_ = dataset_len_ / batch_size_;
        if(!drop_last_ && (dataset_len_ % batch_size_ != 0)) {
            total_batches_ += 1;
        }

        // worker 프로세스 생성
        for(size_t i=0; i<num_workers_; i++) {
            int task_pipe[2];
            int result_pipe[2];
            if(pipe(task_pipe) < 0) {
                throw std::runtime_error("Failed to create task pipe");
            }
            if(pipe(result_pipe) < 0) {
                close(task_pipe[0]); close(task_pipe[1]);
                throw std::runtime_error("Failed to create result pipe");
            }
            pid_t pid = fork();
            if(pid < 0)
                throw std::runtime_error("Failed to fork worker");
            if(pid == 0) {
                // child
                close(task_pipe[1]);
                close(result_pipe[0]);
                worker_main(task_pipe[0], result_pipe[1], reader_);
            } else {
                // parent
                close(task_pipe[0]);
                close(result_pipe[1]);
                Worker w;
                w.pid = pid;
                w.task_fd = task_pipe[1];
                w.result_fd = result_pipe[0];
                workers_.push_back(w);
            }
        }

        // prefetch thread 시작
        prefetch_thread_ = std::thread(&FastDataLoader::prefetch_loop, this);
        // 모니터 thread (worker 감시)
        monitor_thread_ = std::thread(&FastDataLoader::monitor_workers, this);
    }

    ~FastDataLoader() {
        shutdown();
    }

    // shutdown: worker 종료, thread join
    void shutdown() {
        if(stopped_) return;
        stopped_ = true;
        stop_prefetch_ = true;

        // prefetch thread 종료 기다림
        if(prefetch_thread_.joinable())
            prefetch_thread_.join();

        // persistent_workers 여부와 상관없이 객체 소멸 시에는 모두 종료
        for(auto &wk : workers_) {
            // 종료 신호: 빈 task
            std::vector<char> empty_task = serialize_task({});
            robust_write(wk.task_fd, empty_task.data(), empty_task.size());
            close(wk.task_fd);
            close(wk.result_fd);
            int status;
            waitpid(wk.pid, &status, 0);
        }
        if(monitor_thread_.joinable())
            monitor_thread_.join();
    }

    // iterator 프로토콜
    FastDataLoader& __iter__() {
        return *this;
    }

    py::dict __next__() {
        if (current_batch_ >= total_batches_) {
            throw py::stop_iteration();
        }

        std::vector<char> batch_data;
        std::vector<void*> shm_ptrs;
        {
            std::unique_lock<std::mutex> lk(q_mutex_);
            q_cv_.wait(lk, [this]() {
                return !prefetch_queue_.empty() || stop_prefetch_;
            });

            if (prefetch_queue_.empty()) {
                throw py::stop_iteration();
            }

            auto front = std::move(prefetch_queue_.front());
            batch_data = std::move(front.first);
            shm_ptrs = std::move(front.second);
            prefetch_queue_.pop();
        }

        auto descs = deserialize_batch(batch_data);
        py::dict result;
        auto& pool = MemoryPool::instance();

        for (auto& d : descs) {
            void* ptr = pool.openShared(d.shm_name, d.size);

            py::capsule cap(ptr, [](void* p) {
                MemoryPool::instance().release(p);
            });

            std::vector<ssize_t> shape(d.shape.begin(), d.shape.end());
            std::vector<ssize_t> strides(d.strides.begin(), d.strides.end());

            py::array arr(py::buffer_info(
                ptr,
                1,
                d.dtype,
                shape.size(),
                shape,
                strides
            ), cap);

            result[py::str(d.key)] = arr;
        }

        // ✅ 원본 pointer release
        for (void* ptr : shm_ptrs) {
            pool.release(ptr);
        }

        current_batch_++;
        return result;
    }



    // reset(): 여러 epoch 돌릴 때 사용 → worker 유지(persistent_workers=True) 상태에서 다시 batch 0으로
    void reset() {
        // 만약 이전 prefetch_thread 가 살아있으면 join
        if (prefetch_thread_.joinable()) {
            stop_prefetch_ = true;
            prefetch_thread_.join();
        }

        // 큐 비우기
        {
            std::lock_guard<std::mutex> lk(q_mutex_);
            while (!prefetch_queue_.empty()) {
                prefetch_queue_.pop();
            }
        }

        // shuffle 이면 다시 섞기
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

        // batch 카운트 초기화
        current_batch_ = 0;

        // worker process 재생성 (persistent_workers = false 일 때는 매 epoch 마다 필요)
        if (!persistent_workers_) {
            // 이전 worker 종료
            for (auto &wk : workers_) {
                std::vector<char> empty_task = serialize_task({});
                robust_write(wk.task_fd, empty_task.data(), empty_task.size());
                close(wk.task_fd);
                close(wk.result_fd);
                int status;
                waitpid(wk.pid, &status, 0);
            }
            workers_.clear();

            // worker 재생성
            for (size_t i = 0; i < num_workers_; i++) {
                int task_pipe[2];
                int result_pipe[2];
                if (pipe(task_pipe) < 0) {
                    throw std::runtime_error("Failed to create task pipe");
                }
                if (pipe(result_pipe) < 0) {
                    close(task_pipe[0]); close(task_pipe[1]);
                    throw std::runtime_error("Failed to create result pipe");
                }
                pid_t pid = fork();
                if (pid < 0)
                    throw std::runtime_error("Failed to fork worker");
                if (pid == 0) {
                    // child
                    close(task_pipe[1]);
                    close(result_pipe[0]);
                    worker_main(task_pipe[0], result_pipe[1], reader_);
                } else {
                    // parent
                    close(task_pipe[0]);
                    close(result_pipe[1]);
                    Worker w;
                    w.pid = pid;
                    w.task_fd = task_pipe[1];
                    w.result_fd = result_pipe[0];
                    workers_.push_back(w);
                }
            }
        }

        // prefetch thread 다시 시작
        stop_prefetch_ = false;
        prefetch_thread_ = std::thread(&FastDataLoader::prefetch_loop, this);
    }


private:
    // worker 모니터 스레드
    void monitor_workers() {
        while(!stopped_) {
            for (auto &wk : workers_) {
                int status;
                pid_t res = waitpid(wk.pid, &status, WNOHANG);
                if(res > 0) {
                    // worker 비정상 종료
                    std::cerr << "[FastDataLoader] Worker " << wk.pid << " terminated unexpectedly\n";
                    stop_prefetch_ = true;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // prefetch thread
    void prefetch_loop() {
        size_t batch_idx = 0;
        size_t worker_idx = 0;
        auto n_workers = workers_.size();

        while (batch_idx < total_batches_ && !stop_prefetch_) {
            size_t start = batch_idx * batch_size_;
            size_t end = start + batch_size_;
            if (end > indices_.size()) {
                end = indices_.size();
                if (drop_last_) break;
            }

            std::vector<uint64_t> task_indices(indices_.begin() + start, indices_.begin() + end);
            auto task_msg = serialize_task(task_indices);

            auto& wk = workers_[worker_idx];
            robust_write(wk.task_fd, task_msg.data(), task_msg.size());

            // 🔥 worker 가 보낸 descriptor 수신
            uint64_t num_items = read_uint64(wk.result_fd);
            if (num_items == 0) {
                stop_prefetch_ = true;
                break;
            }

            std::vector<ArrayDesc> descs;
            std::vector<void*> shm_ptrs;
            for (size_t i = 0; i < num_items; ++i) {
                auto key = read_string(wk.result_fd);
                auto dtype = read_string(wk.result_fd);

                auto ndim = read_uint64(wk.result_fd);
                std::vector<uint64_t> shape(ndim);
                std::vector<uint64_t> strides(ndim);
                for (size_t j = 0; j < ndim; ++j) shape[j] = read_uint64(wk.result_fd);
                for (size_t j = 0; j < ndim; ++j) strides[j] = read_uint64(wk.result_fd);

                auto total_bytes = read_uint64(wk.result_fd);

                // 🔥 parent 가 shared memory 직접 할당
                std::string shm_name;
                void* shm_ptr = MemoryPool::instance().allocate(total_bytes, shm_name);
                shm_ptrs.push_back(shm_ptr); 
                robust_read(wk.result_fd, shm_ptr, total_bytes);

                ArrayDesc d;
                d.key = key;
                d.dtype = dtype;
                d.shape = shape;
                d.strides = strides;
                d.shm_name = shm_name;
                d.size = total_bytes;
                d.ptr = shm_ptr;

                descs.push_back(d);
            }

            // serialize batch metadata
            auto batch_msg = serialize_batch(descs);

            {
                std::lock_guard<std::mutex> lk(q_mutex_);
                prefetch_queue_.push(std::make_pair(batch_msg, shm_ptrs));
            }
            q_cv_.notify_one();

            batch_idx++;
            worker_idx = (worker_idx + 1) % n_workers;
        }
    }



public:
    size_t batch_size_;
    bool drop_last_;
private:
    // construction params
    py::object reader_;
    size_t dataset_len_;
    size_t num_workers_;
    bool shuffle_;
    bool persistent_workers_;
    size_t prefetch_count_;

    // data index
    std::vector<size_t> indices_;
    size_t total_batches_;
    size_t current_batch_;

    // worker
    std::vector<Worker> workers_;
    std::thread prefetch_thread_;
    std::thread monitor_thread_;
    std::atomic<bool> stop_prefetch_;
    bool stopped_ = false;

    // prefetch queue
    std::queue<std::pair<std::vector<char>, std::vector<void*>>> prefetch_queue_;
    std::mutex q_mutex_;
    std::condition_variable q_cv_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Python 바인딩
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(FastDataLoader, m) {
    m.doc() = "FastDataLoader with persistent_workers / reset / multi-epoch support";
    py::class_<FastDataLoader>(m, "FastDataLoader")
        .def(py::init<py::object, size_t, size_t, size_t, bool, bool, bool, size_t>(),
             py::arg("reader"),
             py::arg("dataset_len"),
             py::arg("batch_size"),
             py::arg("num_workers"),
             py::arg("shuffle"),
             py::arg("drop_last"),
             py::arg("persistent_workers"),
             py::arg("prefetch_count") = 2
        )
        .def("__iter__", &FastDataLoader::__iter__, py::return_value_policy::reference_internal)
        .def("__next__", &FastDataLoader::__next__)
        .def("reset", &FastDataLoader::reset)
        .def("shutdown", &FastDataLoader::shutdown)
        .def_property_readonly("batch_size", [](const FastDataLoader &self) { return self.batch_size_; })
        .def_property_readonly("drop_last", [](const FastDataLoader &self) { return self.drop_last_; });
}
