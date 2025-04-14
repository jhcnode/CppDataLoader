import time
import numpy as np
from tqdm import tqdm
import FastDataLoader as core_loader  # C++ 모듈 (setup.py 빌드 후 생성됨)

# -------------------------------
# 가짜 Dataset 시뮬레이션
# -------------------------------
class DummyDataset:
    def __getitem__(self, idx):
        image = np.ones((3, 128, 128), dtype=np.float32) * idx
        label = np.zeros(10, dtype=np.float32)
        label[idx % 10] = 1.0
        return {"image": image, "label": label}

    def __len__(self):
        return 1000

dataset = DummyDataset()

# -------------------------------
# C++ 모듈에서 호출할 reader 함수
# -------------------------------
def reader(indices):
    items = [dataset[i] for i in indices]
    batch = {}
    for key in items[0].keys():
        batch[key] = np.stack([it[key] for it in items])
    return batch

# -------------------------------
# C++ FastDataLoader Python Wrapper
# -------------------------------
class FastDataLoader:
    def __init__(self, reader, dataset_len, batch_size, num_workers,
                 shuffle, drop_last, persistent_workers, prefetch_count):
        self.loader = core_loader.FastDataLoader(
            reader,
            dataset_len,
            batch_size,
            num_workers,
            shuffle,
            drop_last,
            persistent_workers,
            prefetch_count
        )
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 한 에포크당 배치 수 계산
        if drop_last:
            self.total_batches = dataset_len // batch_size
        else:
            self.total_batches = (dataset_len + batch_size - 1) // batch_size

        self._batch_idx = 0  # 현재 배치 인덱스

    def reset(self):
        self.loader.reset()
        self._batch_idx = 0  # batch index 도 초기화

    def __iter__(self):
        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self.total_batches:
            raise StopIteration
        batch = self.loader.__next__()  # C++ FastDataLoader 의 __next__
        self._batch_idx += 1
        return batch

    def shutdown(self):
        self.loader.shutdown()

# -------------------------------
# DataLoader 준비
# -------------------------------
cpp_loader = FastDataLoader(
    reader=reader,
    dataset_len=len(dataset),
    batch_size=100,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_count=2
)

# -------------------------------
# 시간 비교
# -------------------------------
trial = 10
cpp_total_time = 0

for epoch in tqdm(range(trial), desc="cpp-dataloader prog"):
    cpp_loader.reset()  # 반드시 매 trial 마다 reset 필요
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(cpp_loader):
        print(f"\n✅ [Epoch {epoch}] Batch {batch_idx} shapes:")
        for key, value in batch.items():
            print(f"  - {key}: {value.shape}")

    epoch_time = time.perf_counter() - start_time
    cpp_total_time += epoch_time

    print(f"⏱️  [Epoch {epoch}] Time: {epoch_time:.6f} seconds")

cpp_loader.shutdown()  # 종료 시 shutdown

# -------------------------------
# 결과 출력
# -------------------------------
avg_time = cpp_total_time / trial
print(f"\n🚀 FastDataLoader(proposed) total elapsed time (trial: {trial}): {cpp_total_time:.6f} seconds")
print(f"🚀 FastDataLoader(proposed) average time per trial: {avg_time:.6f} seconds")
