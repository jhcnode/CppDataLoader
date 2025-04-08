import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import CppDataLoader as core_loader # C++ 모듈 (setup.py 빌드 후 생성됨)
from tqdm import tqdm


# 가짜 Dataset 시뮬레이션
class DummyDataset:
    def __getitem__(self, idx):
        image = np.ones((3, 128, 128), dtype=np.float32) * idx
        label = np.zeros(10, dtype=np.float32)
        label[idx % 10] = 1.0
        return {"image": image, "label": label}

    def __len__(self):
        return 1000

# C++ 모듈에서 호출할 reader 함수 정의
def reader(idx):
    return dataset[idx]

# C++ 모듈의 CppDataLoader 객체를 감싸는 wrapper 클래스
class CppDataLoader:
    def __init__(self, reader, dataset_len, batch_size, num_workers, shuffle, drop_last,persistent_workers,prefetch_count):
        self.loader = core_loader.CppDataLoader(
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

    def __iter__(self):
        self._batch_idx = 0  # 반복 시작 시 초기화
        return self

    def __next__(self):
        if self._batch_idx >= self.total_batches:
            raise StopIteration
        batch = self.loader()  # C++ 측 __call__()
        if not batch:
            return batch
        self._batch_idx += 1
        return batch
    
dataset = DummyDataset()
torch_dataset = DummyDataset()

# # Wrapper 객체 생성
# # === C++ DataLoader (our new prefetch version) ===
cpp_loader = CppDataLoader(
    reader=reader,
    dataset_len=len(dataset),
    batch_size=100,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_count=2
)

torch_loader = DataLoader(
    torch_dataset,
    batch_size=100,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    prefetch_factor=2,
    persistent_workers=True
)

##############################
# 시간 비교
##############################
# C++ DataLoader 시간 측정
trial=10


cpp_total_time=0
for _ in tqdm(range(trial),desc="cpp-dataloader prog"):
    start_time = time.perf_counter()
    cpp_batches = []
    for i, batch in enumerate(cpp_loader):
        cpp_batches.append(batch)
    cpp_time = time.perf_counter() - start_time
    cpp_total_time+=cpp_time

# # PyTorch DataLoader 시간 측정
torch_total_time=0
for _ in tqdm(range(trial),desc="torch-dataloader prog"):
    start_time = time.perf_counter()
    torch_batches = []
    for i, batch in enumerate(torch_loader):
        torch_batches.append(batch)
    torch_time = time.perf_counter() - start_time
    torch_total_time+=torch_time

print("CppDataLoader(proposed) elapsed avg time(trial:{}): {:.6f} seconds".format(trial,cpp_total_time/trial))
print("PyTorch DataLoader elapsed avg time(trial:{}): {:.6f} seconds".format(trial,torch_total_time/trial)) 
