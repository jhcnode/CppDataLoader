import numpy as np
import FastDataLoader as core_loader  # C++ 모듈 (setup.py 빌드 후 생성됨)
from tqdm import tqdm

# 가짜 Dataset 시뮬레이션
class DummyDataset:
    def __getitem__(self, idx):
        image = np.ones((3, 128, 128), dtype=np.float32) * idx
        label = np.zeros(10, dtype=np.float32)
        label[idx % 10] = 1.0
        return {"img": image, "label": label}

    def __len__(self):
        return 1000

dataset = DummyDataset()

# C++ 모듈에서 호출할 reader 함수 정의
def reader(idx):
    return dataset[idx]

# C++ 모듈의 FastDataLoader 객체를 감싸는 wrapper 클래스
class FastDataLoaderWrapper:
    def __init__(self, reader, dataset_len, batch_size, num_workers, shuffle, drop_last, persistent_workers,prefetch_count):
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

    def __iter__(self):
        self._batch_idx = 0  # 반복 시작 시 초기화
        return self

    def __next__(self):
        if self._batch_idx >= self.total_batches:
            raise StopIteration
        batch = self.loader()  # C++ 측 __call__() 호출
        self._batch_idx += 1
        return batch

# Wrapper 객체 생성
loader = FastDataLoaderWrapper(
    reader=reader,
    dataset_len=len(dataset),
    batch_size=100,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_count=2
)

# for 루프로 배치 데이터 읽기
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    for i, batch in enumerate(loader):
        pass
        #print(f"Epoch {epoch} -> Batch {i}")

