from typing import Dict
import numpy as np
import torch


class IDSEnvironment:

    def __init__(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.shuffle = shuffle
        self.reset()

    def reset(self) -> None:
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        self.pos = 0

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.pos + batch_size > self.n_samples:
            self.reset()
        idx = self.indices[self.pos : self.pos + batch_size]
        self.pos += batch_size
        return {"states": self.X[idx], "labels": self.y[idx]}

    def __len__(self) -> int:
        return self.n_samples
