import logging
from abc import ABC, abstractmethod

import torch

# from data.utils.Hydrofabric import Hydrofabric

log = logging.getLogger(__name__)


class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, *args, **kwargs): #-> Hydrofabric:
        """
        Collate function with a flexible signature to allow for different inputs
        in subclasses. Implement this method in subclasses to handle specific
        data collation logic.
        """
        raise NotImplementedError
