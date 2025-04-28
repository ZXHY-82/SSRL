from collections import defaultdict
import torch, random, copy
from torch.utils.data import RandomSampler, SequentialSampler, Sampler, Dataset
import math
from typing import TypeVar, Optional, Iterator
import torch.distributed as dist



class WavBatchSampler(object):
    def __init__(self, dataset, tlen_range, shuffle=False, batch_size=1, drop_last=False):
        self.tlen_range = tlen_range
        self.batch_size = batch_size
        self.drop_last = drop_last

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def _renew(self):
        return [], random.uniform(self.tlen_range[0], self.tlen_range[1])

    def __iter__(self):
        batch, tlen = self._renew()
        for idx in self.sampler:
            batch.append((idx, tlen))
            if len(batch) == self.batch_size:
                yield batch
                batch, tlen = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        
        
class WavBalancedBatchSampler(object):
    def __init__(self, dataset, tlen_range, spk_per_batch=32, batch_size=1, drop_last=False):
        
        self.tlen_range = tlen_range
        self.spk_per_batch = spk_per_batch
        
        self.drop_last = drop_last
        self.batch_size = batch_size
        
        self.spk2idx = defaultdict(list)
        for i, (utt, _) in enumerate(dataset.wav_scp):
            spk_id = dataset.utt2label[utt]
            self.spk2idx[spk_id].append(i)
        
        self.sampler = SequentialSampler(dataset)
        self.all_idx = None
            
    def _arrange_idx(self):
        self.all_idx = []
        left_spk2idx = copy.deepcopy(self.spk2idx)
        for spk in left_spk2idx:
            random.shuffle(left_spk2idx[spk])
        utt_per_spk = self.batch_size // self.spk_per_batch
        
        while left_spk2idx:
            if len(left_spk2idx.keys()) >= self.spk_per_batch:
                for spk in random.sample(left_spk2idx.keys(), self.spk_per_batch):
                    self.all_idx += self._select_utt(left_spk2idx, spk, utt_per_spk)
            else:
                for spk in list(left_spk2idx.keys()):
                    self.all_idx += self._select_utt(left_spk2idx, spk, utt_per_spk)
                    
    def _select_utt(self, left_spk2idx, spk, n):
        if len(left_spk2idx[spk]) <= n:
            return left_spk2idx.pop(spk)
        else:
            return [left_spk2idx[spk].pop() for i in range(n)]
        
    def _renew(self):
        return [], random.uniform(self.tlen_range[0], self.tlen_range[1])

    def __iter__(self):
        self._arrange_idx()
        
        batch, tlen = self._renew()
        for i in self.sampler:
            batch.append((self.all_idx[i], tlen))
            if len(batch) == self.batch_size:
                yield batch
                batch, tlen = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class WavBatchDistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, dur_range: Optional[list] = None,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.dur_range = dur_range
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        if self.dur_range is not None:
            # generate wave length
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_num = math.ceil(self.num_samples / self.batch_size)
            #tlen = torch.FloatTensor(batch_num, self.num_replicas).uniform_(self.dur_range[0], self.dur_range[1], generator=g).repeat(1, self.batch_size)            
            tlen = torch.FloatTensor(batch_num, 1).uniform_(self.dur_range[0], self.dur_range[1], generator=g).repeat(1, self.num_replicas).repeat(1, self.batch_size)
            tlen = tlen.reshape(-1).tolist()[:self.total_size]

            indices = [(i, j) for i, j in zip(indices, tlen)]

        # subsample
        if self.shuffle:
            indices = indices[self.rank:self.total_size:self.num_replicas]
        else:
            s = self.rank*self.num_samples
            indices = indices[s:s+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

        
class WavBatchDistributedSamplerTwoView(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, dur_range: Optional[list] = None,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.dur_range = dur_range
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        if self.dur_range is not None:
            # generate wave length
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_num = math.ceil(self.num_samples / self.batch_size)
            #tlen = torch.FloatTensor(batch_num, self.num_replicas).uniform_(self.dur_range[0], self.dur_range[1], generator=g).repeat(1, self.batch_size)            
            tlen = torch.FloatTensor(batch_num, 1).uniform_(self.dur_range[0], self.dur_range[1], generator=g).repeat(1, self.num_replicas).repeat(1, self.batch_size)
            tlen = tlen.reshape(-1).tolist()[:self.total_size]
            tlen2 = torch.FloatTensor(batch_num, 1).uniform_(self.dur_range[0], self.dur_range[1], generator=g).repeat(1, self.num_replicas).repeat(1, self.batch_size)
            tlen2 = tlen2.reshape(-1).tolist()[:self.total_size]

            indices = [(i, j, k) for i, j, k in zip(indices, tlen, tlen2)]

        # subsample
        if self.shuffle:
            indices = indices[self.rank:self.total_size:self.num_replicas]
        else:
            s = self.rank*self.num_samples
            indices = indices[s:s+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

