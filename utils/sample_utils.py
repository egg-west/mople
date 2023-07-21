import random
import copy
from typing import List, NamedTuple

from torch.utils.data import ConcatDataset, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

def get_dataset_name_and_kwargs_from_data_config(data_config):
    if isinstance(data_config, dict):
        name = list(data_config.keys())[0]

        # first copy the dict, then remove the size and fraction
        kwargs = copy.deepcopy(data_config[name])

        kwargs.pop("fraction", None)
        kwargs.pop("size", None)
        return name, kwargs
    else:
        return data_config, {}

def get_dataset_fractions(conf, dataset_sizes: List[int], verbose: bool = False):
    """Calculate fraction of each dataset to use per epoch when sub-sampling"""

    if verbose:
        print("Creating sampler for datasets:")

    fractions = []
    for i, data_config in enumerate(conf):
        dataset_name, _ = get_dataset_name_and_kwargs_from_data_config(data_config)
        if isinstance(data_config, dict):
            if "fraction" in data_config[dataset_name]:
                if data_config[dataset_name]["fraction"] <= 0:
                    raise ValueError("Please specify fraction as a value between 0 < fraction <= 1")
                fractions.append(min(1, data_config[dataset_name]["fraction"]))
            elif "size" in data_config[dataset_name]:
                if data_config[dataset_name]["size"] > dataset_sizes[i]:
                    raise ValueError(f"Please specify a size smaller than number of examples: {dataset_sizes[i]:,.0f}")
                fractions.append(data_config[dataset_name]["size"] / dataset_sizes[i])
            else:
                fractions.append(1)
        else:
            fractions.append(1)

        if verbose:
            print(f"{dataset_name}: {fractions[-1]:.2%} ({int(dataset_sizes[i]*fractions[-1])})")
    return fractions

class PerDatasetSampler(DistributedSampler):
    """Sampler which returns a fixed number of samples per dataset, per epoch.

    Example:

    Dataset 1 has 10,000 examples and we want 200 per epoch
    Dataset 2 has 500 examples and we want all 500 per epoch

    Epoch size will be 700 and every epoch we'll sample a different
    200 from dataset 1.

    Parameters
    ----------
    dataset_sizes : List[int]
        A list with the size of each dataset.
    dataset_size_per_epoch : List[int]
        How many examples to get from each dataset per epoch.

    Note: dataset_sizes & dataset_size_per_epoch must be in the same order.
    Further the examples in the underlying torch.utils.data.Dataset
    must per ordered as dataset_1, dataset_2, ..., dataset_n. This is fine
    if we concatenate a bunch of datasets together
    e.g. using torch.utils.data.ConcatDataset which is current behaviour.
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        dataset_size_per_epoch: List[int],
        rank: int = None,
        world_size: int = None,
        shuffle: bool = True,
        seed: int = 0,
        samples_length: List[int] = None,
    ):
        """
        if samples_length is not None, then the sampler
        will order the samples by dataset length
        with some variability across epochs
        """
        self.dataset_sizes = dataset_sizes
        self.dataset_size_per_epoch = dataset_size_per_epoch
        self.num_datasets = len(dataset_sizes)
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        if world_size == 1:
            self.rank = 0

        self.num_samples = sum(dataset_size_per_epoch)
        self.seed = seed
        self.samples_length = samples_length

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.world_size

    def __iter__(self):
        epoch_idx = []
        n = 0

        random.seed(self.epoch + self.seed)

        print(f"{len(self.dataset_sizes)=}, {len(self.dataset_size_per_epoch)}")
        for i in range(self.num_datasets):
            sampled_idx = random.sample(range(n, self.dataset_sizes[i] + n), self.dataset_size_per_epoch[i])
            n += self.dataset_sizes[i]
            epoch_idx.extend(sampled_idx)

        if self.samples_length is not None:
            # sort by samples length and in case of ties randomize
            epoch_idx = sorted(epoch_idx, key=lambda x: (self.samples_length[x], random.random()))

            if self.shuffle:
                # do some minor shuffling to avoid repeating the same order
                # but not too much to avoid too much padding
                # quasi random basically
                for i in range(0, len(epoch_idx), 200):  # this should be batch_size dependent
                    random.shuffle(epoch_idx[i : i + 200])
        else:
            if self.shuffle:
                random.shuffle(epoch_idx)

        # split epoch_idx in world_size chunks
        epoch_idx = epoch_idx[self.rank : self.num_samples : self.world_size]

        return iter(epoch_idx)

    @classmethod
    def build_sampler_from_config(cls, training_conf, datasets: List[Dataset], verbose: bool = False, *args, **kwargs):
        dataset_sizes = [len(x) for x in datasets]
        fractions = get_dataset_fractions(training_conf.datasets, dataset_sizes, verbose)
        dataset_size_per_epoch = [int(size * frac) for size, frac in zip(dataset_sizes, fractions)]
        return cls(dataset_sizes, dataset_size_per_epoch, *args, **kwargs)
