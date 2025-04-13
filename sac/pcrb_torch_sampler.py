# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import json
import warnings
import numpy as np
import torch
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any
from tensordict import MemoryMappedTensor
from torchrl._extension import EXTENSION_WARNING
from torchrl.data.replay_buffers.storages import Storage, TensorStorage
from torchrl.data.replay_buffers.utils import _is_int, unravel_index
try:
    from torchrl._torchrl import (
        MinSegmentTreeFp32,
        MinSegmentTreeFp64,
        SumSegmentTreeFp32,
        SumSegmentTreeFp64,
    )
except ImportError:
    warnings.warn(EXTENSION_WARNING)
_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


class Sampler(ABC):
    """A generic sampler base class for composable Replay Buffers."""

    # Some samplers - mainly those without replacement -
    # need to keep track of the number of remaining batches
    _remaining_batches = int(torch.iinfo(torch.int64).max)

    # The RNG is set by the replay buffer
    _rng: torch.Generator | None = None

    @abstractmethod
    def sample(self, storage: Storage, batch_size: int) -> tuple[Any, dict]:
        ...

    def add(self, index: int) -> None:
        return

    def extend(self, index: torch.Tensor) -> None:
        return

    def update_priority(
        self,
        index: int | torch.Tensor,
        priority: float | torch.Tensor,
        *,
        storage: Storage | None = None,
    ) -> dict | None:
        warnings.warn(
            f"Calling update_priority() on a sampler {type(self).__name__} that is not prioritized. Make sure this is the indented behavior."
        )
        return

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        return

    @property
    def default_priority(self) -> float:
        return 1.0

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...

    @property
    def ran_out(self) -> bool:
        # by default, samplers never run out
        return False

    @abstractmethod
    def _empty(self):
        ...

    @abstractmethod
    def dumps(self, path):
        ...

    @abstractmethod
    def loads(self, path):
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        return state


class PrioritizedSampler(Sampler):
    """Prioritized sampler for replay buffer.
    Presented in "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay." (https://arxiv.org/abs/1511.05952)
    Args:
        max_capacity (int): maximum capacity of the buffer.
        alpha (:obj:`float`): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (:obj:`float`): importance sampling negative exponent.
        eps (:obj:`float`, optional): delta added to the priorities to ensure that the buffer
            does not contain null priorities. Defaults to 1e-8.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectory). Can be one of "max", "min",
            "median" or "mean".
        max_priority_within_buffer (bool, optional): if ``True``, the max-priority
            is tracked within the buffer. When ``False``, the max-priority tracks
            the maximum value since the instantiation of the sampler.

    Examples:
        >>> from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, PrioritizedSampler
        >>> from tensordict import TensorDict
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(10), sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0))
        >>> priority = torch.tensor([0, 1000])
        >>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
        >>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
        >>> rb.add(data_0)
        >>> rb.add(data_1)
        >>> rb.update_priority(torch.tensor([0, 1]), priority=priority)
        >>> sample, info = rb.sample(10, return_info=True)
        >>> print(sample)
        TensorDict(
                fields={
                    action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    obs: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    priority: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
                    reward: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([10]),
                device=cpu,
                is_shared=False)
        >>> print(info)
        {'_weight': array([1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11,
               1.e-11, 1.e-11], dtype=float32), 'index': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}

    .. note:: Using a :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer` can smoothen the
        process of updating the priorities:

            >>> from torchrl.data.replay_buffers import TensorDictReplayBuffer as TDRB, LazyTensorStorage, PrioritizedSampler
            >>> from tensordict import TensorDict
            >>> rb = TDRB(
            ...     storage=LazyTensorStorage(10),
            ...     sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0),
            ...     priority_key="priority",  # This kwarg isn't present in regular RBs
            ... )
            >>> priority = torch.tensor([0, 1000])
            >>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
            >>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
            >>> data = torch.stack([data_0, data_1])
            >>> rb.extend(data)
            >>> rb.update_priority(data)  # Reads the "priority" key as indicated in the constructor
            >>> sample, info = rb.sample(10, return_info=True)
            >>> print(sample['index'])  # The index is packed with the tensordict
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    """

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
        max_priority_within_buffer: bool = False,
    ) -> None:
        if alpha < 0:
            raise ValueError(
                f"alpha must be greater or equal than 0, got alpha={alpha}"
            )
        if beta < 0:
            raise ValueError(f"beta must be greater or equal to 0, got beta={beta}")

        self._max_capacity = max_capacity
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self.reduction = reduction
        self.dtype = dtype
        self._max_priority_within_buffer = max_priority_within_buffer
        self._init()

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self._alpha}, beta={self._beta}, eps={self._eps}, reduction={self.reduction})"

    @property
    def max_size(self):
        return self._max_capacity

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Samplers of type {type(self)} cannot be shared between processes."
            )
        return super().__getstate__()

    def _init(self):
        if self.dtype in (torch.float, torch.FloatType, torch.float32):
            self._sum_tree = SumSegmentTreeFp32(self._max_capacity)
            self._min_tree = MinSegmentTreeFp32(self._max_capacity)
        elif self.dtype in (torch.double, torch.DoubleTensor, torch.float64):
            self._sum_tree = SumSegmentTreeFp64(self._max_capacity)
            self._min_tree = MinSegmentTreeFp64(self._max_capacity)
        else:
            raise NotImplementedError(
                f"dtype {self.dtype} not supported by PrioritizedSampler"
            )
        self._max_priority = None

    def _empty(self):
        self._init()

    @property
    def _max_priority(self):
        max_priority_index = self.__dict__.get("_max_priority")
        if max_priority_index is None:
            return (None, None)
        return max_priority_index

    @_max_priority.setter
    def _max_priority(self, value):
        self.__dict__["_max_priority"] = value

    def _maybe_erase_max_priority(self, index):
        if not self._max_priority_within_buffer:
            return
        max_priority_index = self._max_priority[1]
        if max_priority_index is None:
            return

        def check_index(index=index, max_priority_index=max_priority_index):
            if isinstance(index, torch.Tensor):
                # index can be 1d or 2d
                if index.ndim == 1:
                    is_overwritten = (index == max_priority_index).any()
                else:
                    is_overwritten = (index == max_priority_index).all(-1).any()
            elif isinstance(index, int):
                is_overwritten = index == max_priority_index
            elif isinstance(index, slice):
                # This won't work if called recursively
                is_overwritten = max_priority_index in range(
                    index.indices(self._max_capacity)
                )
            elif isinstance(index, tuple):
                is_overwritten = isinstance(max_priority_index, tuple)
                if is_overwritten:
                    for idx, mpi in zip(index, max_priority_index):
                        is_overwritten &= check_index(idx, mpi)
            else:
                raise TypeError(f"index of type {type(index)} is not recognized.")
            return is_overwritten

        is_overwritten = check_index()
        if is_overwritten:
            self._max_priority = None

    @property
    def default_priority(self) -> float:
        mp = self._max_priority[0]
        if mp is None:
            mp = 1
        return (mp + self._eps) ** self._alpha

    def sample(self, storage: Storage, batch_size: int) -> torch.Tensor:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        p_sum = self._sum_tree.query(0, len(storage))
        p_min = self._min_tree.query(0, len(storage))

        if p_sum <= 0:
            raise RuntimeError("non-positive p_sum")
        if p_min <= 0:
            raise RuntimeError("non-positive p_min")
        # For some undefined reason, only np.random works here.
        # All PT attempts fail, even when subsequently transformed into numpy
        if self._rng is None:
            mass = np.random.uniform(0.0, p_sum, size=batch_size)
        else:
            mass = torch.rand(batch_size, generator=self._rng) * p_sum

        # mass = torch.zeros(batch_size, dtype=torch.double).uniform_(0.0, p_sum)
        # mass = torch.rand(batch_size).mul_(p_sum)
        index = self._sum_tree.scan_lower_bound(mass)
        index = torch.as_tensor(index)
        if not index.ndim:
            index = index.unsqueeze(0)
        index.clamp_max_(len(storage) - 1)
        weight = torch.as_tensor(self._sum_tree[index])
        # get indices where weight is 0
        zero_weight = weight == 0
        index = index
        while zero_weight.any():
            index = torch.where(zero_weight, index - 1, index)
            if (index < 0).any():
                raise RuntimeError("Failed to find a suitable index")
            weight = torch.as_tensor(self._sum_tree[index])
            zero_weight = weight == 0

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = torch.pow(weight / p_min, -self._beta)
        if storage.ndim > 1:
            index = unravel_index(index, storage.shape)
        return index, {"_weight": weight}

        return index, {"_weight": weight}

    def add(self, index: torch.Tensor | int) -> None:
        super().add(index)
        self._maybe_erase_max_priority(index)

    def extend(self, index: torch.Tensor | tuple) -> None:
        super().extend(index)
        self._maybe_erase_max_priority(index)

    @torch.no_grad()
    def update_priority(
        self,
        index: int | torch.Tensor,
        priority: float | torch.Tensor,
        *,
        storage: TensorStorage | None = None,
    ) -> None:  # noqa: D417
        """Updates the priority of the data pointed by the index.

        Args:
            index (int or torch.Tensor): indexes of the priorities to be
                updated.
            priority (Number or torch.Tensor): new priorities of the
                indexed elements.

        Keyword Args:
            storage (Storage, optional): a storage used to map the Nd index size to
                the 1d size of the sum_tree and min_tree. Only required whenever
                ``index.ndim > 2``.

        """
        priority = torch.as_tensor(priority, device=torch.device("cpu")).detach()
        index = torch.as_tensor(index, dtype=torch.long, device=torch.device("cpu"))
        # we need to reshape priority if it has more than one element or if it has
        # a different shape than index
        if priority.numel() > 1 and priority.shape != index.shape:
            try:
                priority = priority.reshape(index.shape[:1])
            except Exception as err:
                raise RuntimeError(
                    "priority should be a number or an iterable of the same "
                    f"length as index. Got priority of shape {priority.shape} and index "
                    f"{index.shape}."
                ) from err
        elif priority.numel() <= 1:
            priority = priority.squeeze()

        # MaxValueWriter will set -1 for items in the data that we don't want
        # to update. We therefore have to keep only the non-negative indices.
        if _is_int(index):
            if index == -1:
                return
        else:
            if index.ndim > 1:
                if storage is None:
                    raise RuntimeError(
                        "storage should be provided to Sampler.update_priority when the storage has more "
                        "than one dimension."
                    )
                try:
                    shape = storage.shape
                except AttributeError:
                    raise AttributeError(
                        "Could not retrieve the storage shape. If your storage is not a TensorStorage subclass "
                        "or its shape isn't accessible via the shape attribute, submit an issue on GitHub."
                    )
                index = torch.as_tensor(np.ravel_multi_index(index.unbind(-1), shape))
            valid_index = index >= 0
            if not valid_index.any():
                return
            if not valid_index.all():
                index = index[valid_index]
                if priority.ndim:
                    priority = priority[valid_index]

        max_p, max_p_idx = priority.max(dim=0)
        cur_max_priority, cur_max_priority_index = self._max_priority
        if cur_max_priority is None or max_p > cur_max_priority:
            cur_max_priority, cur_max_priority_index = self._max_priority = (
                max_p,
                index[max_p_idx] if index.ndim else index,
            )
        priority = torch.pow(priority + self._eps, self._alpha)
        self._sum_tree[index] = priority
        self._min_tree[index] = priority
        if (
            self._max_priority_within_buffer
            and cur_max_priority_index is not None
            and (index == cur_max_priority_index).any()
        ):
            maxval, maxidx = torch.tensor(
                [self._sum_tree[i] for i in range(self._max_capacity)]
            ).max(0)
            self._max_priority = (maxval, maxidx)


    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        self.update_priority(index, self.default_priority, storage=storage)

    def state_dict(self) -> dict[str, Any]:
        return {
            "_alpha": self._alpha,
            "_beta": self._beta,
            "_eps": self._eps,
            "_max_priority": self._max_priority,
            "_sum_tree": deepcopy(self._sum_tree),
            "_min_tree": deepcopy(self._min_tree),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._alpha = state_dict["_alpha"]
        self._beta = state_dict["_beta"]
        self._eps = state_dict["_eps"]
        self._max_priority = state_dict["_max_priority"]
        self._sum_tree = state_dict.pop("_sum_tree")
        self._min_tree = state_dict.pop("_min_tree")

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        try:
            mm_st = MemoryMappedTensor.from_filename(
                shape=(self._max_capacity,),
                dtype=torch.float64,
                filename=path / "sumtree.memmap",
            )
            mm_mt = MemoryMappedTensor.from_filename(
                shape=(self._max_capacity,),
                dtype=torch.float64,
                filename=path / "mintree.memmap",
            )
        except FileNotFoundError:
            mm_st = MemoryMappedTensor.empty(
                (self._max_capacity,),
                dtype=torch.float64,
                filename=path / "sumtree.memmap",
            )
            mm_mt = MemoryMappedTensor.empty(
                (self._max_capacity,),
                dtype=torch.float64,
                filename=path / "mintree.memmap",
            )
        mm_st.copy_(
            torch.as_tensor([self._sum_tree[i] for i in range(self._max_capacity)])
        )
        mm_mt.copy_(
            torch.as_tensor([self._min_tree[i] for i in range(self._max_capacity)])
        )
        with open(path / "sampler_metadata.json", "w") as file:
            json.dump(
                {
                    "_alpha": self._alpha,
                    "_beta": self._beta,
                    "_eps": self._eps,
                    "_max_priority": self._max_priority,
                    "_max_capacity": self._max_capacity,
                },
                file,
            )

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "sampler_metadata.json") as file:
            metadata = json.load(file)
        self._alpha = metadata["_alpha"]
        self._beta = metadata["_beta"]
        self._eps = metadata["_eps"]
        self._max_priority = metadata["_max_priority"]
        _max_capacity = metadata["_max_capacity"]
        if _max_capacity != self._max_capacity:
            raise RuntimeError(
                f"max capacity of loaded metadata ({_max_capacity}) differs from self._max_capacity ({self._max_capacity})."
            )
        mm_st = MemoryMappedTensor.from_filename(
            shape=(self._max_capacity,),
            dtype=torch.float64,
            filename=path / "sumtree.memmap",
        )
        mm_mt = MemoryMappedTensor.from_filename(
            shape=(self._max_capacity,),
            dtype=torch.float64,
            filename=path / "mintree.memmap",
        )
        for i, elt in enumerate(mm_st.tolist()):
            self._sum_tree[i] = elt
        for i, elt in enumerate(mm_mt.tolist()):
            self._min_tree[i] = elt
