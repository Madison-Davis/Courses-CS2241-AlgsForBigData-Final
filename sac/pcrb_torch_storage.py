# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

import logging
import os
import textwrap
import warnings
from collections import OrderedDict
from copy import copy
from multiprocessing.context import get_spawning_popen
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import tensordict
import torch
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
)
from tensordict.base import _NESTED_TENSORS_AS_LISTS
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import _zip_strict
from torch import multiprocessing as mp
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torchrl._utils import _make_ordinal_device, implement_for, logger as torchrl_logger
from torchrl.data.replay_buffers.checkpointers import (
    ListStorageCheckpointer,
    StorageCheckpointerBase,
    StorageEnsembleCheckpointer,
    TensorStorageCheckpointer,
)
from torchrl.data.replay_buffers.utils import (
    _init_pytree,
    _is_int,
    INT_CLASSES,
    tree_iter,
)


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    ndim = 1
    max_size: int
    _default_checkpointer: StorageCheckpointerBase = StorageCheckpointerBase
    _rng: torch.Generator | None = None

    def __init__(
        self, max_size: int, checkpointer: StorageCheckpointerBase | None = None
    ) -> None:
        self.max_size = int(max_size)
        self.checkpointer = checkpointer

    @property
    def checkpointer(self):
        return self._checkpointer

    @checkpointer.setter
    def checkpointer(self, value: StorageCheckpointerBase | None) -> None:
        if value is None:
            value = self._default_checkpointer()
        self._checkpointer = value

    @property
    def _is_full(self):
        return len(self) == self.max_size

    @property
    def _attached_entities(self):
        # RBs that use a given instance of Storage should add
        # themselves to this set.
        _attached_entities = self.__dict__.get("_attached_entities_set", None)
        if _attached_entities is None:
            _attached_entities = set()
            self.__dict__["_attached_entities_set"] = _attached_entities
        return _attached_entities

    @abc.abstractmethod
    def set(self, cursor: int, data: Any, *, set_cursor: bool = True):
        ...

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        ...

    def dumps(self, path):
        self.checkpointer.dumps(self, path)

    def loads(self, path):
        self.checkpointer.loads(self, path)

    def attach(self, buffer: Any) -> None:
        """This function attaches a sampler to this storage.

        Buffers that read from this storage must be included as an attached
        entity by calling this method. This guarantees that when data
        in the storage changes, components are made aware of changes even if the storage
        is shared with other buffers (eg. Priority Samplers).

        Args:
            buffer: the object that reads from this storage.
        """
        self._attached_entities.add(buffer)


    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, index, value):
        """Sets values in the storage without updating the cursor or length."""
        return self.set(index, value, set_cursor=False)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __len__(self):
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def _empty(self):
        ...

    def _rand_given_ndim(self, batch_size):
        # a method to return random indices given the storage ndim
        if self.ndim == 1:
            return torch.randint(
                0,
                len(self),
                (batch_size,),
                generator=self._rng,
                device=getattr(self, "device", None),
            )
        raise RuntimeError(
            f"Random number generation is not implemented for storage of type {type(self)} with ndim {self.ndim}. "
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    @property
    def shape(self):
        if self.ndim == 1:
            return torch.Size([self.max_size])
        raise RuntimeError(
            f"storage.shape is not supported for storages of type {type(self)} when ndim > 1."
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    def _max_size_along_dim0(self, *, single_data=None, batched_data=None):
        if self.ndim == 1:
            return self.max_size
        raise RuntimeError(
            f"storage._max_size_along_dim0 is not supported for storages of type {type(self)} when ndim > 1."
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    def flatten(self):
        if self.ndim == 1:
            return self
        raise RuntimeError(
            f"storage.flatten is not supported for storages of type {type(self)} when ndim > 1."
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    def save(self, *args, **kwargs):
        """Alias for :meth:`~.dumps`."""
        return self.dumps(*args, **kwargs)


    def dump(self, *args, **kwargs):
        """Alias for :meth:`~.dumps`."""
        return self.dumps(*args, **kwargs)


    def load(self, *args, **kwargs):
        """Alias for :meth:`~.loads`."""
        return self.loads(*args, **kwargs)


    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        return state

    def __contains__(self, item):
        return self.contains(item)

    @abc.abstractmethod
    def contains(self, item):
        ...



class ListStorage(Storage):
    """A storage stored in a list.

    This class cannot be extended with PyTrees, the data provided during calls to
    :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` should be iterables
    (like lists, tuples, tensors or tensordicts with non-empty batch-size).

    Args:
        max_size (int, optional): the maximum number of elements stored in the storage.
            If not provided, an unlimited storage is created.

    """

    _default_checkpointer = ListStorageCheckpointer

    def __init__(self, max_size: int | None = None):
        if max_size is None:
            max_size = torch.iinfo(torch.int64).max
        super().__init__(max_size)
        self._storage = []

    def set(
        self,
        cursor: Union[int, Sequence[int], slice],
        data: Any,
        *,
        set_cursor: bool = True,
    ):
        if not isinstance(cursor, INT_CLASSES):
            if (isinstance(cursor, torch.Tensor) and cursor.numel() <= 1) or (
                isinstance(cursor, np.ndarray) and cursor.size <= 1
            ):
                self.set(int(cursor), data, set_cursor=set_cursor)
                return
            if isinstance(cursor, slice):
                self._storage[cursor] = data
                return
            if isinstance(
                data,
                (
                    list,
                    tuple,
                    torch.Tensor,
                    TensorDictBase,
                    *tensordict.base._ACCEPTED_CLASSES,
                    range,
                    set,
                    np.ndarray,
                ),
            ):
                for _cursor, _data in _zip_strict(cursor, data):
                    self.set(_cursor, _data, set_cursor=set_cursor)
            else:
                raise TypeError(
                    f"Cannot extend a {type(self)} with data of type {type(data)}. "
                    f"Provide a list, tuple, set, range, np.ndarray, tensor or tensordict subclass instead."
                )
            return
        else:
            if cursor > len(self._storage):
                raise RuntimeError(
                    "Cannot append data located more than one item away from "
                    f"the storage size: the storage size is {len(self)} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor >= self.max_size:
                raise RuntimeError(
                    f"Cannot append data to the list storage: "
                    f"maximum capacity is {self.max_size} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor == len(self._storage):
                self._storage.append(data)
            else:
                self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        if isinstance(index, (INT_CLASSES, slice)):
            return self._storage[index]
        else:
            return [self._storage[i] for i in index]

    def __len__(self):
        return len(self._storage)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": [
                elt if not hasattr(elt, "state_dict") else elt.state_dict()
                for elt in self._storage
            ]
        }

    def load_state_dict(self, state_dict):
        _storage = state_dict["_storage"]
        self._storage = []
        for elt in _storage:
            if isinstance(elt, torch.Tensor):
                self._storage.append(elt)
            elif isinstance(elt, (dict, OrderedDict)):
                self._storage.append(
                    TensorDict({}, []).load_state_dict(elt, strict=False)
                )
            else:
                raise TypeError(
                    f"Objects of type {type(elt)} are not supported by ListStorage.load_state_dict"
                )

    def _empty(self):
        self._storage = []

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Cannot share a storage of type {type(self)} between processes."
            )
        state = super().__getstate__()
        return state

    def __repr__(self):
        return f"{self.__class__.__name__}(items=[{self._storage[0]}, ...])"

    def contains(self, item):
        if isinstance(item, int):
            if item < 0:
                item += len(self._storage)

            return 0 <= item < len(self._storage)
        if isinstance(item, torch.Tensor):
            return torch.tensor(
                [self.contains(elt) for elt in item.tolist()],
                dtype=torch.bool,
                device=item.device,
            ).reshape_as(item)
        raise NotImplementedError(f"type {type(item)} is not supported yet.")
