# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import contextlib
import json
import multiprocessing
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

from functools import partial

from tensordict import (
    is_tensor_collection,
    is_tensorclass,
    LazyStackedTensorDict,
    NestedKey,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.nn.utils import _set_dispatch_td_nn_modules
from tensordict.utils import expand_as_right, expand_right
from torch import Tensor
from torch.utils._pytree import tree_map

from torchrl._utils import accept_remote_rref_udf_invocation
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    Sampler,
    SamplerEnsemble,
)
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    _stack_anything,
    ListStorage,
    Storage,
    StorageEnsemble,
)
from torchrl.data.replay_buffers.utils import (
    _is_int,
    _reduce,
    _to_numpy,
    _to_torch,
    INT_CLASSES,
    pin_memory_output,
)
from torchrl.data.replay_buffers.writers import (
    RoundRobinWriter,
    TensorDictRoundRobinWriter,
    Writer,
    WriterEnsemble,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import _InvertTransform, Transform


class ReplayBuffer:
    """
    A generic, composable replay buffer class.
    """
    def __init__(
        self,
        *,
        storage: Storage | Callable[[], Storage] | None = None,
        sampler: Sampler | Callable[[], Sampler] | None = None,
        writer: Writer | Callable[[], Writer] | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: Transform | Callable | None = None,  # noqa-F821
        transform_factory: Callable[[], Transform | Callable]
        | None = None,  # noqa-F821
        batch_size: int | None = None,
        dim_extend: int | None = None,
        checkpointer: StorageCheckpointerBase  # noqa: F821
        | Callable[[], StorageCheckpointerBase]  # noqa: F821
        | None = None,  # noqa: F821
        generator: torch.Generator | None = None,
        shared: bool = False,
        compilable: bool = None,
    ) -> None:
        self._storage = self._maybe_make_storage(storage, compilable=compilable)
        self._storage.attach(self)
        self._sampler = self._maybe_make_sampler(sampler)
        self._writer = self._maybe_make_writer(writer)
        self._writer.register_storage(self._storage)

        self._get_collate_fn(collate_fn)
        self._pin_memory = pin_memory

        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

        self.shared = shared
        self.share(self.shared)

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()

        self._transform = self._maybe_make_transform(transform, transform_factory)

        if batch_size is None and prefetch:
            raise ValueError(
                "Dynamic batch-size specification is incompatible "
                "with multithreaded sampling. "
                "When using prefetch, the batch-size must be specified in "
                "advance. "
            )
        if (
            batch_size is None
            and hasattr(self._sampler, "drop_last")
            and self._sampler.drop_last
        ):
            raise ValueError(
                "Samplers with drop_last=True must work with a predictable batch-size. "
                "Please pass the batch-size to the ReplayBuffer constructor."
            )
        self._batch_size = batch_size
        if dim_extend is not None and dim_extend < 0:
            raise ValueError("dim_extend must be a positive value.")
        self.dim_extend = dim_extend
        self._storage.checkpointer = checkpointer
        self.set_rng(generator=generator)

    def _maybe_make_storage(self, storage: Storage | Callable[[], Storage] | None, compilable) -> Storage:
        if storage is None:
            return ListStorage(max_size=1_000, compilable=compilable)
        elif isinstance(storage, Storage):
            return storage
        elif callable(storage):
            storage = storage()
        if not isinstance(storage, Storage):
            raise TypeError(
                "storage must be either a Storage or a callable returning a storage instance."
            )
        return storage

    def _maybe_make_sampler(self, sampler: Sampler | Callable[[], Sampler] | None) -> Sampler:
        if sampler is None:
            return RandomSampler()
        elif isinstance(sampler, Sampler):
            return sampler
        elif callable(sampler):
            sampler = sampler()
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "sampler must be either a Sampler or a callable returning a sampler instance."
            )
        return sampler

    def _maybe_make_writer(self, writer: Writer | Callable[[], Writer] | None) -> Writer:
        if writer is None:
            return RoundRobinWriter()
        elif isinstance(writer, Writer):
            return writer
        elif callable(writer):
            writer = writer()
        if not isinstance(writer, Writer):
            raise TypeError(
                "writer must be either a Writer or a callable returning a writer instance."
            )
        return writer

    def _maybe_make_transform(
        self,
        transform: Transform | Callable[[], Transform] | None,
        transform_factory: Callable | None,
    ) -> Transform:
        from torchrl.envs.transforms.transforms import (
            _CallableTransform,
            Compose,
            Transform,
        )

        if transform_factory is not None:
            if transform is not None:
                raise TypeError(
                    "transform and transform_factory cannot be used simultaneously"
                )
            transform = transform_factory()
        if transform is None:
            transform = Compose()
        elif not isinstance(transform, Compose):
            if not isinstance(transform, Transform) and callable(transform):
                transform = _CallableTransform(transform)
            elif not isinstance(transform, Transform):
                raise RuntimeError(
                    "transform must be either a Transform instance or a callable."
                )
            transform = Compose(transform)
        transform.eval()
        return transform

    def share(self, shared: bool = True):
        self.shared = shared
        if self.shared:
            self._write_lock = multiprocessing.Lock()
        else:
            self._write_lock = contextlib.nullcontext()

    def set_rng(self, generator):
        self._rng = generator
        self._storage._rng = generator
        self._sampler._rng = generator
        self._writer._rng = generator

    @property
    def dim_extend(self):
        return self._dim_extend

    @dim_extend.setter
    def dim_extend(self, value):
        if (
            hasattr(self, "_dim_extend")
            and self._dim_extend is not None
            and self._dim_extend != value
        ):
            raise RuntimeError(
                "dim_extend cannot be reset. Please create a new replay buffer."
            )

        if value is None:
            if self._storage is not None:
                ndim = self._storage.ndim
                value = ndim - 1
            else:
                value = 1

        self._dim_extend = value

    def _transpose(self, data):
        if is_tensor_collection(data):
            return data.transpose(self.dim_extend, 0)
        return tree_map(lambda x: x.transpose(self.dim_extend, 0), data)

    def _get_collate_fn(self, collate_fn):
        self._collate_fn = (
            collate_fn
            if collate_fn is not None
            else _get_default_collate(
                self._storage, _is_tensordict=isinstance(self, TensorDictReplayBuffer)
            )
        )

    def set_storage(self, storage: Storage, collate_fn: Callable | None = None):
        """
        Sets a new storage in the replay buffer and returns the previous storage.
        """
        prev_storage = self._storage
        self._storage = storage
        self._get_collate_fn(collate_fn)

        return prev_storage

    def set_writer(self, writer: Writer):
        """Sets a new writer in the replay buffer and returns the previous writer."""
        prev_writer = self._writer
        self._writer = writer
        self._writer.register_storage(self._storage)
        return prev_writer

    def set_sampler(self, sampler: Sampler):
        """Sets a new sampler in the replay buffer and returns the previous sampler."""
        prev_sampler = self._sampler
        self._sampler = sampler
        return prev_sampler

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    def _getattr(self, attr):
        # To access properties in remote settings, see RayReplayBuffer.write_count for instance
        return getattr(self, attr)

    @property
    def write_count(self):
        """The total number of items written so far in the buffer through add and extend."""
        return self._writer._write_count

    def __repr__(self) -> str:
        from torchrl.envs.transforms import Compose
        storage = textwrap.indent(f"storage={getattr(self, '_storage', None)}", " " * 4)
        writer = textwrap.indent(f"writer={getattr(self, '_writer', None)}", " " * 4)
        sampler = textwrap.indent(f"sampler={getattr(self, '_sampler', None)}", " " * 4)
        if getattr(self, "_transform", None) is not None and not (
            isinstance(self._transform, Compose)
            and not len(getattr(self, "_transform", None))
        ):
            transform = textwrap.indent(
                f"transform={getattr(self, '_transform', None)}", " " * 4
            )
            transform = f"\n{self._transform}, "
        else:
            transform = ""
        batch_size = textwrap.indent(
            f"batch_size={getattr(self, '_batch_size', None)}", " " * 4
        )
        collate_fn = textwrap.indent(
            f"collate_fn={getattr(self, '_collate_fn', None)}", " " * 4
        )
        return f"{self.__class__.__name__}(\n{storage}, \n{sampler}, \n{writer}, {transform}\n{batch_size}, \n{collate_fn})"

    @pin_memory_output
    def __getitem__(self, index: int | torch.Tensor | NestedKey) -> Any:
        if isinstance(index, str) or (isinstance(index, tuple) and unravel_key(index)):
            return self[:][index]
        if isinstance(index, tuple):
            if len(index) == 1:
                return self[index[0]]
            else:
                return self[:][index]
        index = _to_numpy(index)

        if self.dim_extend > 0:
            index = (slice(None),) * self.dim_extend + (index,)
            with self._replay_lock:
                data = self._storage[index]
            data = self._transpose(data)
        else:
            with self._replay_lock:
                data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        if self._transform is not None and len(self._transform):
            with data.unlock_() if is_tensor_collection(
                data
            ) else contextlib.nullcontext():
                data = self._transform(data)

        return data

    def __setitem__(self, index, value) -> None:
        if isinstance(index, str) or (isinstance(index, tuple) and unravel_key(index)):
            self[:][index] = value
            return
        if isinstance(index, tuple):
            if len(index) == 1:
                self[index[0]] = value
            else:
                self[:][index] = value
            return
        index = _to_numpy(index)

        if self._transform is not None and len(self._transform):
            value = self._transform.inv(value)

        if self.dim_extend > 0:
            index = (slice(None),) * self.dim_extend + (index,)
            with self._replay_lock:
                self._storage[index] = self._transpose(value)
        else:
            with self._replay_lock:
                self._storage[index] = value
        return

    def state_dict(self) -> dict[str, Any]:
        return {
            "_storage": self._storage.state_dict(),
            "_sampler": self._sampler.state_dict(),
            "_writer": self._writer.state_dict(),
            "_transforms": self._transform.state_dict(),
            "_batch_size": self._batch_size,
            "_rng": (self._rng.get_state().clone(), str(self._rng.device))
            if self._rng is not None
            else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._storage.load_state_dict(state_dict["_storage"])
        self._sampler.load_state_dict(state_dict["_sampler"])
        self._writer.load_state_dict(state_dict["_writer"])
        self._transform.load_state_dict(state_dict["_transforms"])
        self._batch_size = state_dict["_batch_size"]
        rng = state_dict.get("_rng")
        if rng is not None:
            state, device = rng
            rng = torch.Generator(device=device)
            rng.set_state(state)
            self.set_rng(generator=rng)

    def dumps(self, path):
        """
        Saves the replay buffer on disk at the specified path.
        """
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        self._storage.dumps(path / "storage")
        self._sampler.dumps(path / "sampler")
        self._writer.dumps(path / "writer")
        if self._rng is not None:
            rng_state = TensorDict(
                rng_state=self._rng.get_state().clone(),
                device=self._rng.device,
            )
            rng_state.memmap(path / "rng_state")

        # fall back on state_dict for transforms
        transform_sd = self._transform.state_dict()
        if transform_sd:
            torch.save(transform_sd, path / "transform.t")
        with open(path / "buffer_metadata.json", "w") as file:
            json.dump({"batch_size": self._batch_size}, file)

    def loads(self, path):
        """
        Loads a replay buffer state at the given path.
        The buffer should have matching components and be saved using :meth:`dumps`.
        """
        path = Path(path).absolute()
        self._storage.loads(path / "storage")
        self._sampler.loads(path / "sampler")
        self._writer.loads(path / "writer")
        if (path / "rng_state").exists():
            rng_state = TensorDict.load_memmap(path / "rng_state")
            rng = torch.Generator(device=rng_state.device)
            rng.set_state(rng_state["rng_state"])
            self.set_rng(rng)
        # fall back on state_dict for transforms
        if (path / "transform.t").exists():
            self._transform.load_state_dict(torch.load(path / "transform.t"))
        with open(path / "buffer_metadata.json") as file:
            metadata = json.load(file)
        self._batch_size = metadata["batch_size"]

    def save(self, *args, **kwargs):
        """Alias for :meth:`dumps`."""
        return self.dumps(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """Alias for :meth:`dumps`."""
        return self.dumps(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Alias for :meth:`loads`."""
        return self.loads(*args, **kwargs)

    def register_save_hook(self, hook: Callable[[Any], Any]):
        """Registers a save hook for the storage.

        .. note:: Hooks are currently not serialized when saving a replay buffer: they must
            be manually re-initialized every time the buffer is created.

        """
        self._storage.register_save_hook(hook)

    def register_load_hook(self, hook: Callable[[Any], Any]):
        """Registers a load hook for the storage.

        .. note:: Hooks are currently not serialized when saving a replay buffer: they must
            be manually re-initialized every time the buffer is created.

        """
        self._storage.register_load_hook(hook)

    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)
        return self._add(data)

    def _add(self, data):
        with self._replay_lock, self._write_lock:
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def _extend(self, data: Sequence) -> torch.Tensor:
        is_comp = is_compiling()
        nc = contextlib.nullcontext()
        with self._replay_lock if not is_comp else nc, self._write_lock if not is_comp else nc:
            if self.dim_extend > 0:
                data = self._transpose(data)
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        If present, the inverse transforms will be called.`

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data added to the replay buffer.

        .. warning:: :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` can have an
          ambiguous signature when dealing with lists of values, which should be interpreted
          either as PyTree (in which case all elements in the list will be put in a slice
          in the stored PyTree in the storage) or a list of values to add one at a time.
          To solve this, TorchRL makes the clear-cut distinction between list and tuple:
          a tuple will be viewed as a PyTree, a list (at the root level) will be interpreted
          as a stack of values to add one at a time to the buffer.
          For :class:`~torchrl.data.replay_buffers.ListStorage` instances, only
          unbound elements can be provided (no PyTrees).

        """
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)
        return self._extend(data)

    def update_priority(
        self,
        index: int | torch.Tensor | tuple[torch.Tensor],
        priority: int | torch.Tensor,
    ) -> None:
        if isinstance(index, tuple):
            index = torch.stack(index, -1)
        priority = torch.as_tensor(priority)
        if self.dim_extend > 0 and priority.ndim > 1:
            priority = self._transpose(priority).flatten()
            # priority = priority.flatten()
        with self._replay_lock, self._write_lock:
            self._sampler.update_priority(index, priority, storage=self.storage)

    @pin_memory_output
    def _sample(self, batch_size: int) -> tuple[Any, dict]:
        with self._replay_lock if not is_compiling() else contextlib.nullcontext():
            index, info = self._sampler.sample(self._storage, batch_size)
            info["index"] = index
            data = self._storage.get(index)
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            is_td = is_tensor_collection(data)
            with data.unlock_() if is_td else contextlib.nullcontext(), _set_dispatch_td_nn_modules(
                is_td
            ):
                data = self._transform(data)

        return data, info

    def empty(self):
        """Empties the replay buffer and reset cursor to 0."""
        self._writer._empty()
        self._sampler._empty()
        self._storage._empty()

    def sample(self, batch_size: int | None = None, return_info: bool = False) -> Any:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A batch of data selected in the replay buffer.
            A tuple containing this batch and info if return_info flag is set to True.
        """
        if (
            batch_size is not None
            and self._batch_size is not None
            and batch_size != self._batch_size
        ):
            warnings.warn(
                f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
                f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments. "
                "The batch-size provided to the sample method "
                "will prevail."
            )
        elif batch_size is None and self._batch_size is not None:
            batch_size = self._batch_size
        elif batch_size is None:
            raise RuntimeError(
                "batch_size not specified. You can specify the batch_size when "
                "constructing the replay buffer, or pass it to the sample method. "
                "Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments."
            )
        if not self._prefetch:
            result = self._sample(batch_size)
        else:
            with self._futures_lock:
                while (
                    len(self._prefetch_queue)
                    < min(self._sampler._remaining_batches, self._prefetch_cap)
                    and not self._sampler.ran_out
                ) or not len(self._prefetch_queue):
                    fut = self._prefetch_executor.submit(self._sample, batch_size)
                    self._prefetch_queue.append(fut)
                result = self._prefetch_queue.popleft().result()

        if return_info:
            out, info = result
            if getattr(self.storage, "device", None) is not None:
                device = self.storage.device
                info = tree_map(lambda x: x.to(device) if hasattr(x, "to") else x, info)
            return out, info
        return result[0]

    def mark_update(self, index: int | torch.Tensor) -> None:
        self._sampler.mark_update(index, storage=self._storage)

    def append_transform(
        self, transform: Transform, *, invert: bool = False  # noqa-F821
    ) -> ReplayBuffer:  # noqa: D417
        """Appends transform at the end.

        Transforms are applied in order when `sample` is called.

        Args:
            transform (Transform): The transform to be appended

        Keyword Args:
            invert (bool, optional): if ``True``, the transform will be inverted (forward calls will be called
                during writing and inverse calls during reading). Defaults to ``False``.

        Example:
            >>> rb = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=4)
            >>> data = TensorDict({"a": torch.zeros(10)}, [10])
            >>> def t(data):
            ...     data += 1
            ...     return data
            >>> rb.append_transform(t, invert=True)
            >>> rb.extend(data)
            >>> assert (data == 1).all()

        """
        from torchrl.envs.transforms.transforms import _CallableTransform, Transform

        if not isinstance(transform, Transform) and callable(transform):
            transform = _CallableTransform(transform)
        if invert:
            transform = _InvertTransform(transform)
        transform.eval()
        self._transform.append(transform)
        return self

    def insert_transform(
        self,
        index: int,
        transform: Transform,  # noqa-F821
        *,
        invert: bool = False,
    ) -> ReplayBuffer:  # noqa: D417
        """Inserts transform.

        Transforms are executed in order when `sample` is called.

        Args:
            index (int): Position to insert the transform.
            transform (Transform): The transform to be appended

        Keyword Args:
            invert (bool, optional): if ``True``, the transform will be inverted (forward calls will be called
                during writing and inverse calls during reading). Defaults to ``False``.

        """
        transform.eval()
        if invert:
            transform = _InvertTransform(transform)
        self._transform.insert(index, transform)
        return self

    def __iter__(self):
        if self._sampler.ran_out:
            self._sampler.ran_out = False
        if self._batch_size is None:
            raise RuntimeError(
                "Cannot iterate over the replay buffer. "
                "Batch_size was not specified during construction of the replay buffer."
            )
        while not self._sampler.ran_out or (
            self._prefetch and len(self._prefetch_queue)
        ):
            yield self.sample()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if getattr(self, "_rng", None) is not None:
            rng_state = TensorDict(
                rng_state=self._rng.get_state().clone(),
                device=self._rng.device,
            )
            state["_rng"] = rng_state
        _replay_lock = state.pop("_replay_lock", None)
        _futures_lock = state.pop("_futures_lock", None)
        if _replay_lock is not None:
            state["_replay_lock_placeholder"] = None
        if _futures_lock is not None:
            state["_futures_lock_placeholder"] = None
        return state

    def __setstate__(self, state: dict[str, Any]):
        rngstate = None
        if "_rng" in state:
            rngstate = state["_rng"]
            if rngstate is not None:
                rng = torch.Generator(device=rngstate.device)
                rng.set_state(rngstate["rng_state"])

        if "_replay_lock_placeholder" in state:
            state.pop("_replay_lock_placeholder")
            _replay_lock = threading.RLock()
            state["_replay_lock"] = _replay_lock
        if "_futures_lock_placeholder" in state:
            state.pop("_futures_lock_placeholder")
            _futures_lock = threading.RLock()
            state["_futures_lock"] = _futures_lock
        self.__dict__.update(state)
        if rngstate is not None:
            self.set_rng(rng)

    @property
    def sampler(self):
        """The sampler of the replay buffer.

        The sampler must be an instance of :class:`~torchrl.data.replay_buffers.Sampler`.

        """
        return self._sampler

    @property
    def writer(self):
        """The writer of the replay buffer.
        The writer must be an instance of :class:`~torchrl.data.replay_buffers.Writer`.

        """
        return self._writer

    @property
    def storage(self):
        """The storage of the replay buffer.

        The storage must be an instance of :class:`~torchrl.data.replay_buffers.Storage`.

        """
        return self._storage


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer.
    """
    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        storage: Storage | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: Transform | None = None,  # noqa-F821
        batch_size: int | None = None,
        dim_extend: int | None = None,
    ) -> None:
        if storage is None:
            storage = ListStorage(max_size=1_000)
        sampler = PrioritizedSampler(storage.max_size, alpha, beta, eps, dtype)
        super().__init__(
            storage=storage,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
            dim_extend=dim_extend,
        )

class TensorDictReplayBuffer(ReplayBuffer):
    """
    TensorDict-specific wrapper around the :class:`~torchrl.data.ReplayBuffer` class.
    """

    def __init__(self, *, priority_key: str = "td_error", **kwargs) -> None:
        writer = kwargs.get("writer", None)
        if writer is None:
            kwargs["writer"] = partial(
                TensorDictRoundRobinWriter, compilable=kwargs.get("compilable")
            )
        super().__init__(**kwargs)
        self.priority_key = priority_key

    def _get_priority_item(self, tensordict: TensorDictBase) -> float:
        priority = tensordict.get(self.priority_key, None)
        if self._storage.ndim > 1:
            # We have to flatten the priority otherwise we'll be aggregating
            # the priority across batches
            priority = priority.flatten(0, self._storage.ndim - 1)
        if priority is None:
            return self._sampler.default_priority
        try:
            if priority.numel() > 1:
                priority = _reduce(priority, self._sampler.reduction)
            else:
                priority = priority.item()
        except ValueError:
            raise ValueError(
                f"Found a priority key of size"
                f" {tensordict.get(self.priority_key).shape} but expected "
                f"scalar value"
            )

        if self._storage.ndim > 1:
            priority = priority.unflatten(0, tensordict.shape[: self._storage.ndim])

        return priority

    def _get_priority_vector(self, tensordict: TensorDictBase) -> torch.Tensor:
        priority = tensordict.get(self.priority_key, None)
        if priority is None:
            return torch.tensor(
                self._sampler.default_priority,
                dtype=torch.float,
                device=tensordict.device,
            ).expand(tensordict.shape[0])
        if self._storage.ndim > 1 and priority.ndim >= self._storage.ndim:
            # We have to flatten the priority otherwise we'll be aggregating
            # the priority across batches
            priority = priority.flatten(0, self._storage.ndim - 1)

        priority = priority.reshape(priority.shape[0], -1)
        priority = _reduce(priority, self._sampler.reduction, dim=1)

        if self._storage.ndim > 1:
            priority = priority.unflatten(0, tensordict.shape[: self._storage.ndim])

        return priority

    def add(self, data: TensorDictBase) -> int:
        if self._transform is not None:
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = super()._add(data)
        if index is not None:
            if is_tensor_collection(data):
                self._set_index_in_td(data, index)

            self.update_tensordict_priority(data)
        return index

    def extend(self, tensordicts: TensorDictBase) -> torch.Tensor:
        if not isinstance(tensordicts, TensorDictBase):
            raise ValueError(
                f"{self.__class__.__name__} only accepts TensorDictBase subclasses. tensorclasses "
                f"and other types are not compatible with that class. "
                "Please use a regular `ReplayBuffer` instead."
            )
        if self._transform is not None:
            tensordicts = self._transform.inv(tensordicts)
        if tensordicts is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = super()._extend(tensordicts)

        # TODO: to be usable directly, the indices should be flipped but the issue
        #  is that just doing this results in indices that are not sorted like the original data
        #  so the actually indices will have to be used on the _storage directly (not on the buffer)
        self._set_index_in_td(tensordicts, index)
        # TODO: in principle this is a good idea but currently it doesn't work + it re-writes a priority that has just been written
        # self.update_tensordict_priority(tensordicts)
        return index

    def _set_index_in_td(self, tensordict, index):
        if index is None:
            return
        if _is_int(index):
            index = torch.as_tensor(index, device=tensordict.device)
        elif index.ndim == 2 and index.shape[:1] != tensordict.shape[:1]:
            for dim in range(2, tensordict.ndim + 1):
                if index.shape[:1].numel() == tensordict.shape[:dim].numel():
                    # if index has 2 dims and is in a non-zero format
                    index = index.unflatten(0, tensordict.shape[:dim])
                    break
            else:
                raise RuntimeError(
                    f"could not find how to reshape index with shape {index.shape} to fit in tensordict with shape {tensordict.shape}"
                )
            tensordict.set("index", index)
            return
        tensordict.set("index", expand_as_right(index, tensordict))

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        if not isinstance(self._sampler, PrioritizedSampler):
            return
        if data.ndim:
            priority = self._get_priority_vector(data)
        else:
            priority = torch.as_tensor(self._get_priority_item(data))
        index = data.get("index")
        if self._storage.ndim > 1 and index.ndim == 2:
            index = index.unbind(-1)
        else:
            while index.shape != priority.shape:
                # reduce index
                index = index[..., 0]
        return self.update_priority(index, priority)

    def sample(self,batch_size: int | None = None,return_info: bool = False,include_info: bool = None,) -> TensorDictBase:
        """Samples a batch of data from the replay buffer.
        Uses Sampler to sample indices, and retrieves them from Storage.
        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A tensordict containing a batch of data selected in the replay buffer.
            A tuple containing this tensordict and info if return_info flag is set to True.
        """
        if include_info is not None:
            warnings.warn(
                "include_info is going to be deprecated soon."
                "The default behavior has changed to `include_info=True` "
                "to avoid bugs linked to wrongly preassigned values in the "
                "output tensordict."
            )

        data, info = super().sample(batch_size, return_info=True)
        is_tc = is_tensor_collection(data)
        if is_tc and not is_tensorclass(data) and include_info in (True, None):
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            for key, val in info.items():
                if key == "index" and isinstance(val, tuple):
                    val = torch.stack(val, -1)
                try:
                    val = _to_torch(val, data.device)
                    if val.ndim < data.ndim:
                        val = expand_as_right(val, data)
                    data.set(key, val)
                except RuntimeError:
                    raise RuntimeError(
                        "Failed to set the metadata (e.g., indices or weights) in the sampled tensordict within TensorDictReplayBuffer.sample. "
                        "This is probably caused by a shape mismatch (one of the transforms has probably modified "
                        "the shape of the output tensordict). "
                        "You can always recover these items from the `sample` method from a regular ReplayBuffer "
                        "instance with the 'return_info' flag set to True."
                    )
            if is_locked:
                data.lock_()
        elif not is_tc and include_info in (True, None):
            raise RuntimeError("Cannot include info in non-tensordict data")
        if return_info:
            return data, info
        return data

    @pin_memory_output
    def _sample(self, batch_size: int) -> tuple[Any, dict]:
        with self._replay_lock if not is_compiling() else contextlib.nullcontext():
            index, info = self._sampler.sample(self._storage, batch_size)
            info["index"] = index
            data = self._storage.get(index)
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            with data.unlock_(), _set_dispatch_td_nn_modules(True):
                data = self._transform(data)
        return data, info

