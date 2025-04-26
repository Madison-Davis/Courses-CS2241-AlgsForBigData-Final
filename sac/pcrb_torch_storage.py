from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import Storage


def get_dtype_bytes(dtype):
    """Helper to get the byte size of a given dtype."""
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float16:
        return 2
    elif dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return 1
    else:
        # Default case for other dtypes
        return torch.tensor(0, dtype=dtype).element_size()


def cast_dict():
    pass


class TieredCacheStorage(Storage):
    def __init__(self,
            max_size: int,
            transition_key_to_bytes: Dict[
                str, Dict[str, Union[torch.dtype, Tuple]]],
            keys_to_quantize: List[str],
            num_tiers: int = 3,
            tier_dtypes: Tuple[torch.dtype, ...] = (
                    torch.float32, torch.float16, torch.float8_e5m2),
            compilable: bool = False):
        """
        Initialize a tiered cache storage with different precision levels.

        Args:
            max_size: Maximum number of transitions to store
            transition_key_to_bytes: Dictionary mapping transition keys to their metadata:
                                   e.g., {'state': {'dtype': torch.float32, 'shape': (84, 84)}}
            keys_to_quantize: List of keys that should be quantized in lower precision tiers
                            If None, all keys except 'reward' and 'done' will be quantized
            num_tiers: Number of different precision tiers (default 3)
            tier_dtypes: Tuple of torch data types for each tier, from highest to lowest precision
                       Default: (torch.float32, torch.float16, torch.float8_e5m2)
            compilable: Whether to use compilable data structures
        """
        super().__init__(max_size=max_size, compilable=compilable)

        # Validate inputs
        if len(tier_dtypes) != num_tiers:
            raise ValueError(
                    f"Must provide {num_tiers} data types, one for each tier")

        # Setup tier configuration
        self.num_tiers = num_tiers
        self.tier_dtypes = tier_dtypes

        # Store transition structure
        self.transition_key_to_bytes = transition_key_to_bytes

        # Ensure all keys to quantize are in the transition
        if not all(k in transition_key_to_bytes for k in keys_to_quantize):
            raise ValueError(
                    "Some keys_to_quantize are not found in transition_key_to_bytes")
        self.keys_to_quantize = keys_to_quantize

        # Calculate memory usage per element in each tier
        tier_bytes_per_element = self._calculate_bytes_per_element()

        # Calculate tier capacities to balance memory usage
        self.tier_capacities = self._calculate_tier_capacities(max_size,
                                                               tier_bytes_per_element)
        self.tier_field_dtypes = self._create_tier_field_dtype_mapping()

        # Initialize data structures
        self.tiers = [[] for _ in range(num_tiers)]  # list of min-heaps
        self._index_map = {}  # global_idx -> (tier_idx, local_idx)
        self._reverse_index = defaultdict(
                dict)  # tier_idx -> {local_idx: global_idx}
        self._cursor = 0

        # Set max_size to be the true number of elements
        self.max_size = sum(self.tier_capacities)

    def _create_tier_field_dtype_mapping(self):
        """
        Create a mapping of data types for each field in each tier.
        This will be used for casting operations when inserting/moving elements.
        """
        tier_field_dtypes = []

        for tier_idx, tier_dtype in enumerate(self.tier_dtypes):
            tier_mapping = {}

            for key, metadata in self.transition_key_to_bytes.items():
                original_dtype = metadata['dtype']

                if tier_idx > 0 and key in self.keys_to_quantize:
                    # Use the tier's dtype for quantizable fields in lower tiers
                    tier_mapping[key] = tier_dtype
                else:
                    # Use original dtype for non-quantizable fields or tier 0
                    tier_mapping[key] = original_dtype

            tier_field_dtypes.append(tier_mapping)

        return tier_field_dtypes

    def _calculate_bytes_per_element(self):
        """
        Calculate the number of bytes used per transition in each tier.
        """
        bytes_per_element = []

        for tier_idx, tier_dtype in enumerate(self.tier_dtypes):
            total_bytes = 0

            for key, metadata in self.transition_key_to_bytes.items():
                # Get the shape and dtype information
                dtype = metadata['dtype']
                shape = metadata['shape']

                # Calculate number of elements in this field
                num_elements = 1
                for dim in shape:
                    num_elements *= dim

                # Determine which dtype to use
                if tier_idx > 0 and key in self.keys_to_quantize:
                    # Use the tier's dtype for quantizable keys in lower tiers
                    field_dtype = tier_dtype
                else:
                    # Use original dtype
                    field_dtype = dtype

                # Add bytes for this field
                total_bytes += num_elements * get_dtype_bytes(field_dtype)

            bytes_per_element.append(total_bytes)

        return bytes_per_element

    def _calculate_tier_capacities(self, max_size, tier_bytes_per_element):
        """
        Calculate capacity for each tier so they use approximately equal memory.
        The total memory used matches what would be needed for max_size elements at full precision.
        """
        # Calculate total memory available (if all elements were in tier 0)
        total_bytes = max_size * tier_bytes_per_element[0]

        # Ideal memory per tier
        target_memory_per_tier = total_bytes / self.num_tiers

        # Calculate capacities based on memory per tier and bytes per element
        capacities = [int(target_memory_per_tier / bytes_per_elem)
                      for bytes_per_elem in tier_bytes_per_element]

        # Handle any remaining memory due to integer division
        # (optional, only if you want to use 100% of the calculated memory)
        remaining_bytes = total_bytes - sum(cap * bytes for cap, bytes in
                                            zip(capacities,
                                                tier_bytes_per_element))

        if remaining_bytes > 0:
            # Prioritize adding capacity to higher precision tiers first
            for tier_idx in range(self.num_tiers):
                while remaining_bytes >= tier_bytes_per_element[tier_idx]:
                    capacities[tier_idx] += 1
                    remaining_bytes -= tier_bytes_per_element[tier_idx]

        return capacities

    def _next_index(self) -> int:
        idx = self._cursor
        self._cursor += 1
        return idx

    def quantize(self, data, to_tier: int):
        """
        Quantize transition data to the appropriate precision level for the specified tier.

        Args:
            data: The transition data dictionary
            to_tier: The tier index to quantize for

        Returns:
            Quantized version of the data appropriate for the specified tier
        """
        tier_dtypes = self.tier_field_dtypes[to_tier]

        for key, value in data.items():
            if key in self.keys_to_quantize:
                target_dtype = tier_dtypes[key]
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(target_dtype)
                else:
                    data[key] = value
            else:
                data[key] = value

        return data

    def set(self, cursor: int | Sequence[int], data: Any, *,
            set_cursor: bool = True):
        if isinstance(cursor, int):
            self._insert_single(data)
        else:
            for d in data:
                self._insert_single(d)

    def _insert_single(self, data):
        """
        Insert a single data point into the tier using a trickle down effect based on its td-error.

        Args:
            data: Dictionary containing the transition data and td_error
        """
        # Grab the td-error of the data
        td_error = data["td_error"]
        if isinstance(td_error, torch.Tensor):
            td_error = td_error.item()

        # Because heaps like unique IDs, if TD errors are same (they're serving as our index), tie-break
        # based on when it came in
        unique_id = self._cursor
        self._cursor += 1

        # Start with the original data
        current_data = (td_error, unique_id, data)

        # Try to insert data in tiers based on TD error
        for tier_idx in range(self.num_tiers):
            tier = self.tiers[tier_idx]

            # If there's space in this tier, insert and we're done
            if len(tier) < self.tier_capacities[tier_idx]:
                quantized_data = self.quantize(current_data[2], tier_idx)
                heap_entry = (current_data[0], current_data[1], quantized_data)

                # Add to heap and update indexing
                heapq.heappush(tier, heap_entry)
                self._index_map[unique_id] = (tier_idx, len(tier) - 1)
                self._reverse_index[tier_idx][len(tier) - 1] = unique_id
                current_data = None
                break

            # If the tier is full, we may need to evict an element
            min_td_error_transition = tier[0]

            # Current TD error is bigger, so we evict and trickle down
            if td_error > min_td_error_transition[0]:
                quantized_data = self.quantize(current_data[2], tier_idx)
                heap_entry = (current_data[0], current_data[1], quantized_data)

                # Replace the minimum element with our new element
                evicted_item = heapq.heapreplace(tier, heap_entry)

                # Update indexing - remove old mapping for evicted item
                evicted_id = evicted_item[1]
                if evicted_id in self._index_map:
                    old_tier_idx, old_local_idx = self._index_map.pop(
                            evicted_id)
                    if old_tier_idx in self._reverse_index and old_local_idx in \
                            self._reverse_index[old_tier_idx]:
                        del self._reverse_index[old_tier_idx][old_local_idx]

                # Add new mapping
                self._index_map[unique_id] = (
                        tier_idx, 0)  # It's at index 0 after heapreplace
                self._reverse_index[tier_idx][0] = unique_id

                # The evicted item becomes the current item for next tier
                current_data = evicted_item
                unique_id = evicted_id  # Keep the original unique ID

        # If we still have data that wasn't inserted, confirm all tiers are full
        if current_data is not None:
            # Assert that all tiers are full
            all_tiers_full = all(
                    len(self.tiers[i]) >= self.tier_capacities[i] for i in
                    range(self.num_tiers))
            assert all_tiers_full, "Failed to insert data but not all tiers are full"

            # Log that the data was discarded due to low priority
            print(
                    f"Data with td_error {current_data[0]} discarded because all tiers are full and it had lower priority")

    def _map_index(self, tier_idx: int, local_idx: int):
        global_idx = self._next_index()
        self._index_map[global_idx] = (tier_idx, local_idx)
        self._reverse_index[tier_idx][local_idx] = global_idx

    def get(self,
            index: int | Sequence[int] | torch.Tensor | TensorDict) -> Any:
        if isinstance(index, torch.Tensor) and index.ndimension() == 0:
            index = [index.item()]
        if isinstance(index, TensorDict):
            return {key: index[key] for key in index.keys()}
        if isinstance(index, int):
            tier_idx, local_idx = self._index_map.get(index, (None, None))
            tier_data = self.tiers[tier_idx]
            return tier_data[local_idx][2]
        if isinstance(index, (list, tuple, torch.Tensor)):
            if isinstance(index, torch.Tensor):
                index = index.tolist()
            results = []
            for i in index:
                result = self.get(i)
                if result is not None:
                    results.append(result)
            return results
        raise TypeError(f"Unsupported index type: {type(index)}")

    def __len__(self):
        return sum(len(tier) for tier in self.tiers)

    def contains(self, item):
        return item in self._index_map

    def state_dict(self) -> dict[str, Any]:
        return {
                "tiers": self.tiers,
                "cursor": self._cursor,
                "index_map": self._index_map,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.tiers = state_dict["tiers"]
        self._cursor = state_dict["cursor"]
        self._index_map = state_dict["index_map"]

    def _empty(self):
        self.tiers = [[] for _ in range(self.num_tiers)]
        self._index_map.clear()
        self._reverse_index.clear()
        self._cursor = 0

    def get_tier(self, tier_idx: int) -> Any:
        if tier_idx < 0 or tier_idx >= self.num_tiers:
            raise IndexError(f"Tier index {tier_idx} out of range.")
        return self.tiers[tier_idx]

    def print_tier_contents(self):
        # More for my debugging
        for i, tier in enumerate(self.tiers):
            print(f"\n--- Tier {i} ---")
            for j, (td_error, data) in enumerate(tier):
                print(
                        f"[{j}] TD Error: {td_error}, Data Keys: {list(data.keys())}")
