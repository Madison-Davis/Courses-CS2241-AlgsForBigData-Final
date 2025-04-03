# segment_tree.py
import numpy as np

class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        """
        A generic Segment Tree.
        Args:
            capacity (int): size of the array (ideally a power of 2).
            operation (callable): a binary operation (e.g., sum or min).
            neutral_element: the identity for the operation.
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        # Tree is stored as a flat array of size 2 * capacity.
        self.tree = np.full(2 * capacity, neutral_element, dtype=np.float64)

    def update(self, idx, value):
        """Updates the leaf at index `idx` with `value` and propagates the change."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = value
        while tree_idx > 1:
            tree_idx //= 2
            left = self.tree[2 * tree_idx]
            right = self.tree[2 * tree_idx + 1]
            self.tree[tree_idx] = self.operation(left, right)

    def query(self, start, end):
        """
        Queries in the interval [start, end).
        Returns the reduction of that interval.
        """
        start += self.capacity
        end += self.capacity
        result = self.neutral_element
        while start < end:
            if start % 2 == 1:
                result = self.operation(result, self.tree[start])
                start += 1
            if end % 2 == 1:
                end -= 1
                result = self.operation(result, self.tree[end])
            start //= 2
            end //= 2
        return result

    def find_prefixsum_idx(self, prefixsum):
        """
        Finds the highest index i in the tree such that the cumulative sum
        of [0, i] is greater than or equal to prefixsum.
        """
        idx = 1
        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            if self.tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operation=lambda a, b: a + b, neutral_element=0.0)

    def sum(self):
        return self.tree[1]

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operation=lambda a, b: min(a, b), neutral_element=float('inf'))
