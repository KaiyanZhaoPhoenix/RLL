import numpy as np
from typing import Any

class SumTree:
    """
    用于优先经验回放的求和树数据结构
    - leaf_size 个叶节点存储优先级
    - 内部节点存储子节点的和
    """
    def __init__(self, leaf_size: int):
        self.leaf_size = leaf_size
        # 为完整二叉树分配内存
        # 树的总大小是 2 * leaf_size - 1
        self.tree = np.zeros(2 * leaf_size - 1)
        # 存储经验数据的数组
        self.data = np.zeros(leaf_size, dtype=object)
        # 下一个要写入的位置
        self.write = 0
        # 当前存储的数据量
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """向上传播节点更新"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> tuple[int, float]:
        """找到对应和为s的叶节点"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx, self.tree[idx]

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """返回所有优先级的总和"""
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """添加新数据和其优先级"""
        idx = self.write + self.leaf_size - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.leaf_size:
            self.write = 0

        if self.size < self.leaf_size:
            self.size += 1

    def update(self, idx: int, priority: float) -> None:
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, Any]:
        """获取优先级总和为s的数据"""
        idx, priority = self._retrieve(0, s)
        data_idx = idx - self.leaf_size + 1
        
        # 确保数据索引在有效范围内
        data_idx = data_idx % self.leaf_size
        
        return idx, priority, self.data[data_idx]

    def get_leaf(self, idx: int) -> tuple[float, Any]:
        """直接通过索引获取叶节点数据"""
        tree_idx = idx + self.leaf_size - 1
        return self.tree[tree_idx], self.data[idx]