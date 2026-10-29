import heapq
import itertools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


GridIndex = Tuple[int, int, int]

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)

# Offsets of a point inside a voxel used for line-of-sight checks.
# Checking a thin tube around the line instead of the line itself prevents
# shortcuts that squeeze between two occupied voxels touching at an edge.
_TUBE_OFFSETS = np.array(list(itertools.product((-0.25, 0.25), repeat=3)))


def dilate_cube(occupancy: np.ndarray, radius: int) -> np.ndarray:
    """
    Mark every voxel inside a (2 * radius + 1)^3 cube around an occupied voxel.
    """
    dilated = occupancy.astype(bool, copy=True)

    for axis in range(3):
        source = dilated.copy()

        for shift in range(1, radius + 1):
            if shift >= dilated.shape[axis]:
                break

            forward = [slice(None)] * 3
            backward = [slice(None)] * 3
            forward[axis] = slice(shift, None)
            backward[axis] = slice(None, -shift)

            dilated[tuple(forward)] |= source[tuple(backward)]
            dilated[tuple(backward)] |= source[tuple(forward)]

    return dilated


class AStar3D:
    def __init__(
            self,
            occupancy: np.ndarray,
            clearance_voxels: int = 2,
            clearance_cost: float = 5.0,
    ) -> None:
        """
        A* over a boolean occupancy grid with 26-neighbour connectivity.

        occupancy:
        True = occupied, never entered
        False = free

        Free voxels within clearance_voxels of an occupied voxel (a
        (2 * clearance_voxels + 1)^3 cube) form the clearance zone. Entering a
        voxel of the clearance zone costs clearance_cost times the move length,
        so paths keep distance from obstacles where possible but can still
        reach start and goal voxels close to them.
        """
        if occupancy.ndim != 3 or occupancy.size == 0:
            raise ValueError('occupancy must be a non-empty 3D array.')

        if clearance_voxels < 0:
            raise ValueError('clearance_voxels must be 0 or greater.')

        if clearance_cost < 1.0:
            raise ValueError('clearance_cost must be 1.0 or greater.')

        self.occupancy = occupancy.astype(bool, copy=False)
        self.shape: GridIndex = tuple(int(n) for n in occupancy.shape)
        self.clearance_cost = float(clearance_cost)

        if clearance_voxels > 0:
            self.clearance_zone = dilate_cube(self.occupancy, clearance_voxels) & ~self.occupancy
        else:
            self.clearance_zone = np.zeros(self.shape, dtype=bool)

        # Pad by one occupied voxel so neighbours never leave the grid
        padded_shape = tuple(n + 2 for n in self.shape)

        blocked = np.ones(padded_shape, dtype=np.uint8)
        blocked[1:-1, 1:-1, 1:-1] = self.occupancy

        zone = np.zeros(padded_shape, dtype=np.uint8)
        zone[1:-1, 1:-1, 1:-1] = self.clearance_zone

        # Plain bytearrays are much faster to index from Python than NumPy arrays
        self._blocked = bytearray(blocked.ravel().tobytes())
        self._zone = bytearray(zone.ravel().tobytes())

        self._stride_x = padded_shape[1] * padded_shape[2]
        self._stride_y = padded_shape[2]
        self._moves = self._build_moves()

    def _to_flat(self, index: GridIndex) -> int:
        return (index[0] + 1) * self._stride_x + (index[1] + 1) * self._stride_y + (index[2] + 1)

    def _to_index(self, flat: int) -> GridIndex:
        ix, rest = divmod(flat, self._stride_x)
        iy, iz = divmod(rest, self._stride_y)
        return ix - 1, iy - 1, iz - 1

    def _build_moves(self) -> List[Tuple[int, float, Tuple[int, ...]]]:
        """
        Return (flat offset, move length, offsets that must be free) per move.

        A diagonal move is allowed only if every axis-aligned and edge move it is
        composed of is free, so the path never cuts a corner of an occupied voxel.
        """
        strides = (self._stride_x, self._stride_y, 1)
        moves = []

        for step in itertools.product((-1, 0, 1), repeat=3):
            if step == (0, 0, 0):
                continue

            axes = [axis for axis in range(3) if step[axis] != 0]
            offset = sum(step[axis] * strides[axis] for axis in axes)
            length = math.sqrt(len(axes))

            required = []
            for count in range(1, len(axes)):
                for subset in itertools.combinations(axes, count):
                    required.append(sum(step[axis] * strides[axis] for axis in subset))

            moves.append((offset, length, tuple(required)))

        return moves

    def in_grid(self, index: Sequence[int]) -> bool:
        return all(0 <= int(index[axis]) < self.shape[axis] for axis in range(3))

    def is_free(self, index: Sequence[int]) -> bool:
        return self.in_grid(index) and not self.occupancy[tuple(int(i) for i in index)]

    def cost_factor(self, index: Sequence[int]) -> float:
        if self.clearance_zone[tuple(int(i) for i in index)]:
            return self.clearance_cost
        return 1.0

    def nearest_free(self, index: Sequence[int], max_radius: int) -> Optional[GridIndex]:
        """
        Find the closest free voxel within max_radius voxels (cube).

        Higher voxels are preferred on ties. Returns None when there is none.
        """
        if not self.in_grid(index):
            return None

        index = tuple(int(i) for i in index)

        if not self.occupancy[index]:
            return index

        lower = [max(index[axis] - max_radius, 0) for axis in range(3)]
        upper = [min(index[axis] + max_radius + 1, self.shape[axis]) for axis in range(3)]

        window = self.occupancy[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        candidates = np.argwhere(~window) + np.asarray(lower)

        if candidates.size == 0:
            return None

        delta = candidates - np.asarray(index)
        distances = np.sum(delta * delta, axis=1)

        # lexsort uses the last key as primary: distance first, then higher z
        best = np.lexsort((-delta[:, 2], distances))[0]

        return tuple(int(i) for i in candidates[best])

    def _heuristic(self, flat: int, goal: GridIndex) -> float:
        """3D octile distance, exact for 26-neighbour moves in free space."""
        ix, rest = divmod(flat, self._stride_x)
        iy, iz = divmod(rest, self._stride_y)

        d = sorted((abs(ix - 1 - goal[0]), abs(iy - 1 - goal[1]), abs(iz - 1 - goal[2])))

        return SQRT3 * d[0] + SQRT2 * (d[1] - d[0]) + (d[2] - d[1])

    def plan(self, start: Sequence[int], goal: Sequence[int]) -> List[GridIndex]:
        """
        Plan a path from start to goal.

        Returns the list of grid indices from start to goal, both included,
        or an empty list when the goal cannot be reached.
        """
        start = tuple(int(i) for i in start)
        goal = tuple(int(i) for i in goal)

        for name, index in (('start', start), ('goal', goal)):
            if not self.in_grid(index):
                raise ValueError(f'{name} voxel {index} is outside the grid {self.shape}.')
            if self.occupancy[index]:
                raise ValueError(f'{name} voxel {index} is occupied.')

        start_flat = self._to_flat(start)
        goal_flat = self._to_flat(goal)

        blocked = self._blocked
        zone = self._zone
        moves = self._moves
        clearance_cost = self.clearance_cost

        closed = bytearray(len(blocked))
        g_score = {start_flat: 0.0}
        came_from = {}

        # (f, -g, voxel): on equal f the deeper node is expanded first
        open_heap = [(self._heuristic(start_flat, goal), 0.0, start_flat)]

        while open_heap:
            _, negative_g, current = heapq.heappop(open_heap)

            if closed[current]:
                continue

            if current == goal_flat:
                return self._reconstruct_path(came_from, current)

            closed[current] = 1
            current_g = -negative_g

            for offset, length, required in moves:
                neighbor = current + offset

                if blocked[neighbor] or closed[neighbor]:
                    continue

                if required and any(blocked[current + r] for r in required):
                    continue

                tentative_g = current_g + (length * clearance_cost if zone[neighbor] else length)

                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    heapq.heappush(
                        open_heap,
                        (tentative_g + self._heuristic(neighbor, goal), -tentative_g, neighbor),
                    )

        return []

    def _reconstruct_path(self, came_from: dict, current: int) -> List[GridIndex]:
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return [self._to_index(flat) for flat in path]

    def _segment_allowed(self, a: GridIndex, b: GridIndex, max_cost_factor: float) -> bool:
        """
        Check whether the straight segment from a to b may replace the path.

        It is allowed if it touches no occupied voxel and no voxel more
        expensive than max_cost_factor.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        samples = int(math.ceil(np.abs(b - a).max() * 4.0)) + 1
        t = np.linspace(0.0, 1.0, samples).reshape(-1, 1)
        points = a + t * (b - a)

        tube = (points[:, None, :] + _TUBE_OFFSETS[None, :, :]).reshape(-1, 3)
        indices = np.unique(np.rint(tube).astype(np.int64), axis=0)
        indices = np.clip(indices, 0, np.asarray(self.shape) - 1)
        indices = tuple(indices.T)

        if self.occupancy[indices].any():
            return False

        if max_cost_factor < self.clearance_cost and self.clearance_zone[indices].any():
            return False

        return True

    def simplify(self, path: List[GridIndex]) -> List[GridIndex]:
        """
        Reduce the path to the voxels where it changes direction.

        A section of the path is replaced by a straight segment only if the
        segment stays clear of obstacles and does not enter the clearance zone
        when the original section avoided it.
        """
        if len(path) <= 2:
            return list(path)

        cost_factors = [self.cost_factor(index) for index in path]
        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1

            while j > i + 1 and not self._segment_allowed(
                    path[i], path[j], max(cost_factors[i:j + 1])):
                j -= 1

            simplified.append(path[j])
            i = j

        return simplified
