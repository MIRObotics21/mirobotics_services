import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


VoxelIndex = Tuple[int, int, int]


class AStar3D:
    def __init__(self, scene_matrix: np.ndarray, voxel_size: float) -> None:
        """
        scene_matrix columns:
        [id, x, y, z, passable]

        passable:
        1.0 = free
        0.0 = occupied
        """
        if scene_matrix.size == 0:
            raise ValueError("Scene matrix is empty.")

        if scene_matrix.shape[1] != 5:
            raise ValueError("Scene matrix must have columns [id, x, y, z, passable].")

        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be greater than 0.")

        self.scene_matrix = scene_matrix
        self.voxel_size = float(voxel_size)

        self.id_to_row: Dict[int, np.ndarray] = {}
        self.id_to_index: Dict[int, VoxelIndex] = {}
        self.index_to_id: Dict[VoxelIndex, int] = {}
        self.passable_ids = set()

        self._build_maps()

    def _build_maps(self) -> None:
        for row in self.scene_matrix:
            cube_id = int(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            passable = float(row[4])

            voxel_index = self._center_to_index(x, y, z)

            self.id_to_row[cube_id] = row
            self.id_to_index[cube_id] = voxel_index
            self.index_to_id[voxel_index] = cube_id

            if passable >= 0.5:
                self.passable_ids.add(cube_id)

    def _center_to_index(self, x: float, y: float, z: float) -> VoxelIndex:
        ix = int(round(x / self.voxel_size - 0.5))
        iy = int(round(y / self.voxel_size - 0.5))
        iz = int(round(z / self.voxel_size - 0.5))
        return ix, iy, iz

    def _heuristic(self, a: VoxelIndex, b: VoxelIndex) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _neighbors_6(self, voxel_index: VoxelIndex) -> List[VoxelIndex]:
        x, y, z = voxel_index
        return [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]

    def _reconstruct_path(
        self,
        came_from: Dict[VoxelIndex, VoxelIndex],
        current: VoxelIndex,
    ) -> List[int]:
        path_indices = [current]

        while current in came_from:
            current = came_from[current]
            path_indices.append(current)

        path_indices.reverse()

        return [self.index_to_id[idx] for idx in path_indices]

    def plan(self, start_id: int, goal_id: int) -> List[List[float]]:
        if start_id not in self.id_to_index:
            raise ValueError(f"start_id {start_id} does not exist in scene.")

        if goal_id not in self.id_to_index:
            raise ValueError(f"goal_id {goal_id} does not exist in scene.")

        if start_id not in self.passable_ids:
            raise ValueError(f"start_id {start_id} is not passable.")

        if goal_id not in self.passable_ids:
            raise ValueError(f"goal_id {goal_id} is not passable.")

        start = self.id_to_index[start_id]
        goal = self.id_to_index[goal_id]

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[VoxelIndex, VoxelIndex] = {}

        g_score: Dict[VoxelIndex, float] = {
            start: 0.0
        }

        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in visited:
                continue

            if current == goal:
                path_ids = self._reconstruct_path(came_from, current)
                return self._path_ids_to_rows(path_ids)

            visited.add(current)

            for neighbor in self._neighbors_6(current):
                neighbor_id: Optional[int] = self.index_to_id.get(neighbor)

                if neighbor_id is None:
                    continue

                if neighbor_id not in self.passable_ids:
                    continue

                tentative_g_score = g_score[current] + 1.0

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score

                    f_score = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return []

    def _path_ids_to_rows(self, path_ids: List[int]) -> List[List[float]]:
        path = []

        for cube_id in path_ids:
            row = self.id_to_row[cube_id]
            path.append([
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
            ])

        return path