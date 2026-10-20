from typing import Sequence, Tuple

import numpy as np


# Guards floor/ceil against float noise such as 0.3 / 0.1 = 2.9999999999999996
_INDEX_EPSILON = 1e-6


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

    if norm == 0.0:
        raise ValueError('Invalid zero-length quaternion.')

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def transform_points(
        xyz_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        translation: Sequence[float],
) -> np.ndarray:
    """Apply target <- source transform to an N x 3 XYZ matrix."""
    transformed = xyz_matrix.astype(np.float64) @ rotation_matrix.T
    transformed += np.asarray(translation, dtype=np.float64)
    return transformed


def crop_to_workspace(
        xyz_matrix: np.ndarray,
        workspace_min: Sequence[float],
        workspace_max: Sequence[float],
) -> np.ndarray:
    """Keep only points inside the axis-aligned workspace box."""
    lower = np.asarray(workspace_min, dtype=np.float64)
    upper = np.asarray(workspace_max, dtype=np.float64)

    mask = np.all((xyz_matrix >= lower) & (xyz_matrix <= upper), axis=1)
    return xyz_matrix[mask]


def compute_grid(
        workspace_min: Sequence[float],
        workspace_max: Sequence[float],
        voxel_size: float,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Fixed grid covering the workspace box.

    The grid is aligned to multiples of voxel_size in the planning frame, so
    voxel centers lie at (index + 0.5) * voxel_size and ids do not depend on
    the captured points.

    Returns:
        min_index: integer index of the first voxel on each axis
        shape: number of voxels on each axis (nx, ny, nz)
    """
    if voxel_size <= 0.0:
        raise ValueError('voxel_size must be greater than 0.')

    lower = np.asarray(workspace_min, dtype=np.float64)
    upper = np.asarray(workspace_max, dtype=np.float64)

    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError('workspace_min and workspace_max must be [x, y, z].')

    if np.any(upper <= lower):
        raise ValueError('workspace_max must be greater than workspace_min on every axis.')

    min_index = np.floor(lower / voxel_size + _INDEX_EPSILON).astype(np.int64)
    max_index = np.ceil(upper / voxel_size - _INDEX_EPSILON).astype(np.int64)

    shape = np.maximum(max_index - min_index, 1)

    return min_index, (int(shape[0]), int(shape[1]), int(shape[2]))


def grid_description(
        min_index: np.ndarray,
        shape: Tuple[int, int, int],
        voxel_size: float,
) -> dict:
    """
    JSON-serializable description of the grid layout.

    With it any voxel can be addressed without searching the scene matrix:
        id = (ix * ny + iy) * nz + iz
        center = origin + (index + 0.5) * voxel_size

    origin is the minimum corner of the grid.
    """
    return {
        'min_index': [int(i) for i in min_index],
        'shape': [int(n) for n in shape],
        'origin': [round(float(i * voxel_size), 6) for i in min_index],
    }


def voxelize(
        xyz_matrix: np.ndarray,
        min_index: np.ndarray,
        shape: Tuple[int, int, int],
        voxel_size: float,
        min_points_per_voxel: int = 1,
) -> np.ndarray:
    """
    Convert XYZ points into a boolean occupancy grid of the given shape.

    A voxel is occupied when at least min_points_per_voxel points fall into it.
    Points outside the grid are ignored.
    """
    occupancy = np.zeros(shape, dtype=bool)

    if xyz_matrix.size == 0:
        return occupancy

    indices = np.floor(xyz_matrix / voxel_size).astype(np.int64) - min_index

    inside = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    indices = indices[inside]

    if indices.size == 0:
        return occupancy

    flat_indices = np.ravel_multi_index(indices.T, shape)
    counts = np.bincount(flat_indices, minlength=occupancy.size)

    return (counts >= max(int(min_points_per_voxel), 1)).reshape(shape)


def fill_below_occupied(occupancy: np.ndarray) -> np.ndarray:
    """
    Mark every voxel below the highest occupied voxel of its column as occupied.

    A camera looking down only sees the top surface, the space under it is
    unknown and must not be planned through.
    """
    return np.logical_or.accumulate(occupancy[:, :, ::-1], axis=2)[:, :, ::-1]


def occupancy_to_matrix(
        occupancy: np.ndarray,
        min_index: np.ndarray,
        voxel_size: float,
) -> np.ndarray:
    """
    Convert occupancy grid into the scene matrix.

    Output matrix:
    N x 5 = [id, x_center, y_center, z_center, passable]

    passable:
    1.0 = free
    0.0 = impassable / occupied

    Row order is x-major, id = (ix * ny + iy) * nz + iz.
    """
    nx, ny, nz = occupancy.shape

    ii, jj, kk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
    )

    grid_indices = np.column_stack((ii.ravel(), jj.ravel(), kk.ravel()))
    centers = (grid_indices + np.asarray(min_index) + 0.5) * voxel_size

    cube_ids = np.arange(occupancy.size, dtype=np.float64).reshape(-1, 1)
    passable = np.where(occupancy.ravel(), 0.0, 1.0).reshape(-1, 1)

    return np.hstack((cube_ids, centers, passable))


def point_to_grid_index(
        point: Sequence[float],
        min_index: np.ndarray,
        voxel_size: float,
) -> Tuple[int, int, int]:
    """Grid index of the voxel containing the point, may lie outside the grid."""
    index = np.floor(np.asarray(point, dtype=np.float64) / voxel_size).astype(np.int64) - min_index
    return int(index[0]), int(index[1]), int(index[2])


def grid_index_to_center(
        index: Sequence[int],
        min_index: np.ndarray,
        voxel_size: float,
) -> Tuple[float, float, float]:
    center = (np.asarray(index) + np.asarray(min_index) + 0.5) * voxel_size
    return float(center[0]), float(center[1]), float(center[2])


def grid_index_to_id(index: Sequence[int], shape: Tuple[int, int, int]) -> int:
    return (int(index[0]) * shape[1] + int(index[1])) * shape[2] + int(index[2])


def id_to_grid_index(voxel_id: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if voxel_id < 0 or voxel_id >= shape[0] * shape[1] * shape[2]:
        raise ValueError(
            f'Voxel id {voxel_id} is outside the grid, valid ids are '
            f'0 -> {shape[0] * shape[1] * shape[2] - 1}.'
        )

    rest, iz = divmod(int(voxel_id), shape[2])
    ix, iy = divmod(rest, shape[1])
    return ix, iy, iz
