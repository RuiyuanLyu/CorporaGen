import numpy as np


def check_bboxes_visibility(bboxes, depth_map, depth_intrinsic, extrinsic_w2c):
    """
        Check the visibility of 3D bounding boxes in a depth map.
        Args:
            bboxes: a list of N open3d.geometry.OrientedBoundingBox
            depth_map: depth map, numpy array of shape (h, w).
            depth_intrinsic: numpy array of shape (4, 4).
            extrinsic_w2c: numpy array of shape (4, 4).
        Returns:
            Boolean array of shape (N, ) indicating the visibility of each bounding box.
    """
    corners = [box.get_box_points() for box in bboxes]
    corners = np.concatenate(corners, axis=0) # shape (N*8, 3)
    assert corners.shape[1] == 3 and len(corners.shape) == 2
    visibles = check_point_visibility(corners, depth_map, depth_intrinsic, extrinsic_w2c)
    visibles = visibles.reshape(len(bboxes), 8)
    visibles = np.sum(visibles, axis=1) >= 1 # at least some corners visible
    return visibles

def check_point_visibility(points, depth_map, depth_intrinsic, extrinsic_w2c):
    """
        Check the visibility of 3D points in a depth map.
        Args:
            points: 3D points, numpy array of shape (n, 3).
            depth_map: depth map, numpy array of shape (h, w).
            depth_intrinsic: numpy array of shape (4, 4).
            extrinsic_w2c: numpy array of shape (4, 4).
        Returns:
            Boolean array of shape (n, ) indicating the visibility of each point.
    """
    # Project 3D points to 2D image plane
    visibles = np.ones(points.shape[0], dtype=bool)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1) # shape (n, 4)
    points = depth_intrinsic @ extrinsic_w2c @ points.T # (4, n)
    xs, ys, zs = points[:3, :]
    visibles &= (zs > 0) # remove points behind the camera
    xs, ys = xs / zs, ys / zs # normalize to image plane
    height, width = depth_map.shape
    visibles &= (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height) # remove points outside the image
    xs[(xs < 0) | (xs >= width)] = 0 # avoid index out of range in depth_map
    ys[(ys < 0) | (ys >= height)] = 0
    visibles &= (depth_map[ys.astype(int), xs.astype(int)] > zs) # remove points occluded by other objects
    return visibles

def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
        Convert rotations given as Euler angles in radians to rotation matrices.

        Args:
            euler_angles: Euler angles in radians as array of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrices as array of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, np.split(euler_angles, 3, axis=-1))
    ]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])
