import numpy as np


def get_corners(bbox):
    """
        Compute the coordinates of a given bbox.
    """
    pass

def project_to_image_plane(points, intrinsic, extrinsic):
    """
        Project 3D points to 2D image plane.
        Args:
            points: 3D points, numpy array of shape (n, 3).
            intrinsic: extended Intrinsic matrix as array of shape (4, 4).
            extrinsic: w2c Extrinsic matrix as array of shape (4, 4).
    """
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1) # shape (n, 4)
    points = intrinsic @ np.linalg.inv(extrinsic) @ points.T # (4, n)
    points = points[:2, :] / points[2, :] # (2, n)
    return points.T # (n, 2)


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
