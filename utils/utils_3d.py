import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import eigh


def interpolate_bbox_points(bbox, granularity=0.2, return_size=False):
    """
    Get the surface points of a 3D bounding box.
    Args:
        bbox: an open3d.geometry.OrientedBoundingBox object.
        granularity: the roughly desired distance between two adjacent surface points.
        return_size: if True, return m1, m2, m3 as well.
    Returns:
        M x 3 numpy array of Surface points of the bounding box
        (m1, m2, m3): if return_size is True, return the number for each dimension.)
    """
    corners = np.array(bbox.get_box_points())
    v1, v2, v3 = (
        corners[1] - corners[0],
        corners[2] - corners[0],
        corners[3] - corners[0],
    )
    l1, l2, l3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    assert (
        np.allclose(v1.dot(v2), 0)
        and np.allclose(v2.dot(v3), 0)
        and np.allclose(v3.dot(v1), 0)
    )
    transformation_matrix = np.column_stack((v1, v2, v3))
    m1, m2, m3 = l1 / granularity, l2 / granularity, l3 / granularity
    m1, m2, m3 = int(np.ceil(m1)), int(np.ceil(m2)), int(np.ceil(m3))
    coords = np.array(
        np.meshgrid(np.arange(m1 + 1), np.arange(m2 + 1), np.arange(m3 + 1))
    ).T.reshape(-1, 3)
    condition = (
        (coords[:, 0] == 0)
        | (coords[:, 0] == m1 - 1)
        | (coords[:, 1] == 0)
        | (coords[:, 1] == m2 - 1)
        | (coords[:, 2] == 0)
        | (coords[:, 2] == m3 - 1)
    )
    surface_points = coords[condition].astype(
        "float32"
    )  # keep only the points on the surface
    surface_points /= np.array([m1, m2, m3])
    mapped_coords = surface_points @ transformation_matrix
    mapped_coords = mapped_coords.reshape(-1, 3) + corners[0]
    if return_size:
        return mapped_coords, (m1, m2, m3)
    return mapped_coords

def check_bboxes_visibility(
    bboxes, depth_map, depth_intrinsic, extrinsic, corners_only=True, granularity=0.2
):
    """
    Check the visibility of 3D bounding boxes in a depth map.
    Args:
        bboxes: a list of N open3d.geometry.OrientedBoundingBox
        depth_map: depth map, numpy array of shape (h, w).
        depth_intrinsic: numpy array of shape (4, 4).
        extrinsic: w2c. numpy array of shape (4, 4).
        corners_only: if True, only check the corners of the bounding boxes.
        granularity: the roughly desired distance between two adjacent surface points.
    Returns:
        Boolean array of shape (N, ) indicating the visibility of each bounding box.
    """
    if corners_only:
        points = [box.get_box_points() for box in bboxes]
        num_points_per_bbox = [8] * len(bboxes)
        points = np.concatenate(points, axis=0)  # shape (N*8, 3)
    else:
        points, num_points_per_bbox, num_points_to_view = [], [], []
        for bbox in bboxes:
            interpolated_points, (m1, m2, m3) = interpolate_bbox_points(
                bbox, granularity=granularity, return_size=True
            )
            num_points_per_bbox.append(interpolated_points.shape[0])
            points.append(interpolated_points)
            num_points_to_view.append(max(m1 * m2, m1 * m3, m2 * m3))
        points = np.concatenate(points, axis=0)  # shape (\sum Mi, 3)
        num_points_to_view = np.array(num_points_to_view)
    num_points_per_bbox = np.array(num_points_per_bbox)
    visibles = check_point_visibility(points, depth_map, depth_intrinsic, extrinsic)
    num_visibles = []
    left = 0
    for i, num_points in enumerate(num_points_per_bbox):
        slice_i = visibles[left : left + num_points]
        num_visibles.append(np.sum(slice_i))
        left += num_points
    num_visibles = np.array(num_visibles)
    visibles = num_visibles / num_points_to_view >= 1  # threshold for visibility
    return visibles


def check_point_visibility(points, depth_map, depth_intrinsic, extrinsic):
    """
    Check the visibility of 3D points in a depth map.
    Args:
        points: 3D points, numpy array of shape (n, 3).
        depth_map: depth map, numpy array of shape (h, w).
        depth_intrinsic: numpy array of shape (4, 4).
        extrinsic: w2c. numpy array of shape (4, 4).
    Returns:
        Boolean array of shape (n, ) indicating the visibility of each point.
    """
    # Project 3D points to 2D image plane
    visibles = np.ones(points.shape[0], dtype=bool)
    points = np.concatenate(
        [points, np.ones_like(points[..., :1])], axis=-1
    )  # shape (n, 4)
    points = depth_intrinsic @ extrinsic @ points.T  # (4, n)
    xs, ys, zs = points[:3, :]
    visibles &= zs > 0  # remove points behind the camera
    xs, ys = xs / zs, ys / zs  # normalize to image plane
    height, width = depth_map.shape
    visibles &= (
        (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
    )  # remove points outside the image
    xs[(xs < 0) | (xs >= width)] = 0  # avoid index out of range in depth_map
    ys[(ys < 0) | (ys >= height)] = 0
    visibles &= (
        depth_map[ys.astype(int), xs.astype(int)] > zs
    )  # remove points occluded by other objects
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
    assert isinstance(euler_angles, np.ndarray)
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
    matrices = [x.squeeze(axis=-3) for x in matrices]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> np.ndarray:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.arctan2(-data[..., i2], data[..., i1])
    return np.arctan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = np.arcsin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = np.arccos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return np.stack(o, -1)


box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

def cal_corners_single(center, size, rotmat):
    center = np.array(center).reshape(3)
    size = np.array(size).reshape(3)
    rotmat = np.array(rotmat).reshape(3, 3)

    relative_corners = np.array(box_corner_vertices)
    relative_corners = 2 * relative_corners - 1
    corners = relative_corners * size / 2.0
    corners = np.dot(corners, rotmat.T).reshape(-1, 3)
    corners += center
    return corners


def cal_corners(center, size, rotmat):
    center = np.array(center).reshape(-1, 3)
    size = np.array(size).reshape(-1, 3)
    rotmat = np.array(rotmat).reshape(-1, 3, 3)
    bsz = center.shape[0]

    relative_corners = np.array(box_corner_vertices)
    relative_corners = 2 * relative_corners - 1
    relative_corners = np.expand_dims(relative_corners, 1).repeat(bsz, axis=1)
    corners = relative_corners * size / 2.0
    corners = corners.transpose(1, 0, 2)
    corners = np.matmul(corners, rotmat.transpose(0, 2, 1)).reshape(-1, 8, 3)
    corners += np.expand_dims(center, 1).repeat(8, axis=1)

    if corners.shape[0] == 1:
        corners = corners[0]
    return corners


def is_inside_box(points, center, size, rotation_mat):
    """
        Check if points are inside a 3D bounding box.
        Args:
            points: 3D points, numpy array of shape (n, 3).
            center: center of the box, numpy array of shape (3, ).
            size: size of the box, numpy array of shape (3, ).
            rotation_mat: rotation matrix of the box, numpy array of shape (3, 3).
        Returns:
            Boolean array of shape (n, ) indicating if each point is inside the box.
    """
    assert points.shape[1] == 3, "points should be of shape (n, 3)"
    center = np.array(center) # n, 3
    size = np.array(size) # n, 3
    rotation_mat = np.array(rotation_mat)
    assert rotation_mat.shape == (3, 3), f"R should be shape (3,3), but got {rotation_mat.shape}"
    # pcd_local = (rotation_mat.T @ (points - center).T).T  The expressions are equivalent
    pcd_local = (points - center) @ rotation_mat # n, 3
    pcd_local = pcd_local / size * 2.0  # scale to [-1, 1] # n, 3
    pcd_local = abs(pcd_local)
    return (pcd_local[:, 0] <= 1) & (pcd_local[:, 1] <= 1) & (pcd_local[:, 2] <= 1)

def is_inside_box_open3d(points, center, size, rotation_mat):
    import open3d as o3d
    """
        We have verified that the two functions are equivalent.
        but the first one is faster.
    """
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = center
    obb.extent = size
    obb.R = rotation_mat
    points_o3d = o3d.utility.Vector3dVector(points)
    point_indices_within_box = obb.get_point_indices_within_bounding_box(points_o3d)
    ret = np.zeros(points.shape[0], dtype=bool)
    ret[point_indices_within_box] = True
    return ret

def compute_bbox_from_points_open3d(points):
    import open3d as o3d
    # 2e-4 seconds for 100 points
    # 1e-3 seconds for 1000 points
    points = np.array(points).astype(np.float32)
    assert points.shape[1] == 3, f"points should be of shape (n, 3), but got {points.shape}"
    o3dpoints = o3d.utility.Vector3dVector(points)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3dpoints)

    center = obb.center
    size = obb.extent
    rotation = obb.R

    points_to_check = center + (points - center)*(1.0-1e-6) # add a small epsilon to avoid numerical issues
    mask = is_inside_box(points_to_check, center, size, rotation)
    # mask2 = is_inside_box_open3d(points, center, size, rotation)
    # assert (mask == mask2).all(), "mask is different from mask2"
    # assert mask2.all(), (center, size, rotation)
    assert mask.all(), (center, size, rotation)

    return center, size, rotation


def compute_bbox_from_points(points):
    # 7.5e-4 seconds for 100 points
    # 1e-3 seconds for 1000 points
    hull = ConvexHull(points)
    points_on_hull = points[hull.vertices]
    center = points_on_hull.mean(axis=0)

    points_centered = points_on_hull - center
    cov_matrix = np.cov(points_centered, rowvar=False)
    eigvals, eigvecs = eigh(cov_matrix)
    rotation = eigvecs
    # donot use eigvals to compute size, but use the min max projections
    proj = points_centered @ eigvecs
    min_proj = np.min(proj, axis=0)
    max_proj = np.max(proj, axis=0)
    size = max_proj - min_proj
    shift = (max_proj + min_proj) / 2.0
    center = center + shift @ rotation.T

    return center, size, rotation

def compute_bbox_from_points_list(points_list):
    centers = []
    sizes = []
    rotations = []
    for points in points_list:
        center, size, rotation = compute_bbox_from_points(points)
        centers.append(center)
        sizes.append(size)
        rotations.append(rotation)
    return np.array(centers), np.array(sizes), np.array(rotations)

def aabb_iou(boxes1, boxes2):
    """
    Compute the aabb IoU between two sets of boxes.
    Args:
        boxes1: numpy array of shape (n, 8, 3)
        boxes2: numpy array of shape (m, 8, 3)
    """
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    min_xyz1 = np.min(boxes1, axis=1).reshape(-1, 1, 3) # n, 1, 3
    max_xyz1 = np.max(boxes1, axis=1).reshape(-1, 1, 3) # n, 1, 3
    min_xyz2 = np.min(boxes2, axis=1).reshape(1, -1, 3) # 1, m, 3
    max_xyz2 = np.max(boxes2, axis=1).reshape(1, -1, 3) # 1, m, 3
    overlap_min = np.maximum(min_xyz1, min_xyz2) # n, m, 3
    overlap_max = np.minimum(max_xyz1, max_xyz2) # n, m, 3
    assert overlap_min.shape == (n, m, 3)
    overlap_size = np.maximum(0, overlap_max - overlap_min) # n, m, 3
    assert overlap_size.shape == (n, m, 3)
    overlap_vol = np.prod(overlap_size, axis=-1) # n, m
    assert overlap_vol.shape == (n, m)
    vol1 = np.prod(max_xyz1 - min_xyz1, axis=-1).reshape(-1, 1) # n, 1
    vol2 = np.prod(max_xyz2 - min_xyz2, axis=-1).reshape(1, -1) # 1, m
    iou = overlap_vol / (vol1 + vol2 - overlap_vol) # n, m
    assert iou.shape == (n, m)
    return iou

def check_pcd_similarity(pcd1, pcd2, ths=1e-2, min_close_num=None):
    """
    check whether two point clouds are close enough.
    There might be a permulatation issue, so we use a threshold to check.
    Args:
        pcd1: np.array of shape (n, 3)
        pcd2: np.array of shape (n, 3)
    """
    if min_close_num is None:
        assert pcd1.shape == pcd2.shape, f"pcd1 and pcd2 should have the same shape, but got {pcd1.shape} and {pcd2.shape}"
    pcd1 = pcd1.astype(np.float64)
    pcd2 = pcd2.astype(np.float64)
    distances_mat = np.sqrt(np.sum((pcd1.reshape(1, -1, 3) - pcd2.reshape(-1, 1, 3))**2, axis=-1))
    distances_rowwise = np.min(distances_mat, axis=1)
    distances_colwise = np.min(distances_mat, axis=0)
    if min_close_num is None:
        return (distances_rowwise < ths).all() and (distances_colwise < ths).all()
    else:
        row_wise_close_num = (distances_rowwise < ths).sum()
        col_wise_close_num = (distances_colwise < ths).sum()
        return row_wise_close_num >= min_close_num and col_wise_close_num >= min_close_num

def corners_from_9dof(boxes):
    """
    Convert 9dof representation of boxes to corners.
    box: shape (n, 9)
    """
    centers = boxes[:, :3]
    sizes = boxes[:, 3:6]
    eulers = boxes[:, 6:9]
    rotmats = euler_angles_to_matrix(eulers, 'ZXY')
    corners = cal_corners(centers, sizes, rotmats)
    return corners

def make_batch_boxes(num, center_bias=1.0, size_bias=1.0):
    centers = np.random.rand(100, 3) * center_bias  
    sizes = np.random.rand(100, 3) * size_bias
    eulers = np.random.rand(100, 3) * np.pi - np.pi/2
    rotmats = euler_angles_to_matrix(eulers, 'ZXY')
    corners = cal_corners(centers, sizes, rotmats)
    return corners


if __name__ == '__main__':
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        corners1 = make_batch_boxes(1000)
        corners2 = make_batch_boxes(5000)
        ious = aabb_iou(corners1, corners2)
        # print(ious)