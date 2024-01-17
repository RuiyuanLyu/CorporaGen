import cv2
import numpy as np
import json
import open3d as o3d
from utils_3d import euler_angles_to_matrix
from utils_read import read_bboxes_json, read_intrinsic
from visualization import visualize_object_types_on_sam_image


def get_mapping_sam_id_to_object_id(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, depth_intrinsic_path, extrinsic_path, extra_extrinsic_path, object_json_path, return_sam_id_to_bbox_idx=False):
    """
        Derives SAM point cloud, then matche SAM ids with bboxes, and returns the idx mapping
        Args:
            sam_img_path: path to SAM image
            depth_img_path: path to depth image
            sam_json_path: path to SAM json
            intrinsic_path: path to intrinsic
            depth_intrinsic_path: path to depth intrinsic
            extrinsic_path: path to extrinsic, camera to world'
            extra_extrinsic_path: path to extra extrinsic, world' to world
            object_json_path: path to object json, containing bboxes, object ids, and object types
        Returns:
            sam_id_to_object_id: dict, mapping from SAM ids to object ids
            sam_id_to_bbox_idx: (optional) dict, mapping from SAM ids to bbox indices

    """
    sam_id_to_bbox_idx = dict()
    sam_id_to_object_id = dict()
    group_of_sam_pcds, sam_id_to_pcd_idx = generate_sam_pointcloud(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, depth_intrinsic_path, extrinsic_path, extra_extrinsic_path, return_idx_mapping=True)
    bboxes, object_ids, object_types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    sam_pcd_to_bbox_idx = match_grouped_points_with_bboxes(group_of_sam_pcds, bboxes)
    for sam_id, pcd_idx in sam_id_to_pcd_idx.items():
        if pcd_idx == -1:
            continue
        bbox_idx = sam_pcd_to_bbox_idx[pcd_idx]
        if bbox_idx == -1:
            continue
        sam_id_to_bbox_idx[sam_id] = bbox_idx
        sam_id_to_object_id[sam_id] = object_ids[bbox_idx]
    if return_sam_id_to_bbox_idx:
        return sam_id_to_object_id, sam_id_to_bbox_idx
    return sam_id_to_object_id


def generate_sam_pointcloud(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, depth_intrinsic_path, extrinsic_path, extra_extrinsic_path, return_idx_mapping=False):
    """
        Derives SAM point cloud from SAM image, depth image, SAM json, intrinsic, extrinsic.
        Args:
            sam_img_path: path to SAM image
            depth_img_path: path to depth image
            sam_json_path: path to SAM json
            intrinsic_path: path to intrinsic
            depth_intrinsic_path: path to depth intrinsic
            extrinsic_path: path to extrinsic, camera to world'
            extra_extrinsic_path: path to extra extrinsic, world' to world
            return_idx_mapping: bool, whether to return the mapping from SAM ids to indices in group_of_points
        Returns:
            group_of_points: list of SAM point clouds, each point cloud is a numpy array of shape (N_i, 3)
            idx_mapping: (optional) dict, mapping from SAM ids to indices in group_of_points
    """
    intrinsic = read_intrinsic(intrinsic_path) # camera to image
    rgb_cx, rgb_cy, rgb_fx, rgb_fy = intrinsic[0][2], intrinsic[1][2], intrinsic[0][0], intrinsic[1][1]
    depth_intrinsic = read_intrinsic(depth_intrinsic_path) # camera to image
    depth_cx, depth_cy, depth_fx, depth_fy = depth_intrinsic[0][2], depth_intrinsic[1][2], depth_intrinsic[0][0], depth_intrinsic[1][1]
    extrinsic = np.loadtxt(extrinsic_path) # camera to world'
    extra_extrinsic = np.load(extra_extrinsic_path) # world' to world
    extrinsic = np.matmul(extra_extrinsic, extrinsic) # camera to world
    # print("camera to world extrinsic:\n", extrinsic)
    depth = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED) # shape 480, 640
    assert depth is not None, "Failed to read depth image. Check path."
    depth = depth.astype(np.float32) / 1000.0
    with open(sam_json_path, 'r') as f:
        sam_id_json = json.load(f)
    sam_id = np.array(sam_id_json, dtype=int)  # shape 480, 640

    group_id = np.unique(sam_id)
    group_of_points = []
    count = 0
    idx_mapping = dict()
    idx_mapping[-1] = -1
    for id in group_id:
        if id == -1:
            continue
        idx_mapping[id] = count
        points = []
        us, vs = np.where(sam_id == id) # rgb coordinates
        us, vs = us.astype(float), vs.astype(float)
        us = depth_cy + (us - rgb_cy) * depth_fy / rgb_fy # convert to depth coordinates
        vs = depth_cx + (vs - rgb_cx) * depth_fx / rgb_fx # convert to depth coordinates
        us, vs = np.round(us).astype(int), np.round(vs).astype(int)
        feasible_mask = np.logical_and(us >= 0, us < depth.shape[1])
        feasible_mask = np.logical_and(feasible_mask, vs >= 0)
        feasible_mask = np.logical_and(feasible_mask, vs < depth.shape[0])
        us, vs = us[feasible_mask], vs[feasible_mask]
        ds = depth[us, vs]
        points = np.stack([vs * ds, us * ds, ds, np.ones((us.shape[0],))], axis=1) # shape (N, 4)
        points = points[ds > 0]
        if points.shape[0] == 0:
            idx_mapping[id] = -1
            continue
        xyzs = np.matmul(np.matmul(extrinsic, np.linalg.inv(depth_intrinsic)), points.transpose(1, 0)).transpose(1, 0)
        xyzs = xyzs[:,:3]
        # print("minx:%.2f, maxx:%.2f, miny:%.2f, maxy:%.2f, minz:%.2f, maxz:%.2f" % (np.min(xyzs[:,0]), np.max(xyzs[:,0]), np.min(xyzs[:,1]), np.max(xyzs[:,1]), np.min(xyzs[:,2]), np.max(xyzs[:,2]))) # print the range o
        group_of_points.append(xyzs)
                
        count += 1
    if not return_idx_mapping:
        return group_of_points
    return group_of_points, idx_mapping
    

def get_points_in_boxes_masks(points, boxes):
    """
        Args:
            points: a numpy array of shape (N, 3), points in world coordinate system
            boxes: numpy array of bounding boxes, shape (M, 9): xyz, lwh, rpy
        Returns:
            binary_mask: a binary numpy array of shape (N, M) where each element is 1 if the corresponding point in the point cloud is within the corresponding bounding box, 0 otherwise.     
    """
    box_centers = boxes[:, :3].reshape(-1, 1, 3) # shape M, 1, 3
    box_sizes = boxes[:, 3:6].reshape(-1, 1, 3) # shape M, 1, 3
    box_euler_angles = boxes[:, 6:9]
    # NOTE: The rotation order for Euler angles in the data read from JSON (used for human annotations) is XYZ, whereas the rotation order in the data read from pickle (used for training) is ZXY.
    box_rotation_matrices = euler_angles_to_matrix(box_euler_angles, "XYZ").reshape(-1, 3, 3) # shape M, 3, 3
    points = points.reshape(1, -1, 3) # shape 1, N, 3
    translation = points - box_centers # shape M, N, 3
    transformed_points = np.matmul(np.linalg.inv(box_rotation_matrices), translation.transpose(0, 2, 1)) # shape M, 3, N
    transformed_points = transformed_points.transpose(0, 2, 1) # shape M, N, 3
    box_sizes = box_sizes.reshape(-1, 1, 3) # shape M, 1, 3
    normalized_points = transformed_points / box_sizes # shape M, N, 3
    binary_mask = np.abs(normalized_points) < 0.5 # shape M, N, 3
    binary_mask = binary_mask.all(axis=-1) # shape M, N
    binary_mask = binary_mask.transpose(1, 0) # shape N, M
    return binary_mask


def match_single_pcd_with_bboxes(points, boxes, binary_mask, threshold=0.8):
    """
        Args:
            points: a numpy array of shape (N, 3)
            boxes: numpy array of bounding boxes, shape (M, 9): xyz, lwh, rpy
            binary_mask: a binary numpy array of shape (N, M). Each element is 1 if the corresponding point in the point cloud is within the corresponding bounding box, 0 otherwise.
        Returns: -1 if no match, otherwise the index of the matched bounding box.
    """
    num_points = points.shape[0]
    num_boxes = boxes.shape[0]
    box_sizes = np.prod(boxes[:, 3:6], axis=1) # shape M
    if num_points == 0 or num_boxes == 0:
        return -1
    scores = np.sum(binary_mask, axis=0) / num_points # shape M
    # find the box with the smallest size that has a score above the threshold
    min_size = float('inf')
    min_size_idx = -1
    for i in range(num_boxes):
        if scores[i] > threshold and box_sizes[i] < min_size:
            min_size = box_sizes[i]
            min_size_idx = i
    return min_size_idx # -1 if no match


def match_grouped_points_with_bboxes(groups_of_points, bboxes):
    """
        Args:
            groups_of_points: list of point clouds (a.k.a. grouped points), each point cloud is a numpy array of shape (N_i, 3). Each group may coorespond to a object.
            bboxes: numpy array of bounding boxes, shape (M, 9): xyz, lwh, ypr
        Returns:
            matched_indices: a list of indices of the matched bounding box for each group of points. -1 if no match. Lenght of the list is the same as the number of groups (of points).
    """
    group_sizes = np.array([points.shape[0] for points in groups_of_points])
    points = np.vstack(groups_of_points) # shape N, 3 ; N = sum(N_i)
    binary_mask_ungrouped = get_points_in_boxes_masks(points, bboxes) # shape N, M
    binary_masks = []
    start_size_idx = 0
    for size in group_sizes:
        end_size_idx = start_size_idx + size
        binary_mask = binary_mask_ungrouped[start_size_idx:end_size_idx] # shape N_i, M
        binary_masks.append(binary_mask)
        start_size_idx = end_size_idx
    matched_indices = []
    for i in range(len(groups_of_points)):
        matched_index = match_single_pcd_with_bboxes(groups_of_points[i], bboxes, binary_masks[i])
        matched_indices.append(matched_index)

    return matched_indices


if __name__ == '__main__':
    # /mnt/petrelfs/share_data/maoxiaohan/data0809/xxxx
    # /mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2/posed_images/xxxx
    # /mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2/rendered_images/xxxx
    # xxxx stands for scene0000_00
    img_id = "02300"
    sam_img_path = f"./example_data/sam_2dmask/{img_id}.jpg"
    depth_img_path = f"./example_data/posed_images/{img_id}.png"
    sam_json_path = f"./example_data/sam_2dmask/{img_id}.json"
    intrinsic_path = "./example_data/posed_images/intrinsic.txt"
    depth_intrinsic_path = "./example_data/posed_images/depth_intrinsic.txt"
    extrinsic_path = f"./example_data/posed_images/{img_id}.txt"
    extra_extrinsic_path = "./example_data/label/rot_matrix.npy"
    object_json_path = "./example_data/label/main_MDJH13.json"
    output_sam_img_path = f"./{img_id}_obj_types.jpg"
    sam_id_to_object_id = get_mapping_sam_id_to_object_id(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, depth_intrinsic_path, extrinsic_path, extra_extrinsic_path, object_json_path)
    # print(sam_id_to_object_id)
    visualize_object_types_on_sam_image(sam_img_path, sam_json_path, object_json_path, sam_id_to_object_id, output_sam_img_path)
