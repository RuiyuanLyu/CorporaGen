import os, sys
# os.system('conda activate open3d')
import open3d as o3d
import numpy as np
import json
from queue import Queue
import pickle
import cv2
from utils.utils_3d import euler_angles_to_matrix, check_pcd_similarity, aabb_iou, corners_from_9dof
from utils.utils_read import read_es_info, read_annotation_pickle
from utils.utils_vis import create_box_mesh, visualize_camera_extrinsics
from utils.grounding_metric import nms_with_iou_matrix
from utils.utils_vis import annotate_image_with_3dbboxes, get_o3d_obb, draw_box3d_on_img
from scipy.spatial.transform import Rotation as R


def create_box(box, color, mode='simple'):
    center = box[:3]
    size = box[3:6]
    euler = box[6:]
    R = euler_angles_to_matrix(euler, 'ZXY')
    if mode == 'linemesh':
        box_mesh = create_box_mesh(center, size, R, color)
    elif mode == 'simple':
        box_mesh = [o3d.geometry.OrientedBoundingBox(center, R, size)]
        box_mesh[0].color = color
    else:
        raise ValueError(f'Unsupported mode: {mode}')
    return box_mesh

# scene_id = 'scene0039_00'
scene_id = 'scene0094_00'
# scene_id = 'office'
# scene_id = 'restroom'
is_open_scan = not ('scene' in scene_id)
# pcd = o3d.io.read_point_cloud("visualization\demo_scenes/{scene_id}/{scene_id}_vh_clean_2.ply")
if not is_open_scan:
    pcd = o3d.io.read_triangle_mesh(f"visualization\demo_scenes/{scene_id}/{scene_id}_vh_clean_2.ply")
else:
    pcd = o3d.io.read_triangle_mesh(f"D:\Projects\corpora_local\\visualization\demo_scenes\{scene_id}\mesh.ply")

if not is_open_scan:
    es_info = read_annotation_pickle(f"splitted_infos/{scene_id}.pkl")[f'{scene_id}']
    axis_align_matrix = np.array(es_info['axis_align_matrix'])
    pcd.transform(axis_align_matrix)
else:
    data_dir = os.path.join('visualization\demo_scenes', scene_id)
    with open(os.path.join(data_dir, 'poses.txt'), 'r') as f:
        poses = f.readlines()

    axis_align_matrix = np.loadtxt(
        os.path.join(data_dir, 'axis_align_matrix.txt'))
    intrinsic = np.loadtxt(os.path.join(data_dir, 'intrinsic.txt'))
    intrinsic = intrinsic.astype(np.float32)
    info = dict(
        axis_align_matrix=axis_align_matrix,
        images=[],
        image_paths=[],
        extrinsics_c2w=[],
        depth_img_path=[],
        depth2img=dict(intrinsic=intrinsic,
                       origin=np.array([.0, .0, .5]).astype(np.float32)),
        depth_cam2img=intrinsic,
        depth_shift=1000.0,
        intrinsic=intrinsic,
    )
    n_frames = len(poses)
    data = []
    for i in range(1, n_frames):
        timestamp, x, y, z, qx, qy, qz, qw = poses[i].split()
        x, y, z, qx, qy, qz, qw = float(x), float(y), float(z), float(
            qx), float(qy), float(qz), float(qw)
        rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = rot_matrix @ [[0, 0, 1], [-1, 0, 0],
                                                 [0, -1, 0]]
        transform_matrix[:3, 3] = [x, y, z]  # CAM to NOT ALIGNED GLOBAL

        info['image_paths'].append(
            os.path.join(data_dir, 'rgb', timestamp + '.jpg'))
        info['depth_img_path'].append(
            os.path.join(data_dir, 'depth', timestamp + '.png'))
        info['extrinsics_c2w'].append(
            transform_matrix.astype(np.float32))
    es_info = info

gt_file = f"visualization\demo_scenes/{scene_id}/{scene_id}_flattened_token_positive.json"
pred_file = f"visualization\demo_scenes/{scene_id}/{scene_id}_results_100Per.json"
gt_data = json.load(open(gt_file))
pred_data = json.load(open(pred_file))
assert len(gt_data) == len(pred_data)
# gt_data[0]: {'sub_class': 'VG_Single_Attribute_Common', 'scan_id': '{scene_id}', 'target_id': [189, 179], 'distractor_ids': [], 'text': 'Find all the items with lighting fixtures coarse grained category in the room.', 'target': ['lamp', 'switch'], 'anchors': [], 'anchor_ids': [], 'tokens_positive': [[13, 65], [13, 65]], 'ID': 'VG_Single_Attribute_Common__scene0000_00__8'}
# pred_data[0]: {'bboxes_3d': [[1.1259186267852783, -1.5820398330688477, 0.4439241290092468, 0.4303528666496277, 0.38400211930274963, 1.0724506378173828, 1.6592847108840942, -0.003870368003845215, -3.1391913890838623], [0.9865046739578247, 3.311486005783081, 1.2967586517333984, 0.23403628170490265, 0.2800081968307495, 0.3285176753997803, -1.535447597503662, 0.0007756948471069336, -3.142092704772949]], 'scores_3d': [0.26313698291778564, 0.25616997480392456]}

def vis_one_case(case_id, filter_failure=True, image_identifier=None, show_gt=True, min_gts=1, max_results=100, result_idx=None, extrinsic_c2w=None):
    pred = pred_data[case_id]
    gt = gt_data[case_id]
    subclass = gt['sub_class']
    if not 'Inter_Space_OO' in subclass:
        return
    if show_gt:
        target_ids = gt['target_id']
        all_obj_ids = list(es_info['object_ids'])
        obj_inds = [all_obj_ids.index(i) for i in target_ids]
        gt_boxes = np.array(es_info['bboxes'])[obj_inds]
        num_gt = len(gt_boxes)
        if num_gt >= max_results and filter_failure:
            return
        if num_gt < min_gts:
            return
    else:
        gt_boxes = np.zeros((0, 9))
        num_gt = 0
    pred_boxs = np.array(pred['bboxes_3d'])
    pred_scores = np.array(pred['scores_3d'])
    pred_corners = corners_from_9dof(pred_boxs)
    ious = aabb_iou(pred_corners, pred_corners)
    keep_inds = nms_with_iou_matrix(ious, 0.15, pred_scores)
    if result_idx is not None:
        keep_inds = [keep_inds[result_idx]]
    pred_boxs = pred_boxs[keep_inds][:max_results]
    # if show_gt:
    #     pred_boxs = pred_boxs[:num_gt]

    centers_pred = pred_boxs[:, :3]
    if show_gt:
        centers_gt = gt_boxes[:, :3]
        if len(centers_gt) > 20:
            return
        if filter_failure and not check_pcd_similarity(centers_gt, centers_pred, 0.2, min_close_num=num_gt):
            return

    boxes = []
    mode = 'linemesh'
    gt_color = (1, 140/255, 0)
    pred_color = (0, 1, 0)
    if show_gt:
        for i in range(num_gt):
            gt_box = gt_boxes[i]
            boxes.extend(create_box(gt_box, gt_color, mode))
        for i in range(len(pred_boxs)):
            pred_box = pred_boxs[i]
            boxes.extend(create_box(pred_box, pred_color, mode))
    else:
        for i in range(len(pred_boxs)):
            pred_box = pred_boxs[i]
            boxes.extend(create_box(pred_box, pred_color, mode))
    print(case_id, gt['text'])
    if show_gt:
        print(f'GT ids: {target_ids}')
    # red for pred, green for gt
    if image_identifier is not None:
        vis_one_case_with_image(gt_boxes, pred_boxs, image_identifier, es_info)
    else:
        o3d.visualization.draw_geometries([pcd, *boxes])

def vis_one_case_with_image(gt_boxs, pred_boxs, image_identifier, es_info):
    # boxes: numpy arrays.
    image_paths = es_info['image_paths']
    image_index = get_index_from_image_identifier(image_identifier, es_info)
    image_path = image_paths[image_index]
    boxes = np.concatenate([gt_boxs, pred_boxs], axis=0)
    boxes = get_o3d_obb(boxes, mode='zxy', colors=(1, 0, 0)) # this color is useless
    object_ids = np.arange(len(boxes))
    object_types = ['gt'] * len(gt_boxs) + ['water dispenser'] * len(pred_boxs)
    try:
        intrinsics = es_info['intrinsics']
        intrinsic = intrinsics[image_index]
    except Exception as e:
        intrinsic = es_info['intrinsic']
    extrinsics_c2w = es_info['extrinsics_c2w']
    extrinsic_c2w = extrinsics_c2w[image_index]
    axis_align_matrix = es_info['axis_align_matrix']
    if is_open_scan:
        axis_align_matrix = np.identity(4)
    # extrinsic_w2c = np.linalg.inv(axis_align_matrix @ extrinsic_c2w)
    # visualize_camera_extrinsics([pcd], [extrinsic_w2c], True)
    out_path = f'visualization\demo_scenes\{scene_id}\\vis_{image_identifier}.jpg'
    os.makedirs(out_path, exist_ok=True) 
    annotate_image_with_3dbboxes(image_path, boxes, object_ids, object_types, intrinsic, extrinsic_c2w, axis_align_matrix, out_path)
    # img = cv2.imread(image_path)
    # for box in boxes:
    #     draw_box3d_on_img(img=img, box=box, color=(255, 0, 0), label=None, extrinsic_c2w=axis_align_matrix@extrinsic_c2w, intrinsic=intrinsic)
    # cv2.imwrite(out_path, img)

def get_index_from_image_identifier(image_identifier, es_info):
    image_paths = es_info['image_paths']
    in_index = [i for i, path in enumerate(image_paths) if image_identifier in path]
    if len(in_index) == 1:
        return in_index[0]
    else:
        print(f"found {len(in_index)} images with identifier {image_identifier}")
        return None

# scene0000_00:
# case 16: Find all the items with ceramic material in the room. Good for failure case.
# case 45: Find all the curtains in the room. Success case.
# case 47: Find all the bins in the room. Failure case.
# vis_one_case(47)

# scene0039_00:
# vis_one_case(72, show_gt=True, max_results=1, filter_failure=False) # Success case: OR
# vis_one_case(128, show_gt=True, max_results=3, filter_failure=False) # Failure case.
# exit()

# scene0094_00:
# vis_one_case(36, show_gt=True, max_results=1, result_idx=0, filter_failure=False) # Success
# vis_one_case(107, show_gt=True, max_results=1, result_idx=0, filter_failure=False) # Success
vis_one_case(153, show_gt=True, max_results=1, result_idx=2, filter_failure=False) # Success
# vis_one_case(167, show_gt=True, max_results=1, result_idx=0, filter_failure=False) # Flexible, can be used for failure case.

exit()

# office:
# vis_one_case(5, show_gt=False, max_results=3)
# vis_one_case(5, show_gt=False, image_identifier='1698815875.880117', max_results=3)
# vis_one_case(0, show_gt=False, max_results=1, result_idx=0)
# vis_one_case(0, show_gt=False, image_identifier='1698815815.780094', max_results=1, result_idx=0)

# restroom:
# vis_one_case(4, show_gt=False, max_results=1, result_idx=3)
# vis_one_case(4, show_gt=False, image_identifier='1700807599.340621', max_results=1, result_idx=3)
# vis_one_case(7, show_gt=False, max_results=1, result_idx=1)
# vis_one_case(7, show_gt=False, image_identifier='1700807605.873954', max_results=1, result_idx=1)
# vis_one_case(5, show_gt=False, max_results=1, result_idx=0)
# vis_one_case(5, show_gt=False, image_identifier='1700807557.173943', max_results=1, result_idx=0)

length = len(gt_data)
if __name__ == '__main__':
    for j in range(length):
        vis_one_case(j, show_gt=not is_open_scan and True, min_gts=1, max_results=10, filter_failure=True)

