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
from utils.utils_vis import create_box_mesh
from utils.grounding_metric import nms_with_iou_matrix
from utils.utils_vis import annotate_image_with_3dbboxes, get_o3d_obb


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

scene_id = 'scene0094_00'

# pcd = o3d.io.read_point_cloud("visualization\demo_scenes/{scene_id}/{scene_id}_vh_clean_2.ply")
pcd = o3d.io.read_triangle_mesh(f"visualization\demo_scenes/{scene_id}/{scene_id}_vh_clean_2.ply")
es_info = read_annotation_pickle(f"splitted_infos/{scene_id}.pkl")[f'{scene_id}']
axis_align_matrix = np.array(es_info['axis_align_matrix'])
pcd.transform(axis_align_matrix)

gt_file = f"visualization\demo_scenes/{scene_id}/{scene_id}_flattened_token_positive.json"
pred_file = f"visualization\demo_scenes/{scene_id}/{scene_id}_results_100Per.json"
gt_data = json.load(open(gt_file))
pred_data = json.load(open(pred_file))
assert len(gt_data) == len(pred_data)
# gt_data[0]: {'sub_class': 'VG_Single_Attribute_Common', 'scan_id': '{scene_id}', 'target_id': [189, 179], 'distractor_ids': [], 'text': 'Find all the items with lighting fixtures coarse grained category in the room.', 'target': ['lamp', 'switch'], 'anchors': [], 'anchor_ids': [], 'tokens_positive': [[13, 65], [13, 65]], 'ID': 'VG_Single_Attribute_Common__scene0000_00__8'}
# pred_data[0]: {'bboxes_3d': [[1.1259186267852783, -1.5820398330688477, 0.4439241290092468, 0.4303528666496277, 0.38400211930274963, 1.0724506378173828, 1.6592847108840942, -0.003870368003845215, -3.1391913890838623], [0.9865046739578247, 3.311486005783081, 1.2967586517333984, 0.23403628170490265, 0.2800081968307495, 0.3285176753997803, -1.535447597503662, 0.0007756948471069336, -3.142092704772949]], 'scores_3d': [0.26313698291778564, 0.25616997480392456]}

def vis_one_case(case_id, apply_filter=True, image_identifier=None):
    gt = gt_data[case_id]
    pred = pred_data[case_id]
    target_ids = gt['target_id']
    all_obj_ids = list(es_info['object_ids'])
    obj_inds = [all_obj_ids.index(i) for i in target_ids]
    gt_boxes = np.array(es_info['bboxes'])[obj_inds]
    num_gt = len(gt_boxes)
    if num_gt >= 5 and apply_filter:
        return
    if num_gt <= 0:
        return
    pred_boxs = np.array(pred['bboxes_3d'])
    pred_scores = np.array(pred['scores_3d'])
    pred_corners = corners_from_9dof(pred_boxs)
    ious = aabb_iou(pred_corners, pred_corners)
    keep_inds = nms_with_iou_matrix(ious, 0.15, pred_scores)
    pred_boxs = pred_boxs[keep_inds]
    pred_boxs = pred_boxs[:num_gt]

    centers_gt = gt_boxes[:, :3]
    centers_pred = pred_boxs[:, :3]
    if len(centers_gt) > 20:
        return
    if not check_pcd_similarity(centers_gt, centers_pred, 0.2) and apply_filter:
        return

    boxes = []
    mode = 'simple' if num_gt >= 10 else 'linemesh'
    for i in range(num_gt):
        gt_box = gt_boxes[i]
        pred_box = pred_boxs[i]
        boxes.extend(create_box(gt_box, (0, 1, 0), mode))
        boxes.extend(create_box(pred_box, (1, 0, 0), mode))
    print(case_id, gt['text'])
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
    boxes = get_o3d_obb(boxes)
    object_ids = np.zeros(len(boxes))
    object_types = ['gt'] * len(gt_boxs) + ['pred'] * len(pred_boxs)
    intrinsics = es_info['intrinsics']
    intrinsic = intrinsics[image_index]
    extrinsics_c2w = es_info['extrinsics_c2w']
    extrinsic = extrinsics_c2w[image_index]
    axis_align_matrix = es_info['axis_align_matrix']
    out_path = 'test.jpg'
    annotate_image_with_3dbboxes(image_path, boxes, object_ids, object_types, intrinsic, extrinsic, axis_align_matrix, out_path)
    
def get_index_from_image_identifier(image_identifier, es_info):
    image_paths = es_info['image_paths']
    in_index = [i for i, path in enumerate(image_paths) if image_identifier in path]
    if len(in_index) == 1:
        return in_index[0]
    else:
        return None

# scene0000_00:
# case 16: Find all the items with ceramic material in the room. Good for failure case.
# case 45: Find all the curtains in the room. Success case.
# case 47: Find all the bins in the room. Failure case.
# vis_one_case(47)

# scene0094_00:
# case 167: Water is poured from the X into the teapot for boiling in the cooking region. Please find the X. Success case.

vis_one_case(167)

exit()
length = len(gt_data)
if __name__ == '__main__':
    for j in range(length):
        vis_one_case(j)

