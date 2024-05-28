# Copyright (c) OpenRobotLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union, Any
# 这份代码并不设计在这个项目中运行。
from terminaltables import AsciiTable
import logging
import numpy as np
import torch
from utils_3d import *
from scipy.optimize import linear_sum_assignment

def to_cpu(x):
    if isinstance(x, (list, tuple)):
        return [to_cpu(y) for y in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x

def ground_eval(gt_anno_list, det_anno_list, logger=None):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'is_hard': bool
            'direct': bool
            'space': bool
            'sub_class': str
    """
    assert len(det_anno_list) == len(gt_anno_list)
    det_anno_list = to_cpu(det_anno_list)
    gt_anno_list = to_cpu(gt_anno_list)
    pred = {}
    gt = {}
    iou_thr = [0.25, 0.5]
    object_types = [
        'Spacial', 'Attribute', 'Direct', 'Indirect', 'Hard', 'Easy',
        'Single', 'Multi', 'Overall', 'Direct_Attribute_O_Individual', 'Direct_Attribute_O_Common', 'Direct_EQ',
        'InDirect_OR', 'InDirect_Space_OO', 'InDirect_Attribute_OO'
    ]

    for t in iou_thr:
        for object_type in object_types:
            pred.update({object_type + '@' + str(t): 0})
            gt.update({object_type + '@' + str(t): 1e-14})

    need_warn = False
    for sample_id in range(len(det_anno_list)):
        det_anno = det_anno_list[sample_id]
        gt_anno = gt_anno_list[sample_id]
        target_scores = det_anno['target_scores_3d']  # (num_query, )
        if isinstance(gt_bboxes, (list, tuple)):
            num_gts = gt_bboxes[0].shape[0]
        else:
            num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            continue
        keep_inds = target_scores.argsort(dim=-1, descending=True)[:num_gts]

        if 'bboxes_3d' in det_anno:
            bboxes = det_anno['bboxes_3d'] # (num_query, 9)
            top_pred_bboxes = bboxes[keep_inds]
        else:
            pcds = det_anno['pcds'] # (num_query, pcd_size, 3)
            pcds = pcds[keep_inds]
            top_pred_bboxes = compute_bbox_from_points_list(pcds) # tuple of (center, size, rotmat)
        # or a (list, tuple) (center, size, rotmat): (num_query, 3), (num_query, 3), (num_query, 3, 3)
        gt_bboxes = gt_anno['gt_bboxes_3d'] # (num_gt, 9) 

        sub_class = gt_anno.get('sub_class', None)
        if sub_class is not None:
            sub_class = sub_class.strip("VG_") # remove VG_ prefix
        hard = gt_anno.get('is_hard', None)
        space = gt_anno.get('space', None)
        direct = gt_anno.get('direct', None)
        if (hard is None or space is None or direct is None) and sub_class is None:
            need_warn = True

        pred_idices, gt_idices, costs = matcher(top_pred_bboxes, gt_bboxes, [iou_cost_fn])
        iou = 1.0 - costs # warning: only applicable when iou_cost_fn is the only cost function used

        for t in iou_thr:
            threshold = iou > t
            found = np.sum(threshold)
            if space:
                gt['Spacial@' + str(t)] += num_gts
                pred['Spacial@' + str(t)] += found
            else:
                gt['Attribute@' + str(t)] += num_gts
                pred['Attribute@' + str(t)] += found
            if direct:
                gt['Direct@' + str(t)] += num_gts
                pred['Direct@' + str(t)] += found
            else:
                gt['Indirect@' + str(t)] += num_gts
                pred['Indirect@' + str(t)] += found
            if hard:
                gt['Hard@' + str(t)] += num_gts
                pred['Hard@' + str(t)] += found
            else:
                gt['Easy@' + str(t)] += num_gts
                pred['Easy@' + str(t)] += found
            if sub_class is not None:
                gt[f'{sub_class}@' + str(t)] += num_gts
                pred[f'{sub_class}@' + str(t)] += found
            if num_gts <= 1:
                gt['Single@' + str(t)] += num_gts
                pred['Single@' + str(t)] += found
            else:
                gt['Multi@' + str(t)] += num_gts
                pred['Multi@' + str(t)] += found

            gt['Overall@' + str(t)] += num_gts
            pred['Overall@' + str(t)] += found
    if need_warn:
        logging.warning('Some annotations are missing "is_hard", "space", or "direct" information.')
    header = ['Type']
    header.extend(object_types)
    ret_dict = {}

    for t in iou_thr:
        table_columns = [['results']]
        for object_type in object_types:
            metric = object_type + '@' + str(t)
            value = pred[metric] / max(gt[metric], 1)
            ret_dict[metric] = value
            table_columns.append([f'{value:.4f}'])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        # print('\n' + table.table)
        if logger is not None:
            logger.info('\n' + table.table)
        else:
            print('\n' + table.table)
    ret_dict['gt'] = gt
    ret_dict['pred'] = pred
    ret_dict['table'] = table
    return ret_dict


def get_corners(box):
    # box should be (n, 9) or a (list, tuple) (center, size, rotmat)
    if isinstance(box, (list, tuple)):
        center, size, rotmat = box
    else:
        if len(box.shape) == 1:
            box = box.reshape(1, 9)
        center = box[:, :3]
        size = box[:, 3:6]
        euler = box[:, 6:9]
        rotmat = euler_angles_to_matrix(euler, convention="ZXY")
    corners = cal_corners(center, size, rotmat)
    return corners

def compute_ious(boxes1, boxes2):
    """Compute the intersection over union one by one between two 3D bounding boxes.
    Boxes1: (N, 9) or a (list, tuple) (center, size, rotmat)
    Boxes2: (M, 9) or a (list, tuple) (center, size, rotmat)
    Return: (N, M)
    """
    from pytorch3d.ops import box3d_overlap
    import torch
    corners1 = torch.tensor(get_corners(boxes1), dtype=torch.float32)
    corners2 = torch.tensor(get_corners(boxes2), dtype=torch.float32)
    _, ious = box3d_overlap(corners1, corners2)
    ious = ious.numpy()
    return ious


def matcher(preds, gts, cost_fns):
    """
    Matcher function that uses the Hungarian algorithm to find the best match
    between predictions and ground truths.

    Parameters:
    - preds: predicted bounding boxes (num_preds) 
    - gts: ground truth bounding boxes (num_gts)
    - cost_fn: a function that computes the cost matrix between preds and gts

    Returns:
    - matched_pred_inds: indices of matched predictions
    - matched_gt_inds: indices of matched ground truths
    - costs: cost of each matched pair
    """
    # Compute the cost matrix
    num_preds = len(preds) if not isinstance(preds, (list, tuple)) else len(preds[0])
    num_gts = len(gts) if not isinstance(gts, (list, tuple)) else len(gts[0])
    cost_matrix = np.zeros((num_preds, num_gts))
    for cost_fn in cost_fns:
        cost_matrix += cost_fn(preds, gts) #shape (num_preds, num_gts)

    # Perform linear sum assignment to minimize the total cost
    matched_pred_inds, matched_gt_inds = linear_sum_assignment(cost_matrix)
    costs = cost_matrix[matched_pred_inds, matched_gt_inds]
    return matched_pred_inds, matched_gt_inds, costs

# Example cost function that calculates the IoU between bounding boxes
def iou_cost_fn(pred_boxes, gt_boxes):
    ious = compute_ious(pred_boxes, gt_boxes)
    ious = np.nan_to_num(ious, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
    return 1.0 - ious


if __name__ == '__main__':
    
    centers = np.random.rand(10, 3)
    sizes = np.random.rand(10, 3) + 10
    euler = np.random.rand(10, 3) - .5
    pred_boxes = np.concatenate([centers, sizes, euler], axis=1)

    centers = np.random.rand(5, 3)
    sizes = np.random.rand(5, 3) + 10
    euler = np.random.rand(5, 3) - .5
    gt_boxes = np.concatenate([centers, sizes, euler], axis=1)

    matched_row_inds, matched_col_inds = matcher(pred_boxes, gt_boxes, [iou_cost_fn])
    print(matched_row_inds, matched_col_inds)