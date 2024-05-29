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

def box_num(box):
    if isinstance(box, (list, tuple)):
        return box[0].shape[0]
    else:
        return box.shape[0]

def index_box(boxes, indices):
    if isinstance(boxes, (list, tuple)):
        return [index_box(box, indices) for box in boxes]
    else:
        return boxes[indices]

mapping = {
    'direct_attribute_o_individual': 'dir_attr_indi',
    'direct_attribute_o_common': 'dir_attr_com',
    'direct_eq': 'dir_eq',
    'indirect_or': 'indir_or',
    'indirect_space_oo': 'indir_space',
    'indirect_attribute_oo': 'indir_attr',
    'other': 'other',
    'overall': 'overall'
}

def ground_eval_subset(gt_anno_list, det_anno_list, logger=None, prefix=''):
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
    iou_thr = [0.25, 0.5]
    num_samples = len(gt_anno_list) # each sample contains multiple pred boxes
    total_pred_boxes = 0
    # these lists records for each sample, whether a gt box is matched or not
    gt_matched_records = []
    # these lists records for each pred box, NOT for each sample        
    sample_indices = [] # each pred box belongs to which sample
    confidences = [] # each pred box has a confidence score
    ious = [] # each pred box has a ious, shape (num_gt) in the corresponding sample
    # record the indices of each reference type

    for sample_idx in range(num_samples):
        det_anno = det_anno_list[sample_idx]
        gt_anno = gt_anno_list[sample_idx]

        target_scores = det_anno['target_scores_3d']  # (num_query, )
        top_idxs =  np.argsort(-target_scores)[:20] #HACK: hard coded
        target_scores = target_scores[top_idxs]
        pred_bboxes = index_box(det_anno['bboxes_3d'], top_idxs)
        gt_bboxes = gt_anno['gt_bboxes_3d']

        num_preds = box_num(pred_bboxes)
        total_pred_boxes += num_preds
        num_gts = len(gt_bboxes)
        gt_matched_records.append(np.zeros(num_gts, dtype=np.bool))

        iou_mat = compute_ious(pred_boxes, gt_boxes)
        for i, score in enumerate(target_scores):
            sample_indices.append(sample_idx)
            confidences.append(score)
            ious.append(iou_mat[i])
    

    confidences = np.array(confidences)
    sorted_inds = np.argsort(-confidences)
    sample_indices = [sample_indices[i] for i in sorted_inds]
    ious = [ious[i] for i in sorted_inds]

    tp_thr = {}
    fp_thr = {}
    for thr in iou_thr:
        tp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))
        fp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))

    for d, sample_idx in enumerate(sample_indices):
        iou_max = -np.inf
        num_gts = len(gt_anno_list[sample_idx]['gt_bboxes_3d'])
        cur_iou = ious[d]
        if num_gts > 0:
            for j in range(num_gts):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
        
        for iou_idx, thr in enumerate(iou_thr):
            if iou_max >= thr:
                if not gt_matched_records[sample_idx][jmax]:
                    gt_matched_records[sample_idx][jmax] = True
                    tp_thr[f'{prefix}@{thr}'][d] = 1.0
                else:
                    fp_thr[f'{prefix}@{thr}'][d] = 1.0
            else:
                fp_thr[f'{prefix}@{thr}'][d] = 1.0

    ret = {}
    for t in iou_thr:
        metric = prefix + '@' + str(t)
        fp = np.cumsum(fp_thr[metric])
        tp = np.cumsum(tp_thr[metric])
        recall = tp / float(total_pred_boxes)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(precision, recall)
        ret[metric] = float(ap)
        best_recall = recall[-1] if len(recall) > 0 else 0
        ret[metric + '_rec'] = float(best_recall)
    return ret

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
    iou_thr = [0.25, 0.5]
    reference_options = [v for k, v in mapping.items()]
    assert len(det_anno_list) == len(gt_anno_list)
    results = {}
    for ref in reference_options:
        indices = [i for i, gt_anno in enumerate(gt_anno_list) if gt_anno.get('sub_class', 'other').strip('vg_') == ref]
        sub_gt_annos = [gt_anno_list[i] for i in indices ]
        sub_det_annos = [det_anno_list[i] for i in indices ]
        ret = ground_eval_subset(sub_gt_annos, sub_det_annos, logger=logger, prefix=ref)
        for k, v in ret.items():
            results[k] = v
    overall_ret = ground_eval_subset(gt_anno_list, det_anno_list, logger=logger, prefix='overall')
    for k, v in overall_ret.items():
        results[k] = v
    
    header = ['Type']
    header.extend(reference_options)
    table_columns = [[] for _ in range(len(header))]
    ret = {}
    for t in iou_thr:
        table_columns[0].append('AP  '+str(t))
        table_columns[0].append('Rec '+str(t))            
        for i, ref in enumerate(reference_options):
            metric = ref + '@' + str(t)
            ap = results[metric]
            best_recall = results[metric + '_rec']
            table_columns[i+1].append(f'{float(ap):.4f}')
            table_columns[i+1].append(f'{float(best_recall):.4f}')

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

    return ret


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap



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
    Return: (N, M) numpy array
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