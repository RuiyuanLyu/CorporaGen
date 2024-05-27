# Copyright (c) OpenRobotLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union, Any
# 这份代码并不设计在这个项目中运行。
from terminaltables import AsciiTable
import logging
import numpy as np
from utils_3d import compute_bbox_from_points_open3d, cal_corners, euler_angles_to_matrix


# TODO: 其它项目的dataloader需要修改.增加一些is view dep和is hard的标注
# 输出的data samples需要增加target_scores_3d, bboxes_3d, gt_bboxes_3d, is_view_dep, is_hard的标注

class GroundingMetricEvaluator(object):
    """Lanuage grounding evaluation metric. We calculate the grounding
    performance based on the alignment score of each bbox with the input
    prompt.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """ 

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def ground_eval(self, gt_annos, det_annos, logger=None):

        assert len(det_annos) == len(gt_annos)

        pred = {}
        gt = {}

        object_types = [
            'Easy', 'Hard', 'View-Dep', 'View-Indep', 'Unique', 'Multi',
            'Overall'
        ]

        for t in self.iou_thr:
            for object_type in object_types:
                pred.update({object_type + '@' + str(t): 0})
                gt.update({object_type + '@' + str(t): 1e-14})
        need_warn = False
        for sample_id in range(len(det_annos)):
            det_anno = det_annos[sample_id]
            gt_anno = gt_annos[sample_id]
            target_scores = det_anno['target_scores_3d']  # (num_query, )

            bboxes = det_anno['bboxes_3d'] # (num_query, 9)
            gt_bboxes = gt_anno['gt_bboxes_3d'] # (num_gt, 9) 

            hard = gt_anno.get('is_hard', None)
            space = gt_anno.get('space', None)
            direct = gt_anno.get('direct', None)
            if hard is None or space is None or direct is None:
                need_warn = True
            multi = gt_bboxes.shape[0] > 1 # require multiple objects

            box_index = target_scores.argsort(dim=-1, descending=True)[:10]
            top_bbox = bboxes[box_index]

            iou = compute_ious(top_bbox, gt_bboxes)

            for t in self.iou_thr:
                threshold = iou > t
                num_gts = gt_bboxes.shape[0]
                found = threshold.any(dim=0).sum().item()
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

        for t in self.iou_thr:
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
            logging.info('\n' + table.table)

        return ret_dict

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results after all batches have
        been processed.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        annotations, preds = zip(*results)
        ret_dict = self.ground_eval(annotations, preds)

        return ret_dict

def corner_from_9dof(box):
    center = box[:3]
    size = box[3:6]
    euler = box[6:9]
    rotmat = euler_angles_to_matrix(euler, convention="ZXY")
    corners = cal_corners(center, size, rotmat)
    return corners

def compute_ious(boxes1, boxes2):
    """Compute the intersection over union one by one between two 3D bounding boxes.
    Boxes1: (N, 9)
    Boxes2: (M, 9)
    Return: (N, M)
    """
    from pytorch3d.ops import box3d_overlap
    assert boxes1.shape == boxes2.shape
    corners1 = corner_from_9dof(boxes1)
    corners2 = corner_from_9dof(boxes2)
    ious = box3d_overlap(corners1, corners2)
    return ious
