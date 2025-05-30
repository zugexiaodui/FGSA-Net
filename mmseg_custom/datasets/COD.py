 # Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from prettytable import PrettyTable
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

from mmseg.core import get_label, cod_eval_metrics, cod_pre_eval_to_metrics


@DATASETS.register_module()
class CODDataset(CustomDataset):
    """COD dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'background', 'animal')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(CODDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pred_label, label = get_label(pred, seg_map)
            FM.step(pred=pred_label, gt=label)

            WFM.step(pred=pred_label, gt=label)

            SM.step(pred=pred_label, gt=label)
            EM.step(pred=pred_label, gt=label)
            M.step(pred=pred_label, gt=label)
        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]
        cur_score = sm + wfm + em["curve"].mean() - mae
        ret_metrics = OrderedDict({"cur_score":cur_score,
                                   "Smeasure": sm,
                                   "wFmeasure": wfm,
                                   "MAE": mae,
                                   "adpEm": em["adp"],
                                   "meanEm": em["curve"].mean(),
                                   "maxEm": em["curve"].max(),
                                   "adpFm": fm["adp"],
                                   "meanFm": fm["curve"].mean(),
                                   "maxFm": fm["curve"].max()})

        ret_metrics = {
            metric: value
            for metric, value in ret_metrics.items()
        }

        pre_eval_results = ret_metrics
        return pre_eval_results


    def evaluate(self,
                 results,
                 metric='cod_metrics',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        # global flag, best_metric_dict, best_score, best_epoch
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = cod_eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = cod_pre_eval_to_metrics(results)

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        # for logger
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            summary_table_data.add_column(key, [val])
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # cur_score = ret_metrics_summary['Smeasure'] + ret_metrics_summary['wFmeasure'] + ret_metrics_summary['meanEm'] - ret_metrics_summary['MAE']
        # print_log('Current Score:', logger)
        # print_log(cur_score, logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            eval_results[key] = value / 100.0
        
        # cur_score = ret_metrics_summary['Smeasure'] + ret_metrics_summary['wFmeasure'] + ret_metrics_summary['meanEm'] - ret_metrics_summary['MAE']
        # if flag == 0:
        #     best_score = cur_score
        #     best_epoch = epoch
        #     best_metric_dict = ret_metrics_summary
        #     flag = 1
        # else:
        #     if cur_score > best_score:
        #         best_metric_dict = ret_metrics_summary
        #         best_score = cur_score
        #         best_epoch = epoch
        # # for logger
        # best_summary_table_data = PrettyTable()
        # for key, val in best_metric_dict.items():
        #     best_summary_table_data.add_column(key, [val])
        # print_log('best_Summary:', logger)
        # print_log('\n' + best_summary_table_data.get_string(), logger=logger)

        return eval_results

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
