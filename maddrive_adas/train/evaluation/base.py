import numpy as np
import logging
from tqdm import tqdm

from ...utils.bbox import iou_xywh_numpy
from .ap import PredictionAPCollector, process_iou_results

logger = logging.getLogger(__name__)


class BatchEvaluator:
    def __init__(self, infer, datasets, metrics_eval, batch_size=1):
        self.infer = infer
        self.datasets = datasets
        self.batch_size = batch_size
        self.metrics_eval = metrics_eval

    def update_infer(self, model_state):
        self.infer.update_model_state(model_state)

    def evaluate(self):
        self.metrics_eval.clear()

        image_id = 0
        _images_list = []
        _trues_list = []

        logger.debug("Start predictions")

        def perform_prediction():
            predictions = self.infer.infer_batch(_images_list)
            local_image_ids = image_id

            for i_pred, pred in enumerate(predictions):
                self.metrics_eval.append_sample(pred, _trues_list[i_pred])

                local_image_ids += 1

            return local_image_ids

        global_len = sum([len(ds) for ds in self.datasets])
        logger.debug(f"Processing {global_len} samples for evaluation")

        with tqdm(total=global_len, desc=f"Evaluation", unit="img", ncols=80) as pbar:
            for dataset in self.datasets:
                for i in range(len(dataset)):
                    img, ann = dataset[i]
                    _images_list.append(img)
                    _trues_list.append(ann)

                    if len(_images_list) >= self.batch_size:
                        image_id = perform_prediction()
                        pbar.update(len(_images_list))
                        _images_list.clear()
                        _trues_list.clear()

            # Last predictions
            if len(_images_list) > 0:
                image_id = perform_prediction()
                pbar.update(len(_images_list))

        logger.debug("Predictions done, start calculating metrics")
        return self.metrics_eval.get_metrics()


class MetricsEvaluator:
    def __init__(self, labels, iou_thresholds=[50, 75]):
        assert all([t > 1 for t in iou_thresholds])

        self.labels = labels
        self.n_classes = len(labels)
        self.iou_thresholds = iou_thresholds

        self.clear()

    def _process_sample(self, pred, true):
        pr_bboxes = pred["bboxes"]
        pr_labels = pred["classes"]
        pr_confs = pred["scores"]

        gt_bboxes = true["bboxes"]
        gt_labels = true["labels"]

        num_gt = len(gt_labels)
        num_pr = len(pr_labels)

        iou_mtrx = None
        if num_pr > 0 and num_gt > 0:
            # Increase to match percentage range [0; 100]
            iou_mtrx = iou_xywh_numpy(pr_bboxes, gt_bboxes) * 100

        process_iou_results(self.ap_preds, iou_mtrx, pr_confs, gt_labels, pr_labels)

    def append_sample(self, pred, true):
        self._process_sample(pred, true)

    def clear(self):
        self.ap_preds = {
            it: [PredictionAPCollector() for _ in range(self.n_classes)]
            for it in self.iou_thresholds
        }

    def get_metrics(self) -> dict:
        aggregated = {}

        for iou_threshold in self.iou_thresholds:
            aps = []
            for i_class in range(self.n_classes):
                ap_pred = self.ap_preds[iou_threshold][i_class]

                AP = 0
                if not ap_pred.is_empty():
                    AP = ap_pred.get_ap()
                    aps.append(AP)
                aggregated[f"ap{int(iou_threshold)}/{self.labels[i_class]}"] = AP

            mAP = np.mean(aps)
            aggregated[f"map{int(iou_threshold)}"] = mAP

        return aggregated
