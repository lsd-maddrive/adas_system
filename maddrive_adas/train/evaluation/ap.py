import numpy as np


def process_iou_results(ap_thresh_cls, iou_matrix, scores, gt_labels=None, pr_labels=None):
    scores = np.array(scores)

    det_sort_indices = np.argsort(-scores)
    n_classes = len(next(iter(ap_thresh_cls.values())))

    if gt_labels is None and pr_labels is None:
        gt_labels = [0]*iou_matrix.shape[1]
        pr_labels = [0]*iou_matrix.shape[0]
    elif gt_labels is None or pr_labels is None:
        raise ValueError(f'labels must be provided or both None')

    num_gt = len(gt_labels)

    for i_class in range(n_classes):
        num_gt_for_class = sum([1 for x in gt_labels if x == i_class])

        for iou_threshold, ap_thresh in ap_thresh_cls.items():
            gt_found = [False] * len(gt_labels)

            ap_pred = ap_thresh[i_class]
            ap_pred.add_gt_positives(num_gt_for_class)

            for i in det_sort_indices:
                if pr_labels[i] != i_class:
                    continue

                max_iou_found = iou_threshold
                max_match_idx = -1

                for j in range(num_gt):
                    if gt_found[j] or gt_labels[j] != i_class:
                        continue

                    iou = iou_matrix[i, j]
                    if iou > max_iou_found:
                        max_iou_found = iou
                        max_match_idx = j

                if max_match_idx >= 0:
                    gt_found[max_match_idx] = True
                    ap_pred.append(scores[i], True)
                else:
                    ap_pred.append(scores[i], False)


class PredictionAPCollector:
    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def append(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        if self.num_gt_positives == 0:
            return 0

        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        for _, is_true in self.data_points:
            if is_true:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        indices = np.searchsorted(recalls, x_range, side="left")
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        return sum(y_range) / len(y_range)


# Other implementations


def calculate_ap_interpolated(rec, prec, generate_plot=False):
    mrec = rec
    mpre = prec

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        recall_gr_inds = np.argwhere(mrec >= r)
        pmax = 0
        # If there are recalls above r
        if recall_gr_inds.size != 0:
            pmax = max(mpre[recall_gr_inds.min() :])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot
    if generate_plot:
        rvals = [recallValid[0]]
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = [0]
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]

    return [ap, rhoInterp, recallValues, None]


def calculate_ap(rec, prec):
    mrec = [0]
    [mrec.append(e) for e in rec]
    mrec.append(1)

    mpre = [0]
    [mpre.append(e) for e in prec]
    mpre.append(0)

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0 : len(mpre) - 1], mrec[0 : len(mpre) - 1], ii]
