
import torch


class TorchNMS(object):
    def __init__(self, iou, type='vanilla', **kwargs):
        self.iou = iou
        self.type = type

        if type == 'vanilla':
            self.iou_threshold = kwargs['iou_threshold']
            self.exec = self.vanilla
        elif type == 'soft':
            self.sigma = kwargs['sigma']
            self.conf_threshold = kwargs['conf_threshold']
            self.exec = self.soft_nms
        else:
            raise NotImplementedError(f'NMS type {type} not supported')

    def vanilla_old(self, preds):
        # preds ~ [[x, y, w, h, score, label]]
        # NMS for one image (prediction ~ bboxes with labels and scores)
        assert preds.shape[1] == 6
        assert isinstance(preds, torch.Tensor)

        ordered_idxs = (-preds[:, 4]).argsort()

        iou_matrix = self.iou(
            preds[:, :4],
            preds[:, :4]
        )

        keep = []
        while ordered_idxs.shape[0] > 0:
            idx_self = ordered_idxs[0]
            keep.append(idx_self)

            preds_other = preds[ordered_idxs, :]
            ious = iou_matrix[idx_self, ordered_idxs]

            check_label = preds[idx_self, 5]

            high_iou_inds = (ious >= self.iou_threshold)
            same_classes_inds = preds_other[:, 5] == check_label
            mask = ~(high_iou_inds & same_classes_inds)

            ordered_idxs = ordered_idxs[mask]

        return torch.LongTensor(keep)

    def vanilla(self, preds):
        # preds ~ [[x, y, w, h, score, label]]
        # NMS for one image (prediction ~ bboxes with labels and scores)
        assert preds.shape[1] == 6
        assert isinstance(preds, torch.Tensor)

        # Decreasing sort
        ordered_idxs = (-preds[:, 4]).argsort()

        # iou_matrix = self.iou(
        #     preds[:, :4],
        #     preds[:, :4]
        # )

        keep = []
        while ordered_idxs.shape[0] > 0:
            idx_self = ordered_idxs[0]
            keep.append(idx_self)

            preds_other = preds[ordered_idxs, :]

            ious = self.iou(
                preds[idx_self:idx_self+1, :4],
                preds[ordered_idxs, :4]
            )[0]
            # ious = iou_matrix[idx_self, ordered_idxs]

            check_label = preds[idx_self, 5]

            high_iou_inds = (ious >= self.iou_threshold)
            same_classes_inds = (preds_other[:, 5] == check_label)
            mask = ~(high_iou_inds & same_classes_inds)

            ordered_idxs = ordered_idxs[mask]

        return torch.LongTensor(keep)

    # https://arxiv.org/pdf/1704.04503.pdf
    def soft_nms(self, preds):
        raise NotImplementedError
        # preds ~ [[x, y, w, h, score, label]]
        # NMS for one image (prediction ~ bboxes with labels and scores)
        assert preds.shape[1] == 6
        assert isinstance(preds, torch.Tensor)

        # Copy to avoid modification
        preds = preds.detach().clone()

        indexes = torch.arange(0, preds.shape[0], dtype=torch.float32)
        preds = torch.cat([preds, indexes[:, None]], dim=1)

        iou_matrix = self.iou(
            preds[:, :4],
            preds[:, :4]
        )

        n_preds = preds.shape[0]

        for i in range(n_preds):
            if i < n_preds-1:
                # Find top score of boxes after i
                max_score, max_score_idx = torch.max(preds[i:, 4], dim=0)

                # Swap only if max score is higher than current
                if max_score > preds[i, 4]:
                    max_score_idx += i
                    # Swap current with max bbox
                    preds[[i, max_score_idx], :] = preds[[max_score_idx, i], :]
                # If next max is less than threshold and current i`th - exit, as we won`t have them processing
                elif max_score < self.conf_threshold:
                    break

            # Get values of current max
            current_label = preds[i, 5]
            current_idx = preds[i, -1].long()
            other_idxs = preds[i+1:, -1].long()
            same_classes_inds = preds[i+1:, 5] == current_label

            ious = iou_matrix[current_idx, other_idxs]

            weight = torch.exp(-(ious * ious) / self.sigma)
            # Supress weightening of another class
            weight[~same_classes_inds] = 1

            preds[i+1:, 4] *= weight

        keep = preds[:, 4] > self.conf_threshold
        return torch.LongTensor(keep)
