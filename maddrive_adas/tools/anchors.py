import random
import numpy as np
import logging

from ..utils.bbox import iou_wh_numpy


class YoloAnchorsGenerator(object):
    def __init__(self, anchors_count=6, seed=42):
        self.anchors_count = anchors_count
        random.seed(seed)

        self.logger = logging.getLogger(__name__)

    def generate(self, bbox_sizes: list):
        """Generate centroid based on k-means clustering

        Args:
            bbox_sizes (list): List of bboxes pairs (WH format)

        Returns:
            list: List of centroids (WH format)
        """
        bbox_sizes = np.array(bbox_sizes, dtype=np.float32)

        return self.run_kmeans(bbox_sizes, self.anchors_count)

    def centroids_as_string(self, centroids):
        anchors = centroids.astype(int)

        widths = anchors[:, 0]
        sorted_indices = np.argsort(widths)

        out_string = ""
        for i in sorted_indices:
            # w, h
            out_string += f"{anchors[i, 0]},{anchors[i, 1]}, "

        return out_string[:-2]
        # print(out_string[:-2])
        # print(avg_IOU(target_dims, centroids))

    def iou_wh(self, ann, centroids):
        w, h = ann
        similarities = []

        for centroid in centroids:
            c_w, c_h = centroid

            if c_w >= w and c_h >= h:
                similarity = w * h / (c_w * c_h)
            elif c_w >= w and c_h <= h:
                similarity = w * c_h / (w * h + (c_w - w) * c_h)
            elif c_w <= w and c_h >= h:
                similarity = c_w * h / (w * h + c_w * (c_h - h))
            else:  # means both w,h are bigger than c_w and c_h respectively
                similarity = (c_w * c_h) / (w * h)
            similarities.append(similarity)  # will become (k,) shape

        return np.array(similarities)

    def avg_IOU(self, anns, centroids):
        return np.mean([max(iou_wh_numpy(np.array([ann]), centroids)) for ann in anns])

    def run_kmeans(self, ann_dims, anchor_num):
        ann_num, anchor_dim = ann_dims.shape[:2]
        prev_assignments = np.ones(ann_num) * (-1)
        iteration = 0

        old_distances = np.zeros((ann_num, anchor_num))

        # Get random indices to capture random points to set them as centroids
        initial_indices = np.random.choice(ann_num, anchor_num, replace=False)
        centroids = ann_dims[initial_indices]

        while True:
            iteration += 1

            distances = 1 - iou_wh_numpy(ann_dims, centroids)

            # for i in range(ann_num):
            # distances += [1 - self.iou_wh(ann_dims[i], centroids)]
            # distances.shape = (ann_num, anchor_num)
            # distances = np.array(distances)

            print(
                "iteration {}: dists = {}".format(
                    iteration, np.sum(np.abs(old_distances - distances))
                )
            )

            # assign samples to centroids
            assignments = np.argmin(distances, axis=1)

            if (assignments == prev_assignments).all():
                return centroids

            # calculate new centroids
            centroid_sums = np.zeros((anchor_num, anchor_dim), dtype=np.float)
            for i in range(ann_num):
                centroid_sums[assignments[i]] += ann_dims[i]

            for j in range(anchor_num):
                centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

                if (centroid_sums[j] == 0).all():
                    self.logger.error("All centroids are 0 -> error")
                    raise Exception("All centroids are 0")

            prev_assignments = assignments.copy()
            old_distances = distances.copy()
