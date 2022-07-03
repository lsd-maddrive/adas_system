import cv2
import numpy as np
import colorsys
import random
import math as m


def draw_bboxes(img, bboxes, line_thinkness=2, line_color=(0, 0, 255)):
    """Render unnamed bboxes on image [resize before drawing]

    Args:
        img (ndarray): image to draw on
        bboxes (list): bboxes in COCO format [x, y, w, h]
        line_thinkness (int, optional): thickness of bbox line. Defaults to 2.
        line_color (tuple, optional): color of bbox line. Defaults to (0, 0, 255).

    Returns:
        ndarray: resized image with bboxes on it
    """
    canvas = img.copy()

    for i_p, (x, y, w, h) in enumerate(bboxes):
        cv2.rectangle(
            canvas,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            line_color,
            line_thinkness,
        )
    return canvas


def put_label(canvas, text, pos, color=(0, 0, 255), font_sz=0.7, font_width=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        text,
        pos,
        font,
        font_sz,
        (0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=font_width * 2,
    )
    cv2.putText(
        canvas,
        text,
        pos,
        font,
        font_sz,
        color,
        lineType=cv2.LINE_AA,
        thickness=font_width,
    )


def draw_detections(
    canvas,
    bboxes,
    labels,
    scores,
    line_color=(0, 0, 255),
    line_width=2,
    font_color=(255, 255, 255),
    font_sz=0.7,
    font_width=1,
):
    for i_p, (x, y, w, h) in enumerate(bboxes):
        label = labels[i_p]
        score = str(round(scores[i_p], 2))

        cv2.rectangle(
            canvas, (int(x), int(y)), (int(x + w), int(y + h)), line_color, line_width
        )

        put_label(
            canvas,
            "{} {}".format(label, score),
            (int(x), int(y - 5)),
            color=font_color,
            font_sz=font_sz,
            font_width=font_width,
        )

    return canvas


class MasksRender(object):
    def __init__(self, alpha=0.5):
        self.n_classes = 0
        self.alpha = alpha

    def render_on_image(self, canvas, mask, skip_zero_index=True, copy=True):
        if copy:
            canvas = canvas.copy()

        n_masks = np.amax(mask) + 1
        if n_masks > self.n_classes:
            self.n_classes = n_masks
            self.colors = generate_colors(self.n_classes)

        for i_mask in range(self.n_classes):
            if i_mask == 0 and skip_zero_index:
                continue
            # Skip 0 - background
            class_mask = (mask == i_mask).astype(int)
            # Remove 3rd dim if exists
            class_mask = np.squeeze(class_mask)
            apply_mask(canvas, class_mask, color=self.colors[i_mask], alpha=self.alpha)
        return canvas

    def colorize_mask(self, mask):
        canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
        return self.render_on_image(canvas, mask)


def generate_colors(N):
    # From https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.arange(N)[:, None] * palette
    return (colors % 255).astype("uint8")


def apply_mask(canvas, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        canvas[:, :, c] = np.where(
            mask == 1, canvas[:, :, c] * (1 - alpha) + alpha * color[c], canvas[:, :, c]
        )
    return canvas


def draw_cat_mask(canvas, cat_mask, alpha=0.5):
    n_masks = np.amax(cat_mask) + 1
    colors = random_colors(N=n_masks)
    if len(cat_mask.shape) > 2:
        cat_mask = cat_mask[..., 0]

    colors[0] = [0, 0, 0]
    # colors[1] = [255, 255, 255]
    for i_mask in range(n_masks):
        class_mask = (cat_mask == i_mask).astype(int)
        canvas = apply_mask(canvas, class_mask, color=colors[i_mask], alpha=alpha)
    return canvas


def draw_instance_masks(canvas, inst_masks, alpha=0.5):
    n_insts = inst_masks.shape[-1]
    colors = random_colors(N=n_insts)
    for i_inst in range(n_insts):
        inst_mask = inst_masks[:,:,i_inst]
        canvas = apply_mask(canvas, inst_mask, color=colors[i_inst], alpha=alpha)
    return canvas


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    colors = np.array(colors, np.float32)
    colors *= 255
    return colors.astype(np.uint8)


def draw_poses(img, kps_objects, connections, kps_colors, connection_colors):
    for obj_kps in kps_objects:
        img = draw_pose(img, obj_kps, connections, kps_colors, connection_colors)
    return img


def draw_pose(img, obj_kps, connections, kps_colors, connection_colors):
    obj_kps = np.array(obj_kps)
    img = img.copy()

    for i, kp in enumerate(obj_kps):
        cv2.circle(img, tuple(kp[0:2].astype(int)), 2, kps_colors[i], thickness=-1)

    for l, (src_idx, dst_idx) in enumerate(connections):
        src_joint = obj_kps[src_idx]
        dst_joint = obj_kps[dst_idx]

        if src_joint[2] == 0 or dst_joint[2] == 0:
            continue

        coords_center = tuple(
            np.round((src_joint[:2] + dst_joint[:2]) / 2.0).astype(int)
        )

        limb_dir = src_joint - dst_joint
        limb_length = np.linalg.norm(limb_dir)
        # Get the angle of limb_dir in degrees using atan2(limb_dir_x,
        # limb_dir_y)
        angle = m.degrees(m.atan2(limb_dir[1], limb_dir[0]))

        limb_thickness = 1

        # For faster plotting, just plot over canvas instead of constantly
        # copying it
        polygon = cv2.ellipse2Poly(
            coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(img, polygon, connection_colors[l])

    return img
