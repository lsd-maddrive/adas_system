import argparse

import cv2
import torch
import numpy as np
import datetime

from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.base import DetectedInstance

VIDEO_PATH = "/home/katya/adas/adas_system/1.avi"
DETECTOR_ARCHIEVE = "/home/katya/adas/adas_system/detector_archive"

def process_video(model: YoloV5Detector, video_file_path: str):
    video_frame_gen = video_frame_generator(video_file_path)

    while True:
        t0 = datetime.datetime.now()
        frame = next(video_frame_gen)
        pred = model.detect(frame)
        video_frame = plot_predictions(pred, frame)
        video_frame = image_resize(video_frame, height=720)
        dt = datetime.datetime.now() - t0
        frame = plot_fps(video_frame, dt)
        cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def plot_fps(frame: np.ndarray, dt: int):
    frame = cv2.putText(
        frame,
        'fps:' + str(round(1 / dt.total_seconds() , 2)),
        (0, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        3, (0, 0, 0),
        3, cv2.LINE_AA
    )
    return frame


def plot_predictions(pred: DetectedInstance, frame: np.ndarray):
    original_frame = frame.copy()

    for idx in range(pred.get_roi_count()):
        COORD_ARR, conf = pred.get_abs_roi(idx)

        frame = cv2.rectangle(frame, (COORD_ARR[0], COORD_ARR[1]),
                              (COORD_ARR[2], COORD_ARR[3]),
                              (255, 0, 0),
                              3)

        frame = cv2.putText(frame, str(round(conf, 3)),
                            (COORD_ARR[2] - 40, COORD_ARR[3]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0),
                            5, cv2.LINE_AA
                            )

        frame = cv2.putText(frame, str(round(conf, 3)),
                            (COORD_ARR[2] - 40, COORD_ARR[3]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0),
                            2, cv2.LINE_AA
                            )

    return np.concatenate([original_frame, frame], axis=0)


def video_frame_generator(video_file_path: str):
    video = cv2.VideoCapture(video_file_path)
    while True:
        frame = video.read()[1]
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YoloV5Detector(
        DETECTOR_ARCHIEVE,
        device=device,
        iou_thres=0.5,
        conf_thres=0.5
    )
    process_video(model, VIDEO_PATH)
