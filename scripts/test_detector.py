import argparse

import cv2
import torch
import numpy as np
import datetime

from maddrive_adas.sign_det.detector import YoloV5Detector
from maddrive_adas.sign_det.base import DetectedInstance


def main():
    args = parse_args()
    device = resolve_device()

    model = YoloV5Detector(
        args.checkpoint_path,
        device=device,
        iou_thres=0.5,
        conf_thres=0.5
    )

    process_video(model, args.video)


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
        show_frame(frame)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def show_frame(frame):
    cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


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


def resolve_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    usage_example = '''python ./scripts/test_detector.py
    -v ./SignDetectorAndClassifier/data/reg_videos/1.mp4
    -c ./detector_archive'''

    parser = argparse.ArgumentParser(epilog=usage_example)

    parser.add_argument(
        '-v',
        '--video',
        required=True,
        help='Path to video file'
    )

    parser.add_argument(
        '-c',
        '--checkpoint_path',
        help='Path to YoloV5 checkpoint weights'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
