from typing import List, Tuple

import argparse

import cv2
import torch
import numpy as np
import datetime

from maddrive_adas.sign_det.base import DetectedInstance, AbstractComposer
from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.detector import YoloV5Detector


def main():
    args = parse_args()
    model = build_composer(args)

    process_video(model, args.video)


def build_composer(args: argparse.Namespace) -> BasicSignsDetectorAndClassifier:
    classifier = EncoderBasedClassifier(
        config_path=args.classifier_checkpoint,
        path_to_subclassifier_3_24_and_3_25_config=args.subclassifier_checkpoint
    )

    detector = YoloV5Detector(
        model_archive_file_path=args.detector_checkpoint,
        iou_thres=0.5,
        conf_thres=0.5,
        device=resolve_device()
    )

    composer: AbstractComposer = BasicSignsDetectorAndClassifier(
        classifier=classifier,
        detector=detector
    )

    return composer


def process_video(model: BasicSignsDetectorAndClassifier, video_file_path: str):
    video_frame_gen = video_frame_generator(video_file_path)

    for frame in video_frame_gen:
        t0 = datetime.datetime.now()
        pred = model.detect_and_classify(frame)
        video_frame = plot_predictions(pred, frame)
        video_frame = image_resize(video_frame, height=920)
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


def plot_predictions(pred: Tuple[DetectedInstance, List[Tuple[str, float]]], frame: np.ndarray):
    delta = 15
    color = (0, 255, 0)
    original_frame = frame.copy()
    detected_instance, predicted_signs = pred
    for idx, sign_conf in enumerate(predicted_signs):
        sign = sign_conf[0]
        conf_classifier = sign_conf[1]

        COORD_ARR, conf_detector = detected_instance.get_abs_roi(idx)

        frame = cv2.rectangle(
            frame,
            (COORD_ARR[0], COORD_ARR[1]),
            (COORD_ARR[2], COORD_ARR[3]),
            color,
            3
        )

        substring = f'{sign}:C:{round(conf_classifier, 2)}|D:{round(conf_detector, 2)}'

        frame = cv2.putText(frame, substring,
                            (COORD_ARR[0], COORD_ARR[3] + delta),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255),
                            3, cv2.LINE_AA
                            )

        frame = cv2.putText(frame, substring,
                            (COORD_ARR[0], COORD_ARR[3] + delta),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0),
                            1, cv2.LINE_AA
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
    usage_example = '''python ./scripts/test_composer.py
    -v ./SignDetectorAndClassifier/data/reg_videos/1.mp4
    -d ./detector_archive
    -c ./encoder_archive
    -s ./subclassifier_archive'''

    parser = argparse.ArgumentParser(epilog=usage_example)

    parser.add_argument(
        '-v',
        '--video',
        required=True,
        help='Path to video file'
    )

    parser.add_argument(
        '-d',
        '--detector_checkpoint',
        help='Path to YoloV5 checkpoint.'
    )

    parser.add_argument(
        '-c',
        '--classifier_checkpoint',
        help='Path to classifier checkpoint'
    )

    parser.add_argument(
        '-s',
        '--subclassifier_checkpoint',
        required=False,
        help='Subclussifier additional model for detect 3.24/3.25 signs.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
