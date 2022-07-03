import logging
import click
import shutil
import os
from maddrive_adas.utils.ya_disk import YandexDiskFolder

logger = logging.getLogger(__name__)

RTSD_URL = "https://yadi.sk/d/TX5k2hkEm9wqZ"

DEFAULT_DATA_DIRECTORY = os.path.abspath(os.path.join(os.curdir, "data"))


@click.command()
@click.option(
    "--data-dir",
    "-d",
    default=DEFAULT_DATA_DIRECTORY,
    show_default=True,
    type=str,
    help="Path to directory to download data in",
)
@click.option(
    "--unpack-full-frames",
    default=False,
    show_default=True,
    type=bool,
    help="Unpack full frames archives after download",
)
@click.option(
    "--unpack-detection",
    default=True,
    show_default=True,
    type=bool,
    help="Unpack detection archives after download",
)
def main(data_dir, unpack_full_frames, unpack_detection):
    ya_folder = YandexDiskFolder(RTSD_URL)

    dataset_dir = os.path.join(data_dir, "rtsd")

    # NOTE - closed to avoid whole data unpacking
    # ya_folder.download_directory("/", dataset_dir)
    ya_folder.download_directory("/detection", dataset_dir)
    ya_folder.download_directory("/classification", dataset_dir)

    logger.info("Start unpacking")

    # Unpacking downloaded archives

    if unpack_full_frames:
        full_frames_arch_fpath = os.path.join(dataset_dir, "full-frames.tar.lzma")
        full_frames_result_fpath = os.path.join(dataset_dir, "full-frames")

        shutil.unpack_archive(full_frames_arch_fpath, full_frames_result_fpath, "xztar")

    if unpack_detection:
        detection_prefixes = [
            "rtsd-d1",
            "rtsd-d2",
            "rtsd-d3",
        ]

        # Unpack detection

        def unpack_detection_pair(dirpath, prefix):
            frames_fpath = os.path.join(dirpath, f"{prefix}-frames.tar.lzma")
            gt_fpath = os.path.join(dirpath, f"{prefix}-gt.tar.lzma")

            logger.info(f"Unpacking {prefix} detection data")

            shutil.unpack_archive(frames_fpath, dirpath, "xztar")
            shutil.unpack_archive(gt_fpath, dirpath, "xztar")

        detection_data_path = os.path.join(dataset_dir, "detection")
        for det_prefix in detection_prefixes:
            unpack_detection_pair(detection_data_path, det_prefix)

    # TODO - classification unpack
    classification_archives = ["rtsd-r1.tar.lzma", "rtsd-r3.tar.lzma"]
