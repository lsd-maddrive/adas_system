import logging
from pathlib import Path
import click
import shutil
import os
from maddrive_adas.utils.ya_disk import YandexDiskFolder

logger = logging.getLogger(__name__)

RTSD_URL = "https://disk.yandex.ru/d/PqR-jb-3xuKYqg"

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
def main(data_dir):
    ya_folder = YandexDiskFolder(RTSD_URL)

    dataset_dir = os.path.join(data_dir, "cityscapes")

    ya_folder.download_directory("/", dataset_dir)

    logger.info("Start unpacking")

    # Unpacking downloaded archives

    def unpack(source_fpath):
        dest_fstem = Path(source_fpath).stem
        dest_fpath = os.path.join(dataset_dir, dest_fstem)

        shutil.unpack_archive(source_fpath, dest_fpath, "zip")

    unpack(os.path.join(dataset_dir, "gtFine_trainvaltest.zip"))
    unpack(os.path.join(dataset_dir, "leftImg8bit_trainvaltest.zip"))
