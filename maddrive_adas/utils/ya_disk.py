import os
import requests
from tqdm import tqdm
import urllib.request

import logging

from .fs import get_file_md5


logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class YandexDiskFolder:
    def __init__(self, public_url) -> None:
        self._pub_url = public_url

    def get_files_list(self, path):
        response = requests.get(
            "https://cloud-api.yandex.net/v1/disk/public/resources",
            params={"public_key": self._pub_url, "path": path},
        )

        if response.status_code != 200:
            raise ValueError("Failed to request files list")

        resp_json = response.json()

        files = []
        disk_items = resp_json["_embedded"]["items"]
        for item in disk_items:
            path = item["path"]
            name = item["name"]
            type_ = item["type"]

            if type_ != "file":
                logger.info(f"Path '{path}' contains non-file ({type_}): {path}")
                continue

            sha256 = item["sha256"]
            md5 = item["md5"]
            files.append({"path": path, "name": name, "sha256": sha256, "md5": md5})

        return files

    def download_file(self, path, output_fpath):
        response = requests.get(
            "https://cloud-api.yandex.net/v1/disk/public/resources/download",
            params={"public_key": self._pub_url, "path": path},
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to download file: {path}")

        resp_json = response.json()

        url = resp_json["href"]

        tqdm_desc = url.split("/")[-1]
        tqdm_desc = tqdm_desc.split("&")[0]
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=tqdm_desc) as t:
            urllib.request.urlretrieve(url, filename=output_fpath, reporthook=t.update_to)

    def download_directory(self, path, output_dirpath):
        dirname = path.split("/")[-1]

        result_directory = os.path.join(output_dirpath, dirname)
        logger.info(f"Downloading to {result_directory}")
        os.makedirs(result_directory, exist_ok=True)

        files = self.get_files_list(path)

        for file_info in files:
            name = file_info["name"]
            file_path = file_info["path"]
            output_fpath = os.path.join(result_directory, name)

            if os.path.exists(output_fpath):
                md5_file = get_file_md5(output_fpath)
                md5_item = file_info["md5"]

                logger.debug(f"{md5_file} vs {md5_item}")
                if md5_file == md5_item:
                    logger.info(f"File {file_path} skipped - already exists")
                    continue

            logger.info(f"Downloading {file_path} to {output_fpath}")
            self.download_file(file_path, output_fpath)
