import os

_encoder_url: str = \
    'https://mega.nz/file/0JdRQChb#fPK3-DbvvZQSiOvCvGUfi3z0VZn1WJJZx5hmVVDTF3U'
_detector_archive_url: str = \
    'https://mega.nz/file/QVVgQD7J#GtmX9qdMuzPmL4nLWJsK0pSdzprCWGf-bTwSMmKmOqc'
_subclassifier_for_3_24_3_25_url: str = \
    'https://mega.nz/file/IdkAFBoB#o4pyZ1H1YNCZxW7_GwiKauV8bLqE1DXaZanYPwTQJwM'

_encoder_output_file_path: str = 'encoder_archive'
_detector_output_file_path: str = 'detector_archive'
_subclassifier_file_path: str = 'subclassifier_3.24_3.25_archive'


def _save_download(output_file_path, url):
    try:
        m.download_url(url, dest_filename=output_file_path)
    except PermissionError:
        pass


if __name__ == '__main__':
    try:
        from mega import Mega
        print('Downloading models')
        m = Mega().login()

        if not os.path.isfile(_encoder_output_file_path):
            _save_download(_encoder_output_file_path, _encoder_url)
        else:
            print(f'{_encoder_output_file_path} exists! Skip downloading Encoder!')

        if not os.path.isfile(_detector_output_file_path):
            _save_download(_detector_output_file_path, _detector_archive_url)
        else:
            print(f'{_detector_output_file_path} exists! Skip downloading Detector!')

        if not os.path.isfile(_subclassifier_file_path):
            _save_download(_subclassifier_file_path, _subclassifier_for_3_24_3_25_url)
        else:
            print(f'{_subclassifier_file_path} exists! Skip downloading Detector!')

    except ImportError:
        print('Unable to import mega.')
        print("Use: pip install mega.py")
