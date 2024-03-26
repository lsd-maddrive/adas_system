import os

_encoder_url: str = \
    'https://mega.nz/file/0JdRQChb#fPK3-DbvvZQSiOvCvGUfi3z0VZn1WJJZx5hmVVDTF3U'
_detector_archive_url: str = \
    'https://mega.nz/file/IIEB3AJA#eMCWAzYaLW5SqjlESpgHjMf7DQyX-c4QYa7sq85tCo0'
_subclassifier_output_file_name: str = \
    'https://mega.nz/file/IdkAFBoB#o4pyZ1H1YNCZxW7_GwiKauV8bLqE1DXaZanYPwTQJwM'

_encoder_output_filename: str = 'classifier_archive'
_detector_output_filename: str = 'detector_archive'
_subclassifier_filename: str = 'subclassifier_archive'


def _save_download(output_file_path, url):
    try:
        m.download_url(url, dest_filename=output_file_path)
        print(f'{output_file_path} done')
    except PermissionError:
        pass


if __name__ == '__main__':
    try:
        from mega import Mega
        print('Downloading models')
        m = Mega().login()

        if not os.path.isfile(_encoder_output_filename):
            _save_download(_encoder_output_filename, _encoder_url)
        else:
            print(f'{_encoder_output_filename} exists! Skip downloading Encoder!')

        if not os.path.isfile(_detector_output_filename):
            _save_download(_detector_output_filename, _detector_archive_url)
        else:
            print(f'{_detector_output_filename} exists! Skip downloading Detector!')

        if not os.path.isfile(_subclassifier_filename):
            _save_download(_subclassifier_filename, _subclassifier_output_file_name)
        else:
            print(f'{_subclassifier_filename} exists! Skip downloading Detector!')

    except ImportError:
        print('Unable to import mega.')
        print("Use: pip install mega.py")
