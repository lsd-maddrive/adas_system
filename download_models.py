import os

_encoder_url: str = \
    'https://mega.nz/file/0JdRQChb#fPK3-DbvvZQSiOvCvGUfi3z0VZn1WJJZx5hmVVDTF3U'
_detector_archive_url: str = \
    'https://mega.nz/file/QVVgQD7J#GtmX9qdMuzPmL4nLWJsK0pSdzprCWGf-bTwSMmKmOqc'

_encoder_output_file_path: str = 'encoder_archive'
_detector_output_file_path: str = 'detector_archive'

if __name__ == '__main__':
    try:
        from mega import Mega
        print('Downloading models')
        m = Mega().login()

        if not os.path.isfile(_encoder_output_file_path):
            try:
                print('Downloading Encoder.')
                m.download_url(_encoder_url, dest_filename=_encoder_output_file_path)
            except PermissionError:
                pass
        else:
            print(f'{_encoder_output_file_path} exists! Skip downloading Encoder!')

        if not os.path.isfile(_detector_output_file_path):
            try:
                print('Downloading Detector.')
                m.download_url(_detector_archive_url, dest_filename=_detector_output_file_path)
            except PermissionError:
                pass
        else:
            print(f'{_detector_output_file_path} exists! Skip downloading Detector!')

    except ImportError:
        print('Unable to import mega.')
        print("Use: pip install mega.py")
