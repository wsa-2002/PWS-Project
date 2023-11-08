import os
import re
import shutil
from typing import List, Tuple

import cv2
from pytube import YouTube
from PIL import Image

from base import do
import log
from utils.identifier import are_different_images
from persistence.s3 import temp

TEMP_PDF_FILENAME = 'temp.pdf'


class SheetExtractor:

    def __init__(self, url, interval: float = 1, identify_threshold: float = 0.07):
        self.yt = YouTube(url)
        self.filename = self.yt.streams.filter(progressive=True, file_extension='mp4').order_by(
            'resolution').desc().first().download()

        self.interval = interval
        self.identify_threshold = identify_threshold

    def __enter__(self, file_extension: str = 'mp4', dir_name: str = 'temp_image') -> Tuple[do.S3File, str]:
        try:
            self.dir_name = dir_name
            os.mkdir(self.dir_name)
            self.extract(dir_name=self.dir_name, filename=self.filename, interval=self.interval)
            self.batch_crop_images(self.dir_name)
            filenames = sorted(list(filter(lambda x: True if 'crop' in x else False, os.listdir(self.dir_name))),
                               key=lambda x: int(re.findall(r'\d+', x)[0]))

            preserved_images = [filenames[0]]
            for i in range(len(filenames) - 1):
                img_1 = cv2.imread(f"{self.dir_name}/{filenames[i]}")
                img_2 = cv2.imread(f"{self.dir_name}/{filenames[i + 1]}")
                if are_different_images(img_1, img_2, threshold=self.identify_threshold):
                    log.info('different image')
                    preserved_images.append(filenames[i+1])
            preserved_images = sorted(preserved_images, key=lambda x: int(re.findall(r'\d+', x)[0]))
            upload_file = self.compose_and_upload_images(filenames=preserved_images, dir_name=self.dir_name)
        finally:
            pass
            os.remove(self.filename)
            shutil.rmtree(self.dir_name)
        return upload_file, self.filename

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @classmethod
    def extract(cls, dir_name: str, filename=None, interval: float = 1):

        file = cv2.VideoCapture(filename)
        fps = round(file.get(cv2.CAP_PROP_FPS))

        idx = 0
        frame_count = 0
        while True:
            # Capture frame-by-frame
            ret, frame = file.read()

            # if frame is read correctly ret is True
            if not ret:
                log.info('File fetch finished.')
                break

            # fetch frame by pre-defined interval
            frame_count += 1
            if frame_count % int(fps * interval):
                continue

            # save selected image
            output_filename = f'{dir_name}/temp_{idx}.png'
            cv2.imwrite(output_filename, frame)
            idx += 1
        file.release()  # TODO: maybe use context manager?

    @staticmethod
    def batch_crop_images(dir_name: str):
        filenames = os.listdir(dir_name)
        for filename in filenames:
            SheetExtractor.crop_image(f'{dir_name}/{filename}')

    @staticmethod
    def crop_image(file_path: str, x_point: int = 0, y_point: int = 0, height: int = 350, width: int = 1280):
        image = cv2.imread(file_path)

        crop = image[y_point:y_point + height, x_point:x_point + width]
        dir_name, file_name = file_path.split('/')
        cv2.imwrite(f'{dir_name}/crop_{file_name}', crop)

    @staticmethod
    def compose_and_upload_images(filenames: List[str], dir_name: str, file_extension='PDF') -> do.S3File:
        images = [cv2.imread(f"{dir_name}/{filename}") for filename in filenames]
        image_pages = [cv2.vconcat(images[i:i+5]) for i in range(0, len(filenames), 5)]
        image_pages = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in image_pages]
        image_pages[0].save(TEMP_PDF_FILENAME, file_extension,
                            resolution=100, save_all=True, append_images=image_pages[1:])

        with open(TEMP_PDF_FILENAME, 'rb') as file:
            s3_file = temp.upload(file)
        try:
            os.remove(TEMP_PDF_FILENAME)
        except OSError:
            pass
        return s3_file
