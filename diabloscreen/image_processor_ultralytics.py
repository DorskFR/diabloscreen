import logging
from collections.abc import Iterator

import numpy as np
from ultralytics import YOLO

from diabloscreen.config import Config
from diabloscreen.processing_stats import ProcessingStats

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, config: Config, stats: ProcessingStats) -> None:
        self.config = config
        self.stats = stats
        self.model = YOLO(self.config.model_path, task="detect")
        self.img_height, self.img_width = 640, 640  # YOLOv8 default size

    def _detect_objects(self, img: np.ndarray) -> list:
        results = self.model(img, verbose=False)[0]
        cropped_images = []

        for i, result in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = result
            if score > self.config.confidence_threshold:
                cropped = img[int(y1) : int(y2), int(x1) : int(x2)]
                cropped_images.append((cropped, i, int(class_id)))

        return cropped_images

    def process(self, img: np.ndarray) -> Iterator[np.ndarray]:
        for cropped, _i, _class_id in self._detect_objects(img):
            yield cropped
