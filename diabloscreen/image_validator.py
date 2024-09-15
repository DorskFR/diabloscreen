import logging

import cv2
import imagehash
import numpy as np
from PIL import Image

from diabloscreen.config import Config
from diabloscreen.processing_stats import ProcessingStats

logger = logging.getLogger(__name__)


class ImageValidator:
    def __init__(self, config: Config, stats: ProcessingStats) -> None:
        self.config = config
        self.stats = stats
        self.saved_hashes = self._load_existing_hashes()

    def _load_existing_hashes(self) -> list[imagehash.ImageHash]:
        # TODO: Replace this with database query
        hashes = []
        for img_path in self.config.output_dir.glob("*.png"):
            with Image.open(img_path) as img:
                hashes.append(imagehash.average_hash(img))
        return hashes

    @staticmethod
    def _calculate_text_area_ratio(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return np.sum(binary == 255) / binary.size

    @staticmethod
    def _analyze_edge_detection(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges == 255) / edges.size

    @staticmethod
    def _analyze_color_histogram(image: np.ndarray) -> float:
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return np.std(hist)

    def _score_image_quality(self, image: np.ndarray) -> float:
        text_ratio = self._calculate_text_area_ratio(image)
        edge_score = self._analyze_edge_detection(image)
        color_score = self._analyze_color_histogram(image)

        norm_text_ratio = min(text_ratio / 0.4, 1.0)
        norm_edge_score = 1 - min(edge_score / 0.1, 1.0)
        norm_color_score = 1 - min(color_score / 50, 1.0)

        return 0.4 * norm_text_ratio + 0.3 * norm_edge_score + 0.3 * norm_color_score

    @staticmethod
    def _compute_hash(image: np.ndarray) -> imagehash.ImageHash:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return imagehash.average_hash(pil_image)

    def _is_similar(self, new_hash: imagehash.ImageHash) -> bool:
        return any(new_hash - saved_hash <= self.config.similarity_threshold for saved_hash in self.saved_hashes)

    def _is_valid_image(self, image: np.ndarray) -> tuple[bool, float]:
        score = self._score_image_quality(image)
        return score > self.config.image_quality_threshold, score

    def validate(self, image: np.ndarray) -> bool:
        valid, score = self._is_valid_image(image)
        if valid:
            img_hash = self._compute_hash(image)
            if self._is_similar(img_hash):
                self.stats.increment("similar")
                return False
            self.saved_hashes.append(img_hash)
            self.stats.increment("saved")
            return True
        self.stats.increment("partial_opacity")
        return False
