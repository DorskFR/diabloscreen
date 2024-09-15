import logging
import os
import time
from datetime import UTC, datetime

import cv2
import mss
import numpy as np
import sentry_sdk

from diabloscreen.config import Config
from diabloscreen.image_processor_ultralytics import ImageProcessor
from diabloscreen.image_validator import ImageValidator
from diabloscreen.processing_stats import ProcessingStats

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    debug = os.getenv("LOGGING_DEBUG", "").lower() in ("true", "1", "t")
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_sentry() -> None:
    if dsn := os.getenv("DIABLO_SENTRY_DSN"):
        sentry_sdk.init(dsn=dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized")
    else:
        logger.info("Sentry DSN not provided. Error tracking is disabled.")

    def process(self, img: np.ndarray) -> None:
        cropped_images = self._detect_and_crop(img)
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")


def main() -> None:
    config = Config.from_env()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sct = mss.mss()
    monitor = sct.monitors[1]
    stats = ProcessingStats()
    image_validator = ImageValidator(config, stats)
    image_processor = ImageProcessor(config, stats)

    try:
        while True:
            img = np.array(sct.grab(monitor))[:, :, :3]
            for item in image_processor.process(img):
                if not image_validator.validate(item):
                    continue
                timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
                filename = config.output_dir / f"item_{timestamp}.png"
                cv2.imwrite(str(filename), item)
                logger.debug(f"Saved new image: {filename}")

            stats.increment("screenshots")
            if stats.counter["screenshots"] % 100 == 0:
                logger.debug(f"Processed {stats.counter['screenshots']} screenshots")

            if config.loop_delay > 0:
                time.sleep(config.loop_delay)
    finally:
        stats.log_stats()


if __name__ == "__main__":
    setup_logging()
    setup_sentry()
    logger.info("Starting Diablo screenshot processing...")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Calculating results...")
    except Exception as e:
        logger.exception("An error occurred during execution")
        sentry_sdk.capture_exception(e)
