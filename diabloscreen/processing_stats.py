import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)


class ProcessingStats:
    def __init__(self) -> None:
        self.counter = Counter()
        self.start_time = time.time()

    def increment(self, key: str, value: int = 1) -> None:
        self.counter[key] += value

    def log_stats(self) -> None:
        elapsed = time.time() - self.start_time
        screenshots_per_second = self.counter["screenshots"] / elapsed
        detections_per_second = self.counter["detections"] / elapsed
        saved_per_second = self.counter["saved"] / elapsed

        logger.debug("\nBenchmark results:")
        logger.debug(f"Total screenshots: {self.counter['screenshots']}")
        logger.debug(f"Total detections: {self.counter['detections']}")
        logger.debug(f"Unique images saved: {self.counter['saved']}")
        logger.debug(f"Partial opacity images skipped: {self.counter['partial_opacity']}")
        logger.debug(f"Similar images skipped: {self.counter['similar']}")
        logger.debug(f"Total time: {elapsed:.2f} seconds")
        logger.debug(f"Screenshots per second: {screenshots_per_second:.2f}")
        logger.debug(f"Detections per second: {detections_per_second:.2f}")
        logger.debug(f"Unique images saved per second: {saved_per_second:.2f}")
