import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass
class Config:
    model_path: Path
    output_dir: Path
    loop_delay: float
    confidence_threshold: float
    similarity_threshold: float
    image_quality_threshold: float

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            Path(os.getenv("DIABLO_MODEL_PATH", "./yolov8diablo.pt")),
            Path(os.getenv("DIABLO_OUTPUT_DIR", "./items")),
            float(os.getenv("DIABLO_LOOP_DELAY", "1")),
            float(os.getenv("DIABLO_CONFIDENCE_THRESHOLD", "0.40")),
            float(os.getenv("DIABLO_SIMILARITY_THRESHOLD", "5")),
            float(os.getenv("DIABLO_IMAGE_QUALITY_THRESHOLD", "0.75")),
        )
