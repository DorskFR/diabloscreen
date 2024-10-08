
# diabloscreen

## What it is

It watches your screen and when it detects a Diablo IV item description it crops and saves it.

The crops can then be used on https://diablo.trade directly.

https://github.com/user-attachments/assets/da643ed8-5883-487e-9125-4b41a90fb0ae

## How it works

Uses a yolov8 model trained on ~600 screenshots of the game.
Recognized item descriptions are then cropped and the image is hashed.
The hash is compared with previously seen hashes to avoid saving duplicates.
Image quality is scored to remove poor images (half transparent, etc.)

## Install

Using https://rye.astral.sh/

```bash
curl -sSf https://rye.astral.sh/get | bash
git clone https://github.com/DorskFR/diabloscreen.git && cd diabloscreen
rye sync
make run
```

Tested on Linux and macos

Not tested on Windows but should work I guess (feedback welcome).

## Environment variables

Some settings can be fine tuned with environment variables:

| variable                       | required | default value   | Notes                               |
| ------------------------------ | -------- | --------------- | ----------------------------------- |
| DIABLO_MONITOR                 | No       | 0               | adjust for multi-monitor setup      |
| DIABLO_MODEL_PATH              | No       | yolov8diablo.pt | can also be .onnx                   |
| DIABLO_OUTPUT_DIR              | No       | items           |                                     |
| DIABLO_LOOP_DELAY              | No       | 1.0             | in seconds, lower requires more cpu |
| DIABLO_CONFIDENCE_THRESHOLD    | No       | 0.25            | detection                           |
| DIABLO_SIMILARITY_THRESHOLD    | No       | 1               | duplicates                          |
| DIABLO_IMAGE_QUALITY_THRESHOLD | No       | 0.75            |                                     |

## Model training

Using:
- https://github.com/ultralytics to train and run the model based on yolov8.
- https://labelstud.io/ to annotate the screenshots

## Things to improve

Ultralytics package is rather big and a little bit too magical with how it sets up dependencies based on the OS and detected GPU. We can export the model to onnx format and we should be able to use OpenCV DNN or onnxruntime to run it.

However, when doing this we get weird results: inaccurate detection, errors when cropping, etc. It appears ultralytics is doing some preprocessing, postprocessing to images to get good results. I have not figured out how to do this properly so for now still using the ultralytics package for inference/detection too.
