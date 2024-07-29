# YOLO Object Detection with OpenCV✨

This repository contains a Python program for object detection using the YOLO (You Only Look Once) model with OpenCV.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This project demonstrates object detection on an input image using the YOLOv3 model. The YOLO algorithm is known for its speed and accuracy in detecting objects in real-time. The program uses OpenCV to handle image processing and visualization tasks.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yolo-object-detection.git
    cd yolo-object-detection
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python numpy
    ```

3. Download the YOLOv3 weights and configuration files:
    - [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights)
    - [YOLOv3 configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    - [COCO names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

4. Place the downloaded files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`) in the project directory.

## Usage

1. Replace `sampleimage1.jpg` with your input image in the project directory.

2. Run the program:
    ```bash
    python yolo_object_detection.py
    ```

3. The program will display the input image with detected objects outlined with bounding boxes and labeled with class names.

## Acknowledgements

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [OpenCV](https://opencv.org/)
- [COCO dataset](https://cocodataset.org/)

