# Raspberry Pi Banknote Classification (MobileNetV2 + ONNX)

## 📌 Overview

This project provides a complete pipeline for capturing images on a
**Raspberry Pi**, training a **MobileNetV2** deep-learning model to
classify Algerian banknotes (1000 DA, 2000 DA, unknown), and deploying
the model in both **PyTorch** and **ONNX** formats for real-time
inference.

## Project Structure

    ├── Data_Acquisition/take_pic.py
    ├── Training_Model/Mobile_net_V2.pth
    ├── Training_Model/Mobile_net_V2.onnx
    ├── Training_Model/mobilenet.ipynb
    ├── Inference_on_RPI/classify_bills.py

## Image Capture

Use `Data_Acquisition/take_pic.py` on Raspberry Pi to capture dataset images.

Controls: - **s**: save image - **q**: quit preview

## Model Training (MobileNetV2)

Training steps include: - Resize to 224×224\
- Grayscale → 3‑channel\
- Normalization\
- Data augmentation\
- Modified MobileNetV2 classifier

Saved model: `Mobile_net_V2.pth`

## ONNX Export

The trained model is exported to ONNX using:

    torch.onnx.export(...)

Saved model: `Mobile_net_V2.onnx`

## ONNX Runtime Test

The notebook includes a full ONNX inference example using `onnxruntime`.

## Requirements

    torch
    torchvision
    opencv-python
    picamera2
    onnxruntime
    numpy
    Pillow

## Deployment
Use `Inference_on_RPI/classify_bills.py` on Raspberry Pi to run AI model and classify bills.
We are using PyTorch for Python inference or ONNX Runtime (Python/C++) for
optimized, embedded deployment.
