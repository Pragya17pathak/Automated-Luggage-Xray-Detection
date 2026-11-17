# Automated Luggage X-Ray Threat Detection

An AI-powered research prototype that detects prohibited objects in luggage X-ray images using YOLOv8-Nano object detection.

## 1. Project Overview

The system analyzes X-ray images and identifies potentially prohibited objects using a trained deep learning model.

It detects five object categories:
- Gun
- Knife
- Pliers
- Scissors
- Wrench

The application provides an interactive Streamlit interface where users can upload X-ray images, select a confidence threshold, view detected objects, and download annotated results.

## 2. Objectives

- Detect prohibited objects in luggage X-ray images.
- Apply YOLOv8-Nano for object detection.
- Identify multiple object categories in a single image.
- Evaluate the model using precision, recall, and mean Average Precision.
- Develop a user-friendly detection interface.

## 3. Technologies Used

- Python 3.11
- YOLOv8-Nano
- PyTorch
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Pillow

## 4. Dataset Description

Dataset: SIXray_v3 Roboflow export.

Dataset details:
- Total images: 14,131
- Training images: 11,638
- Validation images: 1,662
- Testing images: 831
- Number of classes: 5
- Image size: 640 × 640

The dataset contains YOLO-format bounding-box annotations.

The exported dataset uses grayscale images with adaptive contrast preprocessing.

## 5. Model Architecture

The project uses YOLOv8-Nano, a lightweight object detection model.

The model is initialized with pretrained YOLOv8-Nano weights and fine-tuned on the luggage X-ray dataset.

Training configuration:
- Model: YOLOv8-Nano
- Epochs: 30
- Image size: 416 × 416
- Batch size: 2
- Optimizer: AdamW
- Learning rate: 0.001
- Device: CPU

## 6. Model Evaluation

The final trained model was evaluated on the separate test dataset containing 831 images and 1,593 object instances.

### Overall Test Results

| Metric | Result |
|---|---:|
| Precision | 80.87% |
| Recall | 64.17% |
| mAP@50 | 74.85% |
| mAP@50–95 | 47.18% |

### Class-wise Test Results

| Class | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Gun | 92.5% | 88.5% | 95.3% |
| Knife | 67.7% | 52.1% | 60.9% |
| Pliers | 84.9% | 65.5% | 79.2% |
| Scissors | 78.9% | 62.5% | 73.5% |
| Wrench | 80.3% | 52.3% | 65.4% |

These are test-set results, not training accuracy.

## 7. Project Structure

```text
Automated-Luggage-Xray-Detection/
│
├── dataset/
│   ├── images/
│   └── labels/
│
├── models/
│
├── results/
│   ├── xray_detector/
│   ├── xray_detector_improved/
│   ├── final_test_evaluation/
│   └── predictions/
│
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── evaluate.py
│   └── validate_dataset.py
│
├── uploads/
├── app.py
├── data.yaml
├── requirements.txt
├── README.md
└── yolov8n.pt
```

## 8. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Pragya17pathak/Automated-Luggage-Xray-Detection.git
```

Open the project directory:

```bash
cd Automated-Luggage-Xray-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset path in `data.yaml` points to your extracted dataset directory.

## 9. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local address displayed in the terminal.

Upload an X-ray image, adjust the confidence threshold, and click **Detect Threats**.

## 10. Model Files

The trained model weights are saved in:

```text
results/xray_detector_improved/weights/best.pt
```

The original training run weights are saved in:

```text
results/xray_detector/weights/
```

Use the improved model path in the application if that is the model you intend to demonstrate.

## 11. Limitations

- Detection performance varies across object categories.
- Small or heavily occluded objects can be difficult to identify.
- Lower confidence thresholds may increase false positives.
- The model requires further training and validation before real-world deployment.

## 12. Future Improvements

- Improve detection of knives and wrenches.
- Experiment with higher image resolutions.
- Increase training data diversity.
- Explore attention-based detection methods.
- Optimize inference speed.
- Evaluate performance on additional X-ray datasets.

## 13. Disclaimer

This project is an academic research prototype. It is not certified for operational airport security screening and should not replace professional security inspection systems.
