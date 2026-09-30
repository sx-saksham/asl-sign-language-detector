# ASL Sign Language Detection — Real-Time Webcam

A Python computer-vision and machine-learning application that recognizes single-hand American Sign Language (ASL) alphabet signs from a webcam and builds letters into words and sentences in real time.

## What it does

- Detects one hand using MediaPipe
- Extracts and normalizes 21 hand landmarks into 42 numerical features (X/Y coordinates)
- Trains a Random Forest classifier on collected ASL samples
- Uses an 80/20 stratified train/test split
- Applies training-data augmentation using mirroring and small coordinate noise
- Smooths predictions to reduce flicker and repeated letters
- Builds detected letters into words and sentences
- Provides a visual hand-skeleton overlay, prediction confidence, and notepad UI

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** scikit-learn Random Forest, LabelEncoder
- **Data:** NumPy, Pandas, CSV
- **Model persistence:** Python pickle

## Pipeline

```text
Webcam
  ↓
OpenCV frame
  ↓
MediaPipe hand landmarks
  ↓
21 landmarks → normalized 42-feature vector
  ↓
Random Forest classifier
  ↓
Prediction smoothing
  ↓
Letter → Word → Sentence
```

## Run it

```bash
pip install -r requirements.txt
```

### 1. Collect data

```bash
python collect_data.py
```

The collector stores landmark features as CSV files in `data/asl/`.

### 2. Train the model

```bash
python train_model.py
```

The training script uses a Random Forest with 150 trees and evaluates the model on a held-out test set.

### 3. Start real-time detection

```bash
python detect.py
```

Use the keyboard controls shown by the application to clear the sentence, add spaces, delete characters, or exit.

## Dataset and model

This repository includes the collected landmark CSV data and the trained model used by the project. The included model should be treated as a project-specific model trained on the included dataset rather than as a general-purpose ASL recognizer.

## Limitations

- The current detector is designed for the ASL alphabet and single-hand signs.
- Recognition quality depends on the training data, camera angle, lighting, and hand pose.
- The project is a student/research prototype, not a certified accessibility system.
