"""
train_model.py — Train ASL Sign Language Classifier
=============================================================
Trains a Random Forest classifier on collected ASL hand landmark data.
Supports:
- ASL (American Sign Language): 42 features (single hand).

Usage:
    python train_model.py

Reads CSV data from 'data/asl/' and saves models to 'model/asl/'.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import time


# ─── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "model"


def load_data():
    """Load all CSV data files for ASL and combine into a single dataset."""
    data_dir = os.path.join(DATA_DIR, "asl")
    expected_features = 42

    all_data = []
    all_labels = []

    if not os.path.exists(data_dir):
        print(f"  ERROR: '{data_dir}' folder not found!")
        print("  Run collect_data.py first to collect training data.")
        return None, None

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"  ERROR: No CSV files found in '{data_dir}'!")
        print("  Run collect_data.py first to collect training data.")
        return None, None

    print(f"  Found {len(csv_files)} letter files:\n")

    for f in sorted(csv_files):
        letter = f.replace(".csv", "").upper()
        filepath = os.path.join(data_dir, f)
        df = pd.read_csv(filepath)

        samples = len(df)
        values = df.values

        # Validate and fix feature size
        if values.shape[1] == expected_features:
            pass  # Correct
        elif values.shape[1] == 87:
            # Legacy ISL data accidentally placed in ASL dir — take first 42 features
            values = values[:, :42]
        else:
            print(f"    [SKIP] {letter}: unexpected feature count {values.shape[1]}, skipping.")
            continue

        all_data.append(values)
        all_labels.extend([letter] * samples)

        bar = "#" * min(samples // 10, 30)
        print(f"    {letter}: {samples:>4} samples  {bar}")

    if not all_data:
        print("\n  ERROR: No valid data loaded!")
        return None, None

    X = np.vstack(all_data)
    y = np.array(all_labels)

    print(f"\n  Total: {len(y)} samples across {len(csv_files)} letters\n")
    return X, y


def augment_asl(X_train, y_train):
    """Apply symmetric hand-mirroring and coordinate jitter data augmentation for ASL."""
    print("  Applying symmetric hand-mirroring and jitter data augmentation...")
    X_aug = []
    y_aug = []

    for features, label in zip(X_train, y_train):
        # 1. Original
        X_aug.append(features)
        y_aug.append(label)

        # 2. Mirrored on X-axis (even indices in interleaved layout [x0, y0, x1, y1, ...])
        mirrored = features.copy()
        mirrored[0::2] = -mirrored[0::2]
        X_aug.append(mirrored)
        y_aug.append(label)

        # 3. Jittered Original
        jittered = features.copy()
        noise = np.random.normal(0, 0.015, size=features.shape)
        # Wrist is at index 0 and 1 (x0, y0), keep it exactly at [0.0, 0.0]
        noise[0] = 0.0
        noise[1] = 0.0
        jittered += noise
        X_aug.append(jittered)
        y_aug.append(label)

        # 4. Jittered Mirrored
        jittered_mirrored = mirrored.copy()
        noise_mirrored = np.random.normal(0, 0.015, size=mirrored.shape)
        noise_mirrored[0] = 0.0
        noise_mirrored[1] = 0.0
        jittered_mirrored += noise_mirrored
        X_aug.append(jittered_mirrored)
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)


def train_model(X, y):
    """Train Random Forest classifier on landmark data."""
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train/test split (stratify to keep class distribution balanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples\n")

    # Apply data augmentation on training set
    X_train, y_train = augment_asl(X_train, y_train)
    print(f"  Training set (after ASL augmentation): {len(X_train)} samples\n")

    # Train Random Forest
    print("  Training Random Forest classifier...")
    start_time = time.time()

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.1f} seconds\n")

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("  " + "=" * 50)
    print(f"  ACCURACY: {accuracy * 100:.1f}%")
    print("  " + "=" * 50)

    # Detailed report
    print("\n  Classification Report:")
    print("  " + "-" * 50)
    target_names = encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names)
    for line in report.split('\n'):
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n  Most confused letter pairs:")
    print("  " + "-" * 50)
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confusions.append((target_names[i], target_names[j], cm[i][j]))

    confusions.sort(key=lambda x: x[2], reverse=True)
    for true_l, pred_l, count in confusions[:10]:
        print(f"    {true_l} -> {pred_l}: {count} times")

    if not confusions:
        print("    None - perfect classification!")

    return clf, encoder, accuracy


def save_model(clf, encoder):
    """Save trained model and label encoder to asl directory."""
    model_dir = os.path.join(MODEL_DIR, "asl")
    os.makedirs(model_dir, exist_ok=True)

    model_file = os.path.join(model_dir, "sign_language_model.pkl")
    encoder_file = os.path.join(model_dir, "label_encoder.pkl")

    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\n  [SAVED] Model saved to '{model_file}'")

    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"  [SAVED] Encoder saved to '{encoder_file}'")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("  ASL Sign Language Classifier — Model Training")
    print("=" * 60 + "\n")

    print("  Training ASL model...")
    print("  Loading data from 'data/asl/'...\n")

    # Load data
    X, y = load_data()
    if X is None:
        return

    # Check minimum samples
    unique, label_counts = np.unique(y, return_counts=True)
    min_samples = np.min(label_counts)
    min_letter = unique[np.argmin(label_counts)]

    if min_samples < 30:
        print(f"\n  [WARNING] Letter '{min_letter}' only has {min_samples} samples.")
        print("  Recommend at least 50 samples per letter for good accuracy.\n")

    # Train
    clf, encoder, accuracy = train_model(X, y)

    # Save
    save_model(clf, encoder)

    print(f"\n  * Training complete! Accuracy: {accuracy * 100:.1f}%")

    if accuracy >= 0.95:
        print("  [Excellent accuracy! Ready for real-time detection]")
    elif accuracy >= 0.85:
        print("  [Good accuracy. Collect more data for confused letters to improve]")
    else:
        print("  [Accuracy could be better. Try collecting more diverse samples]")

    print(f"\n  Next step: Run 'python detect.py' to start real-time detection!\n")


if __name__ == "__main__":
    main()
