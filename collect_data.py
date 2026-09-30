"""
collect_data.py — ASL Data Collection Tool
=================================================================
Captures hand landmarks from your webcam for training ASL classifiers.
Supports:
- ASL (American Sign Language): Single-handed, 42 features.

Usage:
    python collect_data.py

Controls:
    Press A-Z    → Select the letter you want to record
    Hold sign    → Samples are captured automatically while hand visible
    Press SPACE  → Pause/resume recording
    Press ESC    → Save data and quit

Data is saved to the 'data/asl/' folder as CSV files.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from utils import (
    extract_asl_features, draw_hand_skeleton, draw_ui_panel,
    put_text_with_bg, draw_gradient_bar, Colors, HandDetector
)


# ─── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
SAMPLES_PER_LETTER_TARGET = 300
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def get_existing_counts(data_dir):
    """Count existing samples per letter from saved CSVs in specific directory."""
    counts = {}
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                letter = f.replace(".csv", "").upper()
                try:
                    df = pd.read_csv(os.path.join(data_dir, f))
                    counts[letter] = len(df)
                except Exception:
                    counts[letter] = 0
    return counts


def save_data(letter, data, data_dir):
    """Save collected landmark data to CSV in the ASL folder."""
    if not data:
        return

    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"{letter}.csv")

    # 42 features: active hand landmarks
    columns = []
    for i in range(21):
        columns.append(f"hand_x{i}")
    for i in range(21):
        columns.append(f"hand_y{i}")

    df_new = pd.DataFrame(data, columns=columns)

    if os.path.exists(filepath):
        try:
            df_existing = pd.read_csv(filepath)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new

    df.to_csv(filepath, index=False)
    print(f"  * Saved {len(df_new)} samples for '{letter}' (total: {len(df)})")


def draw_collection_ui(frame, current_letter, sample_count, is_recording,
                       hands_detected, counts, fps, num_hands):
    """Draw the data collection UI overlay."""
    h, w = frame.shape[:2]

    # ── Top panel: Current status ────────────────────────────────────────
    draw_ui_panel(frame, 0, 0, w, 100, alpha=0.80)

    # Title
    put_text_with_bg(frame, "ASL SIGN LANGUAGE DATA COLLECTOR", (20, 35),
                     font_scale=0.75, color=Colors.NEON_CYAN, thickness=2)

    # Language badge
    put_text_with_bg(frame, "ASL Mode (1 Hand)", (480, 35),
                     font_scale=0.5, color=Colors.NEON_CYAN, thickness=2)

    # FPS
    put_text_with_bg(frame, f"FPS: {fps:.0f}", (w - 130, 35),
                     font_scale=0.5, color=Colors.TEXT_DIM, thickness=1)

    # Current letter & status
    if current_letter:
        letter_color = Colors.NEON_GREEN if is_recording else Colors.NEON_ORANGE
        status_text = "RECORDING" if is_recording else "PAUSED"
        put_text_with_bg(frame, f"Letter: {current_letter}", (20, 75),
                         font_scale=0.9, color=letter_color, thickness=2)
         # Clear status indicator
        put_text_with_bg(frame, f"| {status_text}", (200, 75),
                         font_scale=0.6, color=letter_color, thickness=1)
        put_text_with_bg(frame, f"Samples: {sample_count}/{SAMPLES_PER_LETTER_TARGET}",
                         (320, 75), font_scale=0.55, color=Colors.TEXT_WHITE, thickness=1)

        # Progress bar
        progress = min(sample_count / SAMPLES_PER_LETTER_TARGET, 1.0)
        bar_x, bar_y = 560, 60
        bar_w, bar_h = 250, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      Colors.TEXT_DIM, 1)
        draw_gradient_bar(frame, (bar_x + 1, bar_y + 1),
                          (bar_x + bar_w - 1, bar_y + bar_h - 1),
                          Colors.NEON_BLUE, Colors.NEON_GREEN, progress)
    else:
        put_text_with_bg(frame, "Press A-Z to select a letter", (20, 75),
                         font_scale=0.7, color=Colors.TEXT_GRAY, thickness=1)

    # Hand detection indicator
    if num_hands >= 1:
        indicator_color = Colors.NEON_GREEN
        indicator_text = "HAND READY"
    else:
        indicator_color = (60, 60, 200)
        indicator_text = "NO HAND"

    cv2.circle(frame, (w - 180, 70), 8, indicator_color, -1)
    put_text_with_bg(frame, indicator_text, (w - 165, 75),
                     font_scale=0.45, color=indicator_color, thickness=1)

    # ── Bottom panel: Letter grid showing progress ───────────────────────
    panel_h = 80
    draw_ui_panel(frame, 0, h - panel_h, w, panel_h, alpha=0.80)

    # Draw letter grid
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    start_x = 20
    spacing = (w - 40) // 27

    for i, letter in enumerate(letters):
        x = start_x + i * spacing
        y = h - panel_h + 25

        count = counts.get(letter, 0)

        # Color based on completion
        if count >= SAMPLES_PER_LETTER_TARGET:
            color = Colors.NEON_GREEN
        elif count > 0:
            color = Colors.NEON_ORANGE
        else:
            color = Colors.TEXT_DIM

        # Highlight current letter
        if letter == current_letter:
            cv2.rectangle(frame, (x - 8, y - 18), (x + 16, y + 8),
                          Colors.NEON_CYAN, 2)

        put_text_with_bg(frame, letter, (x, y),
                         font_scale=0.5, color=color, thickness=1)

        # Sample count below letter
        count_text = str(count) if count > 0 else "-"
        put_text_with_bg(frame, count_text, (x - 2, y + 22),
                         font_scale=0.3, color=Colors.TEXT_DIM, thickness=1)

    # ── Instructions ─────────────────────────────────────────────────────
    instructions = [
        "Press A-Z: Select letter",
        "Press _ or -: Select SPACE sign",
        "SPACE: Pause/Resume",
        "ESC: Save & Quit"
    ]
    for i, inst in enumerate(instructions):
        put_text_with_bg(frame, inst, (w - 240, 120 + i * 25),
                         font_scale=0.45, color=Colors.TEXT_DIM, thickness=1)


def main():
    """Main data collection loop."""
    print("\n" + "=" * 60)
    print("  ASL Sign Language Data Collection Tool")
    print("=" * 60)
    print("\n  Supports ASL (1 hand, 42 features).")
    print("  Press A-Z to select a letter, then hold the sign.\n")

    # Initialize Hand Detector — detect 1 hand for ASL
    detector = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("  ERROR: Could not open webcam!")
        return

    # State variables
    data_dir = os.path.join(DATA_DIR, "asl")
    os.makedirs(data_dir, exist_ok=True)

    current_letter = None
    collected_data = {}  # {letter: [samples]}
    counts = get_existing_counts(data_dir)
    is_recording = True
    prev_time = time.time()
    last_key_time = 0  # For keyboard debouncing

    print("  Webcam opened. Starting in ASL mode...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        h, w = frame.shape[:2]

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time

        # Process frame with hand detector
        multi_hand_landmarks, multi_handedness = detector.process(frame)

        hands_detected = False
        num_hands = 0
        sample_count = counts.get(current_letter, 0) + len(collected_data.get(current_letter, []))

        if multi_hand_landmarks:
            hands_detected = True
            num_hands = len(multi_hand_landmarks)

            # Draw skeleton
            draw_hand_skeleton(frame, multi_hand_landmarks[0], w, h)

            # Collect data if recording and a letter is selected
            if current_letter and is_recording:
                normalized = extract_asl_features(multi_hand_landmarks, multi_handedness)
                
                if current_letter not in collected_data:
                    collected_data[current_letter] = []
                collected_data[current_letter].append(normalized)

        # Draw UI
        draw_collection_ui(frame, current_letter, sample_count,
                           is_recording, hands_detected, counts, fps, num_hands)

        # Display
        cv2.imshow("Sign Language Data Collector", frame)

        # Handle key presses with debounce
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == ord(' '):
            if curr_time - last_key_time > 0.3:
                is_recording = not is_recording
                status = "RECORDING" if is_recording else "PAUSED"
                print(f"  [{status}]")
                last_key_time = curr_time
        elif 65 <= key <= 90 or 97 <= key <= 122 or key == ord('_') or key == ord('-'):
            # A-Z or a-z or _ or -
            new_letter = "_" if key in (ord('_'), ord('-')) else chr(key).upper()

            # Save current letter data before switching
            if current_letter and current_letter in collected_data:
                save_data(current_letter, collected_data[current_letter], data_dir)
                counts[current_letter] = counts.get(current_letter, 0) + len(collected_data[current_letter])
                collected_data[current_letter] = []

            current_letter = new_letter
            print(f"  → Selected letter: {current_letter}")

    # Save any remaining data
    for letter, data in collected_data.items():
        if data:
            save_data(letter, data, data_dir)

    # Cleanup
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n  ✓ Data collection complete!")
    print("  All progress saved successfully.\n")


if __name__ == "__main__":
    main()
