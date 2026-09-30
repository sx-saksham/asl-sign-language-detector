"""
detect.py — Real-Time ASL Sign Language Detection & Sentence Builder
============================================================================
Detects hand signs via webcam and builds words/sentences in real-time.

Supports:
  ASL (American Sign Language): Single-handed, 42 features.

Usage:
    python detect.py

Controls:
    C          -> Clear the sentence
    ESC        -> Quit
    Spacebar   -> Add space (new word)
    Backspace  -> Delete last letter
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from utils import (
    extract_asl_features, draw_hand_skeleton, draw_ui_panel,
    put_text_with_bg, draw_gradient_bar, Colors,
    PredictionSmoother, SentenceBuilder, HandDetector
)


# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_DIR = "model"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
KEY_DEBOUNCE = 0.3  # seconds between repeated key actions


def load_model():
    """Load trained ASL model and label encoder."""
    model_dir = os.path.join(MODEL_DIR, "asl")
    model_file = os.path.join(model_dir, "sign_language_model.pkl")
    encoder_file = os.path.join(model_dir, "label_encoder.pkl")

    if not os.path.exists(model_file) or not os.path.exists(encoder_file):
        print(f"  ERROR: No ASL model found in '{model_dir}'!")
        print(f"  Run these steps first:")
        print(f"    1. python collect_data.py")
        print(f"    2. python train_model.py asl")
        return None, None

    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    with open(encoder_file, 'rb') as f:
        encoder = pickle.load(f)

    print(f"  * ASL model loaded ({len(encoder.classes_)} letters: {', '.join(encoder.classes_)})")
    return clf, encoder





def wrap_text(text, max_chars_per_line=23):
    """
    Wrap text into a list of lines of a maximum character width,
    keeping words together.
    """
    if not text:
        return []
    words = text.split(' ')
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        added_len = len(word) + (1 if current_line else 0)
        if current_length + added_len <= max_chars_per_line:
            current_line.append(word)
            current_length += added_len
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def draw_detection_ui(canvas, prediction, confidence, sentence_builder,
                      smoother, hands_detected, fps, gesture_status,
                      no_hand_time, last_action, num_hands):
    """Draw the split-pane UI overlay on the canvas (left camera, right notepad)."""
    # Canvas is 1280x720. Left pane is 960x720, Right pane is 320x720.
    
    # ── Top panel: Header (Camera feed side only) ────────────────────────
    draw_ui_panel(canvas, 0, 0, 960, 60, alpha=0.82)

    # Title
    put_text_with_bg(canvas, "ASL SIGN LANGUAGE DETECTOR", (20, 38),
                     font_scale=0.75, color=Colors.NEON_CYAN, thickness=2)

    # FPS
    put_text_with_bg(canvas, f"FPS: {fps:.0f}", (960 - 100, 38),
                     font_scale=0.45, color=Colors.TEXT_DIM, thickness=1)

    # ── Right prediction panel (Camera feed side only) ───────────────────
    panel_w = 220
    panel_h = 200
    panel_x = 960 - panel_w - 15  # 725
    panel_y = 80

    draw_ui_panel(canvas, panel_x, panel_y, panel_w, panel_h, alpha=0.80)

    if hands_detected and prediction:
        # Large letter display
        letter_scale = 3.5
        (tw, th), _ = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX,
                                       letter_scale, 4)
        letter_x = panel_x + (panel_w - tw) // 2
        letter_y = panel_y + 90

        # Color based on confidence
        if confidence > 0.8:
            letter_color = Colors.NEON_GREEN
        elif confidence > 0.6:
            letter_color = Colors.NEON_YELLOW
        else:
            letter_color = Colors.NEON_ORANGE

        cv2.putText(canvas, prediction, (letter_x, letter_y),
                     cv2.FONT_HERSHEY_SIMPLEX, letter_scale, letter_color, 5,
                     cv2.LINE_AA)

        # Confidence text
        conf_text = f"Confidence: {confidence * 100:.0f}%"
        put_text_with_bg(canvas, conf_text,
                         (panel_x + 35, panel_y + 120),
                         font_scale=0.5, color=Colors.TEXT_GRAY, thickness=1)

        # Confidence bar
        bar_x = panel_x + 20
        bar_y = panel_y + 140
        bar_w2 = panel_w - 40
        bar_h2 = 12

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w2, bar_y + bar_h2),
                      Colors.TEXT_DIM, 1)

        if confidence > 0.8:
            bar_color_start, bar_color_end = Colors.NEON_BLUE, Colors.NEON_GREEN
        elif confidence > 0.6:
            bar_color_start, bar_color_end = Colors.NEON_ORANGE, Colors.NEON_YELLOW
        else:
            bar_color_start, bar_color_end = (60, 60, 200), Colors.NEON_ORANGE

        draw_gradient_bar(canvas, (bar_x + 1, bar_y + 1),
                          (bar_x + bar_w2 - 1, bar_y + bar_h2 - 1),
                          bar_color_start, bar_color_end, confidence)

        # Stability indicator
        current, stability = smoother.get_current()
        stab_text = f"Stability: {stability * 100:.0f}%"
        put_text_with_bg(canvas, stab_text,
                         (panel_x + 45, panel_y + 180),
                         font_scale=0.45, color=Colors.TEXT_DIM, thickness=1)

    elif hands_detected:
        put_text_with_bg(canvas, "Analyzing...",
                         (panel_x + 50, panel_y + 90),
                         font_scale=0.6, color=Colors.TEXT_GRAY, thickness=1)
    else:
        put_text_with_bg(canvas, "Show hand",
                         (panel_x + 45, panel_y + 80),
                         font_scale=0.6, color=Colors.TEXT_DIM, thickness=1)
        put_text_with_bg(canvas, "to camera",
                         (panel_x + 55, panel_y + 110),
                         font_scale=0.6, color=Colors.TEXT_DIM, thickness=1)

    # ── Gesture status overlay (Centered in camera feed) ───
    if gesture_status:
        status_color = Colors.NEON_YELLOW
        if "SPACE" in gesture_status:
            status_color = Colors.NEON_GREEN
        elif "BACK" in gesture_status:
            status_color = Colors.NEON_PINK

        put_text_with_bg(canvas, gesture_status,
                         (960 // 2 - 120, 720 // 2),
                         font_scale=0.8, color=status_color, thickness=2,
                         bg_color=Colors.BG_DARK, bg_alpha=0.7, padding=10)

    # ── Last action indicator (Bottom-left of camera feed) ───────────────
    if last_action and time.time() - last_action[1] < 2.0:
        action_text = last_action[0]
        put_text_with_bg(canvas, action_text,
                         (20, 720 - 40),
                         font_scale=0.6, color=Colors.NEON_GREEN, thickness=1)

    # ── Hand detection indicator dot (Bottom-right of camera feed) ───────
    if num_hands >= 1:
        indicator_color = Colors.NEON_GREEN
    else:
        indicator_color = (60, 60, 200)
    cv2.circle(canvas, (930, 700), 8, indicator_color, -1)
    put_text_with_bg(canvas, f"{num_hands}H", (900, 705),
                     font_scale=0.4, color=indicator_color, thickness=1)

    # ── Right Column: Notepad (320x720) ──────────────────────────────────
    # Draw background panel
    cv2.rectangle(canvas, (960, 0), (1280, 720), (22, 20, 18), -1)
    
    # Left border of the notepad (separating line)
    cv2.line(canvas, (960, 0), (960, 720), (45, 40, 35), 2)

    # Title
    put_text_with_bg(canvas, "TYPED TEXT", (980, 42),
                     font_scale=0.7, color=Colors.NEON_CYAN, thickness=2)
    
    # Title Underline
    cv2.line(canvas, (980, 52), (1260, 52), Colors.NEON_GREEN, 2, cv2.LINE_AA)

    # Wrap the typed sentence (with cursor)
    sentence_with_cursor = sentence_builder.get_display_text()
    wrapped_lines = wrap_text(sentence_with_cursor, max_chars_per_line=23)

    # Display up to 13 lines, scroll from the bottom if longer
    max_visible_lines = 13
    if len(wrapped_lines) > max_visible_lines:
        visible_lines = wrapped_lines[-max_visible_lines:]
    else:
        visible_lines = wrapped_lines

    start_y = 90
    line_height = 32
    for idx, line_str in enumerate(visible_lines):
        cv2.putText(canvas, line_str, (980, start_y + idx * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.TEXT_WHITE, 1, cv2.LINE_AA)

    # ── Controls / Help section at bottom of notepad ──────────────────────
    help_start_y = 560
    # Top divider of help section
    cv2.line(canvas, (980, help_start_y - 15), (1260, help_start_y - 15), (45, 40, 35), 1)

    put_text_with_bg(canvas, "CONTROLS", (980, help_start_y + 10),
                     font_scale=0.45, color=Colors.NEON_ORANGE, thickness=1)

    controls_lines = [
        "SPACE : Add Space",
        "BKSP  : Delete Last",
        "C     : Clear Notepad",
        "ESC   : Quit Application"
    ]

    for idx, ctrl_str in enumerate(controls_lines):
        cv2.putText(canvas, ctrl_str, (980, help_start_y + 35 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, Colors.TEXT_GRAY, 1, cv2.LINE_AA)


def main():
    """Main detection loop."""
    print("\n" + "=" * 60)
    print("  ASL Sign Language Detector")
    print("=" * 60 + "\n")

    clf, encoder = load_model()
    if clf is None:
        return

    # Initialize Hand Detector (ASL is single-handed)
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

    # Initialize components
    smoother = PredictionSmoother(buffer_size=15, min_confidence=0.50, min_votes=8)
    sentence_builder = SentenceBuilder()

    # State variables
    prev_time = time.time()
    last_action = None
    no_hand_timer = 0
    gesture_status = ""
    gesture_status_time = 0
    last_key_time = 0  # keyboard debounce timer

    print(f"  * Detection started! Show ASL signs with ONE hand.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Crop frame horizontally to 960x720 (center-cropping from 1280)
        if w == 1280 and h == 720:
            cropped_frame = frame[:, 160:1120]
        else:
            # Fallback: Resize/crop to 960x720
            target_w, target_h = 960, 720
            scale = max(target_w / w, target_h / h)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rh, rw = resized.shape[:2]
            start_x = (rw - target_w) // 2
            start_y = (rh - target_h) // 2
            cropped_frame = resized[start_y:start_y+target_h, start_x:start_x+target_w]

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        delta_time = curr_time - prev_time
        prev_time = curr_time

        # Clear gesture_status after 1.5 seconds
        if gesture_status and curr_time - gesture_status_time > 1.5:
            gesture_status = ""

        # Process cropped frame with hand detector
        multi_hand_landmarks, multi_handedness = detector.process(cropped_frame)

        hands_detected = False
        num_hands = 0
        prediction = None
        confidence = 0.0

        if multi_hand_landmarks:
            hands_detected = True
            num_hands = len(multi_hand_landmarks)
            no_hand_timer = 0

            # Draw hand skeleton on cropped frame (ASL uses primary hand)
            draw_hand_skeleton(cropped_frame, multi_hand_landmarks[0], 960, 720)

            # Extract ASL features
            features = extract_asl_features(
                multi_hand_landmarks, multi_handedness
            ).reshape(1, -1)

            # Predict
            try:
                pred_proba = clf.predict_proba(features)[0]
                pred_idx = np.argmax(pred_proba)
                confidence = pred_proba[pred_idx]
                prediction = encoder.inverse_transform([pred_idx])[0]

                # Smooth prediction
                registered = smoother.update(prediction, confidence)

                if registered:
                    if registered == "_":
                        sentence_builder.add_space()
                        last_action = ("+ SPACE", curr_time)
                        print(f"  Detected: SPACE ({confidence * 100:.0f}%) -> \"{sentence_builder.get_sentence()}\"")
                    else:
                        sentence_builder.add_letter(registered)
                        last_action = (f"+ {registered}", curr_time)
                        print(f"  Detected: {registered} ({confidence * 100:.0f}%) -> \"{sentence_builder.get_sentence()}\"")
                    smoother.reset()
            except Exception as e:
                print(f"  Predict error: {e}")

        else:
            no_hand_timer += delta_time
            # Update prediction smoother with None to trigger gap reset
            smoother.update(None, 0.0)

        # Create canvas of size 1280x720 and place cropped_frame on the left
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:, :960] = cropped_frame

        # Draw UI overlay
        draw_detection_ui(
            canvas, prediction, confidence, sentence_builder,
            smoother, hands_detected, fps, gesture_status,
            no_hand_timer, last_action, num_hands
        )

        # Display the canvas
        cv2.imshow("Sign Language Detector", canvas)

        # ── Handle key presses ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break

        elif key == ord('c') or key == ord('C'):
            if curr_time - last_key_time > KEY_DEBOUNCE:
                sentence_builder.clear()
                smoother.reset(hard_reset=True)
                last_action = ("Cleared", curr_time)
                print("  [CLEARED]")
                last_key_time = curr_time

        elif key == 32:  # Spacebar
            if curr_time - last_key_time > KEY_DEBOUNCE:
                sentence_builder.add_space()
                smoother.reset(hard_reset=True)
                last_action = ("+ SPACE", curr_time)
                print(f"  [SPACE] -> \"{sentence_builder.get_sentence()}\"")
                last_key_time = curr_time

        elif key == 8:  # Backspace
            if curr_time - last_key_time > KEY_DEBOUNCE:
                sentence_builder.backspace()
                smoother.reset(hard_reset=True)
                last_action = ("<- BACKSPACE", curr_time)
                print(f"  [BACKSPACE] -> \"{sentence_builder.get_sentence()}\"")
                last_key_time = curr_time

    # Cleanup
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    final_sentence = sentence_builder.get_sentence()
    if final_sentence:
        print(f"\n  Final sentence: \"{final_sentence}\"")

    print("\n  * Detection ended. Goodbye!\n")


if __name__ == "__main__":
    main()
