"""
utils.py — Shared utilities for ASL (American Sign Language) Detection
Provides landmark normalization, UI drawing helpers, prediction smoothing,
and sentence building logic.

ASL uses a SINGLE-HANDED alphabet.
"""

import cv2
import numpy as np
from collections import deque, Counter
import os
import urllib.request
import time

# Try importing tasks/solutions
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    TASKS_API_AVAILABLE = True
except ImportError:
    TASKS_API_AVAILABLE = False

try:
    import mediapipe as mp
    SOLUTIONS_API_AVAILABLE = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')
except ImportError:
    SOLUTIONS_API_AVAILABLE = False


# ─── MediaPipe Tasks API Compatibility Wrappers ─────────────────────────────
class ClassificationWrapper:
    def __init__(self, label):
        self.label = label

class HandednessWrapper:
    def __init__(self, label):
        self.classification = [ClassificationWrapper(label)]

class HandLandmarksWrapper:
    def __init__(self, landmarks):
        self.landmark = landmarks



# ─── Color Palette (BGR for OpenCV) ─────────────────────────────────────────
class Colors:
    # Primary accents
    NEON_BLUE = (255, 180, 50)
    NEON_GREEN = (100, 255, 130)
    NEON_PURPLE = (200, 100, 255)
    NEON_PINK = (180, 80, 255)
    NEON_CYAN = (255, 230, 50)
    NEON_ORANGE = (50, 160, 255)
    NEON_YELLOW = (50, 240, 255)

    # UI colors
    BG_DARK = (30, 25, 20)
    BG_PANEL = (40, 35, 30)
    TEXT_WHITE = (255, 255, 255)
    TEXT_GRAY = (180, 180, 180)
    TEXT_DIM = (120, 120, 120)
    CONFIDENCE_HIGH = (100, 255, 130)
    CONFIDENCE_MED = (50, 230, 255)
    CONFIDENCE_LOW = (80, 80, 255)

    # Hand skeleton
    LANDMARK_DOT = (255, 200, 80)
    LANDMARK_LINE = (200, 150, 50)
    WRIST_DOT = (100, 255, 200)

    # Finger colors for skeleton
    FINGER_COLORS = [
        (255, 100, 100),   # Thumb - blue
        (100, 255, 100),   # Index - green
        (100, 200, 255),   # Middle - orange
        (255, 100, 255),   # Ring - pink
        (100, 255, 255),   # Pinky - yellow
    ]


# ─── MediaPipe Hand Connections (finger groups) ─────────────────────────────
FINGER_CONNECTIONS = [
    # Thumb
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    # Index
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    # Middle
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    # Ring
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    # Pinky
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]

# Palm connections
PALM_CONNECTIONS = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]


def normalize_single_hand(hand_landmarks, mirror_x=False):
    """
    Normalize 21 hand landmarks relative to the wrist (landmark 0).
    Returns a flat array of 42 values (21 x/y pairs), all relative to wrist
    and scaled by the max distance from wrist for scale invariance.
    If mirror_x is True, mirrors the x coordinates first.

    Parameters:
        hand_landmarks: MediaPipe hand landmarks object
        mirror_x: If True, flips the X coordinates

    Returns:
        np.array of shape (42,) with normalized x, y coordinates
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y])
    landmarks = np.array(landmarks)

    if mirror_x:
        landmarks[:, 0] = -landmarks[:, 0]

    # Make relative to wrist
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    # Scale by max distance from wrist for size invariance
    max_dist = np.max(np.sqrt(np.sum(landmarks ** 2, axis=1)))
    if max_dist > 0:
        landmarks = landmarks / max_dist

    return landmarks.flatten()


def extract_asl_features(multi_hand_landmarks, multi_handedness):
    """
    Extract features for single-hand ASL mode.
    If multiple hands are detected, uses the first hand.
    If no hands are detected, returns 42 zeros.
    """
    if not multi_hand_landmarks:
        return np.zeros(42)
    return normalize_single_hand(multi_hand_landmarks[0], mirror_x=False)



def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.7):
    """
    Draw a rounded rectangle with transparency.

    Parameters:
        img: OpenCV image (modified in-place)
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color: BGR color tuple
        radius: Corner radius
        thickness: -1 for filled
        alpha: Transparency (0-1)
    """
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw the main rectangle
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw four corner circles
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_gradient_bar(img, pt1, pt2, color_start, color_end, progress=1.0):
    """
    Draw a gradient progress bar.

    Parameters:
        img: OpenCV image
        pt1: Top-left corner
        pt2: Bottom-right corner
        color_start: Starting BGR color
        color_end: Ending BGR color
        progress: Fill amount 0-1
    """
    x1, y1 = pt1
    x2, y2 = pt2
    bar_width = int((x2 - x1) * progress)

    if bar_width <= 0:
        return

    for x in range(bar_width):
        t = x / max(bar_width - 1, 1)
        color = tuple(int(color_start[i] + t * (color_end[i] - color_start[i])) for i in range(3))
        cv2.line(img, (x1 + x, y1), (x1 + x, y2), color, 1)


def draw_hand_skeleton(img, hand_landmarks, w, h):
    """
    Draw a stylized hand skeleton with colored fingers and glowing nodes.

    Parameters:
        img: OpenCV image
        hand_landmarks: MediaPipe hand landmarks
        w: Frame width
        h: Frame height
    """
    # Get pixel coordinates
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))

    # Draw palm connections
    for (i, j) in PALM_CONNECTIONS:
        cv2.line(img, pts[i], pts[j], Colors.LANDMARK_LINE, 2, cv2.LINE_AA)

    # Draw finger connections with individual colors
    for finger_idx, connections in enumerate(FINGER_CONNECTIONS):
        color = Colors.FINGER_COLORS[finger_idx]
        for (i, j) in connections:
            cv2.line(img, pts[i], pts[j], color, 3, cv2.LINE_AA)

    # Draw landmark dots
    for idx, pt in enumerate(pts):
        if idx == 0:
            # Wrist — larger, different color
            cv2.circle(img, pt, 8, Colors.WRIST_DOT, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 10, Colors.WRIST_DOT, 1, cv2.LINE_AA)
        elif idx in [4, 8, 12, 16, 20]:
            # Fingertips — glowing dots
            finger_idx = [4, 8, 12, 16, 20].index(idx)
            color = Colors.FINGER_COLORS[finger_idx]
            cv2.circle(img, pt, 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 10, (*[min(255, c + 50) for c in color],), 1, cv2.LINE_AA)
        else:
            # Joint dots
            cv2.circle(img, pt, 4, Colors.LANDMARK_DOT, -1, cv2.LINE_AA)


def draw_ui_panel(img, x, y, w, h, alpha=0.75):
    """
    Draw a semi-transparent dark panel for UI elements.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), Colors.BG_PANEL, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_with_bg(img, text, pos, font_scale=0.7, color=Colors.TEXT_WHITE,
                     thickness=2, bg_color=None, bg_alpha=0.6, padding=5):
    """
    Draw text with an optional background rectangle.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos
    if bg_color is not None:
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (x - padding, y - th - padding),
                      (x + tw + padding, y + baseline + padding),
                      bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


class PredictionSmoother:
    """
    Smooths predictions using a rolling buffer with majority voting and gap-resetting.
    Only outputs a prediction when votes exceed a threshold in the rolling window.
    Resets the last registered letter on a gap to allow double-letter entry.
    """

    def __init__(self, buffer_size=15, min_confidence=0.50, min_votes=8):
        """
        Parameters:
            buffer_size: Number of recent predictions to keep
            min_confidence: Minimum model confidence to consider a prediction
            min_votes: Minimum votes required inside the window to register
        """
        self.buffer = deque(maxlen=buffer_size)
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        self.last_registered = None
        self.gap_counter = 0
        self.waiting_for_gap = False
        self.lock_until_hand_removed = False

    def update(self, prediction, confidence):
        """
        Update with a new prediction.

        Parameters:
            prediction: Predicted label (letter) or None
            confidence: Model confidence (0-1)

        Returns:
            str or None: The stable prediction if conditions are met, else None
        """
        if confidence < self.min_confidence or prediction is None:
            self.buffer.append(None)
            # Count low confidence or missing hand frames (gaps)
            self.gap_counter += 1
            if self.gap_counter >= 8:
                # Reset last registered and unlock to allow new letter typing
                self.lock_until_hand_removed = False
                self.last_registered = None
                self.waiting_for_gap = False
            return None

        # Reset gap counter on valid high-confidence prediction
        self.gap_counter = 0
        self.buffer.append(prediction)

        # If typing is locked until hand is removed, ignore all new predictions
        if self.lock_until_hand_removed:
            return None

        if self.waiting_for_gap:
            return None

        # Count occurrences in buffer
        valid_preds = [p for p in self.buffer if p is not None]
        if not valid_preds:
            return None

        counter = Counter(valid_preds)
        most_common, count = counter.most_common(1)[0]

        # Register if held with enough votes in window
        if count >= self.min_votes:
            if self.last_registered != most_common:
                self.last_registered = most_common
                self.lock_until_hand_removed = True  # Lock further typing until hand is removed
                return most_common

        return None

    def get_current(self):
        """Get the current most common prediction (may not be registered yet)."""
        valid_preds = [p for p in self.buffer if p is not None]
        if not valid_preds:
            return None, 0
        counter = Counter(valid_preds)
        most_common, count = counter.most_common(1)[0]
        return most_common, count / len(self.buffer)

    def reset(self, hard_reset=False):
        """Reset the smoother state."""
        self.buffer.clear()
        self.last_registered = None
        self.gap_counter = 0
        if hard_reset:
            self.waiting_for_gap = False
            self.lock_until_hand_removed = False
        else:
            self.waiting_for_gap = True


class SentenceBuilder:
    """
    Manages building words and sentences from individual letter detections.
    """

    def __init__(self):
        self.current_word = ""
        self.words = []
        self.sentence = ""

    def add_letter(self, letter):
        """Add a letter to the current word."""
        self.current_word += letter
        self._update_sentence()

    def add_space(self):
        """Finish current word, start a new one."""
        if self.current_word:
            self.words.append(self.current_word)
            self.current_word = ""
            self._update_sentence()

    def backspace(self):
        """Delete the last letter from the current word."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.words:
            self.current_word = self.words.pop()
        self._update_sentence()

    def clear(self):
        """Clear everything."""
        self.current_word = ""
        self.words = []
        self.sentence = ""

    def _update_sentence(self):
        """Update the full sentence string."""
        parts = self.words.copy()
        if self.current_word:
            parts.append(self.current_word)
        self.sentence = " ".join(parts)

    def get_display_text(self):
        """Get the sentence with a cursor indicator (blinking underscore)."""
        # Blink every 500ms
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        if self.sentence:
            return self.sentence + cursor
        return cursor

    def get_sentence(self):
        """Get the plain sentence text."""
        return self.sentence


class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.use_tasks = TASKS_API_AVAILABLE
        self.detector = None
        self.legacy_hands = None

        if self.use_tasks:
            # Path to the task model in the project directory
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
            if not os.path.exists(self.model_path):
                print("  [Detector] Downloading hand_landmarker.task (approx. 5.6 MB)...")
                try:
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, self.model_path)
                    print("  [Detector] Download completed successfully.")
                except Exception as e:
                    print(f"  [Detector] ERROR downloading model: {e}")
                    print("  [Detector] Trying legacy MediaPipe Solutions instead...")
                    self.use_tasks = False

        if self.use_tasks:
            try:
                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                print("  [Detector] Initialized MediaPipe Tasks HandLandmarker successfully.")
            except Exception as e:
                print(f"  [Detector] Failed to initialize Tasks HandLandmarker: {e}")
                print("  [Detector] Falling back to legacy Solutions API...")
                self.use_tasks = False

        if not self.use_tasks:
            if SOLUTIONS_API_AVAILABLE:
                import mediapipe as mp
                self.legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("  [Detector] Initialized legacy MediaPipe Solutions Hands successfully.")
            else:
                raise RuntimeError("Neither MediaPipe Tasks nor legacy Solutions API is available!")

    def process(self, frame_bgr):
        """
        Process a BGR image frame and return landmarks and handedness.
        
        Parameters:
            frame_bgr: OpenCV BGR image
            
        Returns:
            tuple: (multi_hand_landmarks, multi_handedness) in a format compatible with legacy API.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        if self.use_tasks:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)
                
                wrapped_landmarks = []
                wrapped_handedness = []
                
                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        wrapped_landmarks.append(HandLandmarksWrapper(hand_lms))
                    for handedness in result.handedness:
                        # handedness[i] is a list containing Category objects
                        label = handedness[0].category_name
                        # Note: In MediaPipe Tasks, sometimes labels are inverted or need mapping.
                        # We use the raw category_name ('Left' / 'Right').
                        wrapped_handedness.append(HandednessWrapper(label))
                        
                return wrapped_landmarks, wrapped_handedness
            except Exception as e:
                print(f"  [Detector] Error in processing frame: {e}")
                return [], []
        else:
            results = self.legacy_hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks, results.multi_handedness
            return [], []

    def close(self):
        try:
            if self.use_tasks and self.detector:
                self.detector.close()
            elif self.legacy_hands:
                self.legacy_hands.close()
        except Exception:
            pass

