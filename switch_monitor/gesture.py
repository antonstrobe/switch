from __future__ import annotations

from dataclasses import dataclass

try:
    import cv2
    import mediapipe as mp
except ImportError:  # pragma: no cover
    cv2 = None
    mp = None


@dataclass
class GestureDetection:
    digit: int | None
    confidence: str
    reason: str
    source: str = "mediapipe"


class FingerCounter:
    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe is not installed")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr) -> GestureDetection:
        if cv2 is None or mp is None:
            return GestureDetection(None, "low", "Локальный детектор руки недоступен.")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return GestureDetection(None, "low", "Рука не найдена в кадре.")

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness_label = "Right"
        handedness_score = 0.5
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness_label = classification.label
            handedness_score = float(classification.score)

        landmarks = hand_landmarks.landmark
        digit = self._count_fingers(landmarks, handedness_label)
        confidence = "high" if handedness_score >= 0.8 else "medium"
        reason = f"Локально распознано {digit} пальцев."
        return GestureDetection(digit, confidence, reason)

    def _count_fingers(self, landmarks, handedness_label: str) -> int:
        fingers = 0

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        if handedness_label.lower() == "right":
            thumb_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_open = thumb_tip.x > thumb_ip.x
        if thumb_open:
            fingers += 1

        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_id, pip_id in finger_pairs:
            if landmarks[tip_id].y < landmarks[pip_id].y:
                fingers += 1

        return fingers
