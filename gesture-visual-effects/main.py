import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pygame

# -------------------------------
# Initialize sound system
# -------------------------------
pygame.mixer.init()
smile_sound = pygame.mixer.Sound("smile.wav")
cry_sound   = pygame.mixer.Sound("cry.wav")
heart_sound = pygame.mixer.Sound("heart.wav")

# -------------------------------
# Safe image loader
# -------------------------------
def load_overlay(path, size=220):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.resize(img, (size, size))

smile_img   = load_overlay("smile.png")
cry_img     = load_overlay("cry.png")
thumbs_img  = load_overlay("thumpsup.jpeg", size=250)

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# -------------------------------
# Helper functions
# -------------------------------
def overlay_image(bg, overlay, cx, cy):
    h, w = overlay.shape[:2]
    x = max(0, cx - w // 2)
    y = max(0, cy - h // 2)

    x2 = min(bg.shape[1], x + w)
    y2 = min(bg.shape[0], y + h)

    overlay = overlay[:y2-y, :x2-x]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y2, x:x2, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * bg[y:y2, x:x2, c]
            )
    else:
        bg[y:y2, x:x2] = overlay


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# -------------------------------
# Thumbs-Up detection
# -------------------------------
def is_thumbs_up(hand):
    lm = hand.landmark

    if lm[4].y > lm[3].y:
        return False

    for f in [8, 12, 16, 20]:
        if lm[f].y < lm[f - 2].y:
            return False

    return True


# -------------------------------
# Heart Gesture detection
# -------------------------------
def is_heart_gesture(hand):
    lm = hand.landmark

    thumb_tip = np.array([lm[4].x, lm[4].y])
    index_tip = np.array([lm[8].x, lm[8].y])

    distance = np.linalg.norm(thumb_tip - index_tip)

    if distance < 0.05:
        return True

    return False


# -------------------------------
# Expression stabilizer
# -------------------------------
expr_buffer = deque(maxlen=5)

def stable_expression(expr):
    expr_buffer.append(expr)
    if expr_buffer.count(expr) >= 3:
        return expr
    return "neutral"


# -------------------------------
# Main loop
# -------------------------------
last_expression = "neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- HAND --------
    hand_result = hands.process(rgb)
    thumbs_detected = False
    heart_detected = False

    if hand_result.multi_hand_landmarks:
        hand_landmarks = hand_result.multi_hand_landmarks[0]

        if is_thumbs_up(hand_landmarks):
            thumbs_detected = True

        if is_heart_gesture(hand_landmarks):
            heart_detected = True

    # -------- FACE --------
    face_result = face_mesh.process(rgb)
    expression = "neutral"
    face_center = None

    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0]
        lm = lambda p: (int(p.x * w), int(p.y * h))

        face_width = dist(lm(face.landmark[234]), lm(face.landmark[454]))
        if face_width > 1:
            face_center = lm(face.landmark[1])

            mouth_w = dist(lm(face.landmark[61]), lm(face.landmark[291])) / face_width
            mouth_o = dist(lm(face.landmark[13]), lm(face.landmark[14])) / face_width
            brow_eye = (
                dist(lm(face.landmark[65]), lm(face.landmark[159])) +
                dist(lm(face.landmark[295]), lm(face.landmark[386]))
            ) / (2 * face_width)

            if mouth_w > 0.38 and mouth_o > 0.06:
                expression = "smile"
            elif mouth_o < 0.035 and brow_eye > 0.045:
                expression = "neutral"
            else:
                expression = "cry"

            expression = stable_expression(expression)

    # -------- SOUND --------
    if heart_detected:
        heart_sound.play()

    elif not thumbs_detected and expression != last_expression:
        if expression == "smile":
            smile_sound.play()
        elif expression == "cry":
            cry_sound.play()

        last_expression = expression

    # -------- OVERLAY --------
    if face_center:
        cx, cy = face_center

        if thumbs_detected:
            overlay_image(frame, thumbs_img, cx, cy)
            cv2.putText(frame, "THUMBS UP 👍", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        elif expression == "smile":
            overlay_image(frame, smile_img, cx, cy)

        elif expression == "cry":
            overlay_image(frame, cry_img, cx, cy)

    if heart_detected:
        cv2.putText(frame, "HEART ❤️", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("Expression + Gesture Interaction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()