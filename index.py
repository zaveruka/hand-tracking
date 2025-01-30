import cv2
import mediapipe as mp
import numpy as np
from collections import deque


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cv2.namedWindow("jarvis, track my fingers", cv2.WINDOW_NORMAL)
cv2.namedWindow("cube", cv2.WINDOW_NORMAL)

cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=np.float32)

cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

angle_x, angle_y = 0, 0
scale_factor = 100  

hand_positions = deque(maxlen=8)  # stores 8 previous hand positions(by frame)
gesture_threshold = 120  # minimum movement, in pixels, for a gesture

def project_point(point, screen_size):
    """Projects 3D points onto a 2D screen."""
    fov = 400
    z = point[2] + 5
    z = max(z, 0.1)
    x = int((point[0] / z) * fov + screen_size[0] / 2)
    y = int((point[1] / z) * fov + screen_size[1] / 2)
    return (x, y)

def detect_gesture(hand_positions, threshold):
    if len(hand_positions) > 1:
        start_x, start_y = hand_positions[0]
        end_x, end_y = hand_positions[-1]
        displacement_x = end_x - start_x
        displacement_y = end_y - start_y

        if abs(displacement_x) > threshold:
            return "Swipe Left" if displacement_x < 0 else "Swipe Right"
        elif abs(displacement_y) > threshold:
            return "Swipe Up" if displacement_y < 0 else "Swipe Down"
    
    return None  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    normalized_lengths = [0, 0]
    hand_center = None  

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            h, w, _ = frame.shape
            points = []
            for idx in [4, 8]: 
                x = int(hand_landmarks.landmark[idx].x * w)
                y = int(hand_landmarks.landmark[idx].y * h)
                points.append((x, y))
            if len(points) == 2:
                length = int(((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) ** 0.5)
                min_length, max_length = 25, 140
                normalized_length = (length - min_length) / (max_length - min_length)
                normalized_length = max(0, min(1, normalized_length))
                normalized_lengths[hand_index % 2] = normalized_length

            hand_center = (int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h))
            hand_positions.append(hand_center)

    gesture_detected = detect_gesture(hand_positions, gesture_threshold)

    visual_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    angle_x += 5 * normalized_lengths[0]  
    angle_y += 5 * normalized_lengths[1] 
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(angle_x)), -np.sin(np.radians(angle_x))],
        [0, np.sin(np.radians(angle_x)), np.cos(np.radians(angle_x))]
    ])
    rotation_y = np.array([
        [np.cos(np.radians(angle_y)), 0, np.sin(np.radians(angle_y))],
        [0, 1, 0],
        [-np.sin(np.radians(angle_y)), 0, np.cos(np.radians(angle_y))]
    ])
    
    rotation_matrix = rotation_x @ rotation_y
    transformed_vertices = [rotation_matrix @ vertex for vertex in cube_vertices]
    transformed_vertices = [vertex + [0, 0, 5] for vertex in transformed_vertices]
    projected_vertices = [project_point(vertex * scale_factor, visual_frame.shape[:2]) for vertex in transformed_vertices]

    for edge in cube_edges:
        pt1, pt2 = projected_vertices[edge[0]], projected_vertices[edge[1]]
        cv2.line(visual_frame, pt1, pt2, (255, 255, 255), 2)

    if gesture_detected:
        cv2.putText(frame, f"Gesture: {gesture_detected}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("jarvis, track my fingers", frame)
    cv2.imshow("cube", visual_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
