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

visual_frame_size = (800, 800)

cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=np.float32)

cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

angle_x, angle_y = 0, 0
scale_factor = 100  

# **Tracking for swipe gestures**
hand_positions = deque(maxlen=8)  # stores 8 previous hand positions (by frame)
gesture_threshold = 120  # minimum movement, in pixels, for a swipe

# **Tracking for pinch detection**
pinch_distances = deque(maxlen=8)  # store last 8 pinch distances(by frame )
pinch_threshold = 0.8  # minimum variation in normalized distance to detect pinch

def project_point(point, screen_size):
    fov = 700
    z = point[2] + 8
    z = max(z, 0.1)
    x = int((point[0] / z) * fov + screen_size[0] / 2)
    y = int((point[1] / z) * fov + screen_size[1] / 2)
    return (x, y)

def detect_swipe(hand_positions, threshold):
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

def detect_pinch(pinch_distances, threshold):
    if len(pinch_distances) > 1:
        start_distance = pinch_distances[0]
        end_distance = pinch_distances[-1]
        variation = end_distance - start_distance

        if abs(variation) > threshold:
            return "Pinch Out" if variation > 0 else "Pinch In"
    
    return None  

def translate_cube(vertices, translation):
    """
    :param vertices: np.array of shape (8, 3), representing the cube's vertices.
    :param translation: tuple (dx, dy, dz) representing the translation along x, y, and z axes.
    :return: np.array of translated vertices.
    """
    translation_matrix = np.array(translation, dtype=np.float32)
    translated_vertices = vertices + translation_matrix
    return translated_vertices

translation_x, translation_y, translation_z = 0, 0, 5
scaling_factor = 1.0  # scale modifier

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    normalized_lengths = [0, 0]
    hand_center = None  
    pinch_distance = None  # current frame pinch distance

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

                # store the normalized pinch distance
                pinch_distance = normalized_length
                pinch_distances.append(pinch_distance)

            # store hand center for swipe detection
            hand_center = (int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h))
            hand_positions.append(hand_center)

    # **Detect gestures**
    swipe_detected = detect_swipe(hand_positions, gesture_threshold)
    pinch_detected = detect_pinch(pinch_distances, pinch_threshold)

    visual_frame = np.zeros((visual_frame_size[0], visual_frame_size[1], 3), dtype=np.uint8)
    
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

    if swipe_detected:
        if swipe_detected == "Swipe Left":
            translation_x -= 0.3 
        elif swipe_detected == "Swipe Right":
            translation_x += 0.3  
        elif swipe_detected == "Swipe Up":
            translation_y -= 0.3  
        elif swipe_detected == "Swipe Down":
            translation_y += 0.3  

    if pinch_detected:
        if pinch_detected == "Pinch In":
            scaling_factor *= 0.95 
        elif pinch_detected == "Pinch Out":
            scaling_factor *= 1.05  

    transformed_vertices = [vertex * scaling_factor for vertex in transformed_vertices]
    transformed_vertices = [vertex + [translation_x, translation_y, translation_z] for vertex in transformed_vertices]
    
    projected_vertices = [project_point(vertex * scale_factor, visual_frame_size) for vertex in transformed_vertices]

    # draw cube edges
    for edge in cube_edges:
        pt1, pt2 = projected_vertices[edge[0]], projected_vertices[edge[1]]
        cv2.line(visual_frame, pt1, pt2, (255, 255, 255), 2)

    if swipe_detected:
        cv2.putText(frame, f"Gesture: {swipe_detected}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    if pinch_detected:
        cv2.putText(frame, f"Gesture: {pinch_detected}", (50, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("jarvis, track my fingers", frame)
    cv2.imshow("cube", visual_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()