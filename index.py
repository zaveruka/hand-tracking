import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cv2.namedWindow("jarvis, track my fingers", cv2.WINDOW_NORMAL)
cv2.namedWindow("cube", cv2.WINDOW_NORMAL)

cube_vertices = np.array([
    [-1, -1, -1],  # Bottom-left-back
    [1, -1, -1],   # Bottom-right-back
    [1, 1, -1],    # Top-right-back
    [-1, 1, -1],   # Top-left-back
    [-1, -1, 1],   # Bottom-left-front
    [1, -1, 1],    # Bottom-right-front
    [1, 1, 1],     # Top-right-front
    [-1, 1, 1]     # Top-left-front
], dtype=np.float32)

cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), 
    (4, 5), (5, 6), (6, 7), (7, 4), 
    (0, 4), (1, 5), (2, 6), (3, 7)  
]

angle_x, angle_y, angle_z = 0, 0, 0
scale_factor = 100  

def project_point(point, screen_size):
    fov = 400
    z = point[2] + 5 
    if z == 0:
        z = 0.1
    x = int((point[0] / z) * fov + screen_size[0] / 2)
    y = int((point[1] / z) * fov + screen_size[1] / 2)
    return (x, y)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    normalized_lengths = [0, 0]
    
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            h, w, c = frame.shape
            points = []
            for idx in [4, 8]: 
                x = int(hand_landmarks.landmark[idx].x * w)
                y = int(hand_landmarks.landmark[idx].y * h)
                points.append((x, y))
            if len(points) == 2:
                length = int(((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) ** 0.5)
                min_length = 25  
                max_length = 140 
                normalized_length = (length - min_length) / (max_length - min_length)
                normalized_length = max(0, min(1, normalized_length))  
                normalized_lengths[hand_index % 2] = normalized_length
    
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
        pt1 = projected_vertices[edge[0]]
        pt2 = projected_vertices[edge[1]]
        cv2.line(visual_frame, pt1, pt2, (255, 255, 255), 2)
    
    cv2.imshow("jarvis, track my fingers", frame)
    cv2.imshow("cube", visual_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
