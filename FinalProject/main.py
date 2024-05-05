import cv2
import mediapipe as mp
import numpy as np
import time
mp_hands = mp.solutions.hands

def calculate_line_thickness(base_thickness, z_coordinate, scale_factor=50):
    """ Calculate line thickness based on depth information.
        base_thickness: base thickness of the line when z_coordinate is 0.
        z_coordinate: depth value of the point (negative is closer).
        scale_factor: how much the depth affects the line thickness.
    """
    return max(int(base_thickness - (z_coordinate * scale_factor)), 1)  # Ensure at least 1 pixel thickness

def is_pinching(thumb, index):
    """ Determine if two fingers are close enough to consider pinching. """
    distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
    return distance < 0.05  # Adjust threshold based on your testing

def find_line_close_to_pinch(lines, pinch_pos, threshold=10):
    """ Find a line close to the pinch position. """
    for line in lines:
        if np.linalg.norm(np.cross(line[1]-line[0], line[0]-pinch_pos)) / np.linalg.norm(line[1]-line[0]) < threshold:
            return line
    return None

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_palm_width(landmarks, height):
    # Using the wrist and the middle finger MCP (metacarpophalangeal joint) to estimate palm width
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Calculate the distance between these points as the palm width
    palm_width = np.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2) * height  # height is used to scale the width appropriately
    return palm_width

def check_open_palm(landmarks, height):
    palm_width = calculate_palm_width(landmarks, height)

    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distances between the tips of the thumb, index, and pinky fingers
    dist_thumb_index = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) / palm_width
    dist_thumb_pinky = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2) / palm_width

    # Check if both distances are above a threshold that indicates an open hand
    open_hand = dist_thumb_index > 0.0013 and dist_thumb_pinky > 0.0015
    print(f"Distances: Thumb-Index = {dist_thumb_index/ palm_width}, \nThumb-Pinky = {dist_thumb_pinky/ palm_width}, \npalm_width = {palm_width}")
    return open_hand

def check_finger_up(hand_landmarks, width, height):
    fingers_up = []
    # Finger angle threshold to determine if a finger is up
    angle_threshold = 160
    # List of finger landmark triplets (MCP, PIP, DIP) for each finger
    finger_joints = [
        # (mp.solutions.hands.HandLandmark.THUMB_MCP, mp.solutions.hands.HandLandmark.THUMB_IP, mp.solutions.hands.HandLandmark.THUMB_TIP),
        (mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP),
        (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP),
        (mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP, mp.solutions.hands.HandLandmark.RING_FINGER_DIP),
        (mp.solutions.hands.HandLandmark.PINKY_MCP, mp.solutions.hands.HandLandmark.PINKY_PIP, mp.solutions.hands.HandLandmark.PINKY_DIP)
    ]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    # pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    palm_width = calculate_palm_width(hand_landmarks, height)
    # # Calculate distances between the tips of the thumb, index, and pinky fingers
    dist_thumb_index_mcp = np.sqrt((thumb_tip.x - hand_landmarks.landmark[finger_joints[0][0]].x)**2 + (thumb_tip.y - hand_landmarks.landmark[finger_joints[0][0]].y)**2) / palm_width
    # # print('thumb pinky', dist_thumb_pinky)
    # thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    # wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

    # palm_width = calculate_palm_width(hand_landmarks, height)
    # # Calculate distance between the thumb tip and the wrist
    # dist_thumb_wrist = np.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2) / palm_width
    print('Thumb to Wrist Distance:', dist_thumb_index_mcp)

    if dist_thumb_index_mcp > 0.0005:
        fingers_up.append(True)
    else:
        fingers_up.append(False)

    for mcp_joint, pip_joint, dip_joint in finger_joints:
        mcp = [hand_landmarks.landmark[mcp_joint].x * width, hand_landmarks.landmark[mcp_joint].y * height]
        pip = [hand_landmarks.landmark[pip_joint].x * width, hand_landmarks.landmark[pip_joint].y * height]
        dip = [hand_landmarks.landmark[dip_joint].x * width, hand_landmarks.landmark[dip_joint].y * height]

        # Calculate the angle of the finger
        angle = calculate_angle(mcp, pip, dip)

        # Check if the finger is up
        if angle > angle_threshold:
            fingers_up.append(True)
        else:
            fingers_up.append(False)

    return fingers_up

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, test_img = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        cap.release()
        exit()
    
    h, w = test_img.shape[:2]
    draw = np.zeros((h, w, 4), dtype='uint8')  # Ensure this matches the camera frame's height and width
    last_point = None
    drawing = False  # Control drawing state
    palm_open_duration = 2
    palm_open_time = None
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        img = cv2.flip(img, 1)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert frame to BGRA
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Process image in RGB

        # Process the image and detect hands
        results = hands.process(img2)

        # Check if any hand landmarks were detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if check_open_palm(hand_landmarks, h):
                    if palm_open_time is None:
                        palm_open_time = time.time()
                    elif time.time() - palm_open_time >= palm_open_duration:
                        draw = np.zeros((h, w, 4), dtype='uint8')  # Clear drawing
                else:
                    print("here")
                    palm_open_time = None

                    finger_state = check_finger_up(hand_landmarks, w, h)
                    print(finger_state)
                    # tip_depth = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z  # Depth of the fingertip
                    # # Calculate dynamic thickness based on the depth
                    # thickness = calculate_line_thickness(5, tip_depth)

                    # if finger_angle > 160:  # Threshold for finger to be considered as "straight"
                    #     current_point = (int(tip[0]), int(tip[1]))
                    #     drawing = True
                    # else:
                    #     drawing = False

                    # if palm_open_time is None:  # Only draw if palm is not open
                    #     if drawing and last_point is not None:
                    #         cv2.line(draw, last_point, current_point, (0, 255, 0, 255), thickness)
                    # # if drawing and last_point is not None:
                    # #     cv2.line(draw, last_point, current_point, (0, 255, 0, 255), 2)  # Green color

                    #     last_point = current_point if drawing else None
        else:
            # If no hands are detected, reset last_point
            palm_open_time = None
            last_point = None
            drawing = False


        for j in range(w):
            img[:,j,0] = img[:,j,0]*(1-draw[:,j,3]/255) + draw[:,j,0]*(draw[:,j,3]/255)
            img[:,j,1] = img[:,j,1]*(1-draw[:,j,3]/255) + draw[:,j,1]*(draw[:,j,3]/255)
            img[:,j,2] = img[:,j,2]*(1-draw[:,j,3]/255) + draw[:,j,2]*(draw[:,j,3]/255)
        # Overlay the drawing canvas on the video frame
        # img = cv2.addWeighted(img, 1, draw, 0.8, 0)

        cv2.imshow('Finger Drawing', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
