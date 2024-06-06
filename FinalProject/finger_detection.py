import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
def calculate_palm_width(landmarks, height):
    # Using the wrist and the middle finger MCP (metacarpophalangeal joint) to estimate palm width
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Calculate the distance between these points as the palm width
    palm_width = np.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2) * height  # height is used to scale the width appropriately
    return palm_width

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
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
    palm_width = calculate_palm_width(hand_landmarks, height)
    dist_thumb_index_mcp = np.sqrt((thumb_tip.x - hand_landmarks.landmark[finger_joints[0][0]].x)**2 + (thumb_tip.y - hand_landmarks.landmark[finger_joints[0][0]].y)**2) / palm_width

    if dist_thumb_index_mcp > 0.0005:
        fingers_up.append(True)
    else:
        fingers_up.append(False)

    for mcp_joint, pip_joint, dip_joint in finger_joints:
        mcp = [hand_landmarks.landmark[mcp_joint].x * width, hand_landmarks.landmark[mcp_joint].y * height]
        pip = [hand_landmarks.landmark[pip_joint].x * width, hand_landmarks.landmark[pip_joint].y * height]
        dip = [hand_landmarks.landmark[dip_joint].x * width, hand_landmarks.landmark[dip_joint].y * height]

        angle = calculate_angle(mcp, pip, dip)

        if angle > angle_threshold:
            fingers_up.append(True)
        else:
            fingers_up.append(False)

    return fingers_up

if __name__ == "__main__":
  # For static images:
  IMAGE_FILES = []
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      print('Handedness:', results.multi_handedness)
      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      cv2.imwrite(
          '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
      # Draw hand world landmarks.
      if not results.multi_hand_world_landmarks:
        continue
      for hand_world_landmarks in results.multi_hand_world_landmarks:
        mp_drawing.plot_landmarks(
          hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

  # For webcam input:
  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      # print(image.shape)
      if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                  finger_state = check_finger_up(hand_landmarks, image.shape[0], image.shape[1])
                  print(finger_state)
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      # if cv2.waitKey(5) & 0xFF == 27:
      #   break

      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        break
  cap.release()