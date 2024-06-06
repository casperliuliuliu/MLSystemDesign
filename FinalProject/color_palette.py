import cv2
import numpy as np
import mediapipe as mp
import time
from finger_detection import check_finger_up

def is_pinching(thumb, index):
    """ Determine if two fingers are close enough to consider pinching. """
    distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
    return distance < 0.05  # Adjust threshold based on your testing

def draw_palette(frame):
    """ Draws a circular color palette on the frame vertically on the left side. """
    palette_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 128), (255, 165, 0), (128, 128, 0)
    ]
    palette_radius = 25
    palette_centers = []

    for i, color in enumerate(palette_colors):
        center = (35, 35 + i * 60)
        cv2.circle(frame, center, palette_radius, color, -1)
        palette_centers.append(center)

    return palette_centers, palette_colors

def check_palette_touch(palette_centers, palette_colors, point):
    """ Checks if the point touches any color in the palette and returns the color. """
    palette_radius = 25
    for center, color in zip(palette_centers, palette_colors):
        distance = np.sqrt((center[0] - point[0])**2 + (center[1] - point[1])**2)
        if distance <= palette_radius:
            return color
    return None

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    """ Draw text on the image """
    cv2.putText(image, text, position, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def main():
    # Initialize Mediapipe Hands and Webcam
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Cannot open camera")
        return

    ret, test_img = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        cap.release()
        return
    
    h, w = test_img.shape[:2]
    draw = np.zeros((h, w, 4), dtype='uint8')  # Ensure this matches the camera frame's height and width
    last_point = None
    drawing = False  # Control drawing state
    current_color = (0, 255, 0, 255)  # Initial drawing color
    palm_open_duration = 2
    palm_open_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        # Draw the color palette on the frame
        palette_centers, palette_colors = draw_palette(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                if is_pinching(thumb_tip, index_tip):
                    # Clear the drawing when pinching is detected
                    draw = np.zeros((h, w, 4), dtype='uint8')
                    draw_text(frame, 'Pinching!', (50, 50))
                # Check if the index finger touches any color in the palette
                current_point = (int(index_tip.x * w), int(index_tip.y * h))
                selected_color = check_palette_touch(palette_centers,palette_colors, current_point)
                if selected_color:
                    current_color = selected_color + (255,)  # Add alpha channel for drawing

                finger_state = check_finger_up(hand_landmarks, w, h)
                if finger_state[1]:
                    current_point = (int(index_tip.x * w), int(index_tip.y * h))
                    if last_point is not None:
                        cv2.line(draw, last_point, current_point, current_color, thickness=2)
                    last_point = current_point
                else:
                    last_point = None

                # Draw hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            palm_open_time = None
            last_point = None

        # Combine the original image with the drawing
        img_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        combined_img = cv2.addWeighted(img_rgba, 1.0, draw, 1.0, 0)

        # Display the frame
        cv2.imshow('Hand Tracking', combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
