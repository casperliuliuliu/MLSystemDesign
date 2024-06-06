import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import pytesseract
from main import check_finger_up
def is_pinching(thumb, index):
    """ Determine if two fingers are close enough to consider pinching. """
    distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
    return distance < 0.05  # Adjust threshold based on your testing

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    """ Draw text on the image """
    cv2.putText(image, text, position, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands

# Initialize TTS engine
tts_engine = pyttsx3.init()

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
    palm_open_duration = 2
    palm_open_time = None
    drawn_text = ""  # To store the drawn text

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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                if is_pinching(thumb_tip, index_tip):
                    # Use OCR to recognize the drawn text
                    gray_draw = cv2.cvtColor(draw, cv2.COLOR_BGRA2GRAY)
                    _, binary_draw = cv2.threshold(gray_draw, 128, 255, cv2.THRESH_BINARY_INV)
                    recognized_text = pytesseract.image_to_string(binary_draw, config='--psm 8')
                    if recognized_text.strip():
                        drawn_text = recognized_text.strip()
                        tts_engine.say(drawn_text)
                        tts_engine.runAndWait()
                        draw = np.zeros((h, w, 4), dtype='uint8')  # Clear drawing after recognition

                finger_state = check_finger_up(hand_landmarks, w, h)
                if finger_state[1]:
                    drawing = True
                    current_point = (int(index_tip.x * w), int(index_tip.y * h))
                    if last_point is not None:
                        cv2.line(draw, last_point, current_point, (0, 255, 0, 255), thickness=2)
                    last_point = current_point
                else:
                    drawing = False
                    last_point = None

                # Draw hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            palm_open_time = None
            drawing = False
            last_point = None

        # Combine the original image with the drawing
        img_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        combined_img = cv2.addWeighted(img_rgba, 1.0, draw, 1.0, 0)

        # Draw the recognized text on the frame
        if drawn_text:
            draw_text(combined_img, f"Recognized: {drawn_text}", (50, 50))

        # Display the frame
        cv2.imshow('Hand Tracking', combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
