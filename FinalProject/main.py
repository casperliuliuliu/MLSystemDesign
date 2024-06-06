import cv2
import os
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from finger_detection import check_finger_up
from finger_detection import calculate_palm_width
from color_palette import draw_palette, check_palette_touch, draw_text
from background_remove import remove_background, change_background
mp_hands = mp.solutions.hands

def save_frame_with_timestamp(frame, prefix='DPAML_'):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}{timestamp}.png"
    cv2.imwrite(filename, frame)

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
    # print(f"Distances: Thumb-Index = {dist_thumb_index/ palm_width}, \nThumb-Pinky = {dist_thumb_pinky/ palm_width}, \npalm_width = {palm_width}")
    return open_hand

def list_images_in_folder(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_files.append(filename)
    return image_files

def main():
    background_folder_path = "/Users/liushiwen/Desktop/大四下/群體智慧/SwarmIntelligence/midterm_report/function_sim/"
    pinch_time = 0
    background_index = 0
    palm_open_duration = 1
    pinch_time = 2
    yeah_time = 0
    background_list = list_images_in_folder(background_folder_path)
    print(background_list)
    background_img = cv2.imread(background_folder_path + background_list[background_index])
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    current_color = (0, 200, 0, 200)  # Initial drawing color
    is_pinching_displayed = False
    is_palm_opened = False
    is_yeah_displayed = False
    save_flag = False
    with mp_hands.Hands(
        model_complexity=0,
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
        palm_open_time = None
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            img = cv2.flip(img, 1)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Process image in RGB

            # Process the image and detect hands
            results = hands.process(img2)

            # Check if any hand landmarks were detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    is_palm_opened = check_open_palm(hand_landmarks, h)
                    if is_palm_opened:
                        cv2.putText(background_img, '', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        if palm_open_time is None:
                            palm_open_time = time.time()
                        elif time.time() - palm_open_time >= palm_open_duration:
                            draw = np.zeros((h, w, 4), dtype='uint8')  # Clear drawing
                    else:
                        palm_open_time = None
                        finger_state = check_finger_up(hand_landmarks, w, h)
                        # print(finger_state)
                        if finger_state[1] and finger_state[1:] == [True, False, False, False]:
                            current_point = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
                            selected_color = check_palette_touch(palette_centers, palette_colors, current_point)
                            if selected_color:
                                current_color = selected_color + (255,)  # Add alpha channel for drawing

                            if last_point is not None:
                                cv2.line(draw, last_point, current_point, current_color, thickness=4)
                            last_point = current_point
                            pinch_time = 0

                        else:
                            last_point = None
                    
                    
                    if finger_state == [False, True, True, False, False]: 
                        yeah_time += 1
                        is_yeah_displayed = True
                    else:
                        save_flag = False
                        yeah_time = 0
                        is_yeah_displayed = False

                    if yeah_time > 10:
                        save_flag = True


                    if finger_state == [False, False, True, True, True]: 
                        pinch_time += 1
                        is_pinching_displayed = True
                    else:
                        pinch_time = 0
                        is_pinching_displayed = False
                        
                    if pinch_time > 10:
                        background_index +=1
                        if background_index >= len(background_list):
                            background_index %= len(background_list)
                        pinch_time = 0
                        draw = np.zeros((h, w, 4), dtype='uint8')
                        background_img = cv2.imread(background_folder_path + background_list[background_index])
                        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
                    
            else:
                palm_open_time = None
                last_point = None

            # img = remove_background(img)
            img = change_background(img, background_img) 
            palette_centers, palette_colors = draw_palette(img)
            
            for j in range(w):
                img[:,j,0] = img[:,j,0]*(1-draw[:,j,3]/255) + draw[:,j,0]*(draw[:,j,3]/255)
                img[:,j,1] = img[:,j,1]*(1-draw[:,j,3]/255) + draw[:,j,1]*(draw[:,j,3]/255)
                img[:,j,2] = img[:,j,2]*(1-draw[:,j,3]/255) + draw[:,j,2]*(draw[:,j,3]/255)
            # Overlay the drawing canvas on the video frame
            # img = cv2.addWeighted(img, 1, draw, 0.8, 0)
            if is_yeah_displayed:
                if save_flag:
                    save_frame_with_timestamp(img)
                    draw_text(img, 'IMG_saved!', (100, 50))
                    save_flag = False
                    is_yeah_displayed = False
                else:
                    draw_text(img, 'Yeah!', (100, 50))
            elif is_pinching_displayed:
                draw_text(img, 'Pinching!', (100, 50))
            elif is_palm_opened:
                draw_text(img, 'PALM OPENED!', (100, 50))
                


            cv2.imshow('Finger Drawing', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def plot_img(img):
    # Plot the image using matplotlib
    plt.imshow(img)
    plt.title('Background Image')
    plt.axis('off')  # Hide the axis
    plt.show()

if __name__ == "__main__":
    main()


