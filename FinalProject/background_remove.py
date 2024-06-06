import cv2
import mediapipe as mp
import numpy as np
# 初始化MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

def change_background(frame, img):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_frame)
    mask = results.segmentation_mask
    condition = mask > 0.6
    h, w = frame.shape[:2]
    resized_img = cv2.resize(img, (w, h))
    output_frame = np.where(condition[:, :, None], frame, resized_img)
    return output_frame

def remove_background(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 处理图像，获取分割结果
    results = selfie_segmentation.process(rgb_frame)
    # 获取分割掩码
    mask = results.segmentation_mask
    condition = mask > 0.6

    # 创建白色背景
    bg_frame = np.ones_like(frame, dtype=np.uint8) * 255

    # 应用掩码来移除背景
    output_frame = np.where(condition[:, :, None], frame, bg_frame)
    return output_frame

if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 翻转图像，使其与镜像效果一致
        frame = cv2.flip(frame, 1)

        # 将图像从BGR转换为RGB
        output_frame = remove_background(frame)
        # 显示结果
        cv2.imshow('MediaPipe Background Removal', output_frame)

        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()