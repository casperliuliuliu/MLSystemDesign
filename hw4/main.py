import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from modnet import MODNet
import torch.nn as nn

def crop_and_resize(image, target_height, target_width):
    """
    對圖片進行中心裁切並調整至目標尺寸，使其維持等比例放縮。
    """
    height, width, _ = image.shape

    # 根據目標高寬比例決定新的裁切尺寸
    if height > width:
        new_height = int(width * (target_height / target_width))
        new_width = width
    else:
        new_height = height
        new_width = int(height * (target_width / target_height))

    # 計算裁切中心點
    start_x = width // 2
    start_y = height // 2

    # 進行中心裁切
    cropped_image = image[start_y - new_height // 2 :start_y + new_height // 2, start_x - new_width // 2:start_x + new_width // 2]

    # 調整圖片至目標尺寸
    resized_image = cv2.resize(cropped_image, (target_width, target_height))

    return resized_image

def process_and_display(camera_index, modnet, background_paths, gpu=False):
    # 初始化攝影機
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    # 初始化背景圖片
    background = np.full((512, 512, 3), 255.0)  # 初始空白背景
    backgrounds = [background] + [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in background_paths]  # 讀取並轉換背景圖片

    # 圖片轉換規則定義
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])

    current_background = background  # 設定當前背景為空白

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            h, w = frame.shape[:2]

            # 調整背景圖片大小以匹配攝像頭捕捉的畫面
            if current_background.shape[:2] != (h, w):
                background_resized = crop_and_resize(current_background, h, w)
            else:
                background_resized = current_background

            # 將攝像頭捕捉的BGR圖像轉為RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch_transforms(Image.fromarray(frame_rgb)).unsqueeze(0)

            if gpu:
                frame_tensor = frame_tensor.cuda()

            # 使用MODNet進行人像分割
            with torch.no_grad():
                matte_tensor = modnet(frame_tensor).repeat(1, 3, 1, 1)

            # 轉換分割結果並調整大小
            matte_np = matte_tensor[0].cpu().numpy().transpose(1, 2, 0)
            matte_resized = cv2.resize(matte_np, (w, h))

            # 結合人像和背景圖片
            view_np = (matte_resized * frame_rgb + (1 - matte_resized) * background_resized).astype(np.uint8)
            cv2.imshow('Masked Output with Background', cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key - ord('0') < len(backgrounds):
                current_background = backgrounds[key - ord('0')]
                background_resized = cv2.resize(current_background, (w, h))

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 配置和載入預訓練的MODNet模型
    print('Load pre-trained MODNet...')
    pretrained_ckpt = '/Users/liushiwen/Desktop/大四下/modnet_webcam_portrait_matting.ckpt'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    # 檢查是否使用GPU
    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_ckpt))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    modnet.eval()
    # 背景圖片路徑列表
    background_imgs = ["/Users/liushiwen/Desktop/大四下/bg.jpg", "/Users/liushiwen/Desktop/大四下/Ackley.jpg", "/Users/liushiwen/Desktop/大四下/resized_image.jpg", "/Users/liushiwen/Desktop/大四下/bg4.jpg"]
    process_and_display(1, modnet,background_imgs, gpu=GPU)
