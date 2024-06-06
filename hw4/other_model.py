import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

def crop_and_resize(image, target_height, target_width):
    height, width, _ = image.shape
    if height > width:
        new_height = int(width * (target_height / target_width))
        new_width = width
    else:
        new_height = height
        new_width = int(height * (target_width / target_height))

    start_x = width // 2
    start_y = height // 2
    cropped_image = image[start_y - new_height // 2 :start_y + new_height // 2, start_x - new_width // 2:start_x + new_width // 2]
    resized_image = cv2.resize(cropped_image, (target_width, target_height))
    return resized_image

def process_and_display(camera_index, model, background_paths, gpu=False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    background = np.full((512, 512, 3), 255.0)
    backgrounds = [background] + [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in background_paths]
    
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    current_background = background
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            h, w = frame.shape[:2]
            if current_background.shape[:2] != (h, w):
                background_resized = crop_and_resize(current_background, h, w)
            else:
                background_resized = current_background

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch_transforms(Image.fromarray(frame_rgb)).unsqueeze(0)

            if gpu:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad():
                output = model(frame_tensor)['out'][0]
                matte_tensor = torch.sigmoid(output)
                matte_np = matte_tensor.cpu().numpy().transpose(1, 2, 0)
            print(matte_np.shape)
            matte_resized = cv2.resize(matte_np, (w, h))
            matte_resized = np.repeat(matte_resized[:, :, np.newaxis], 3, axis=2)  # 扩展为3通道
            print(matte_resized.shape)
            matte_resized = matte_resized[:,:,:,0]
            print("shape")
            print(matte_resized.shape)
            print(frame_rgb.shape)
            print(background_resized.shape)

            view_np = (matte_resized * frame_rgb + (1 - matte_resized) * background_resized).astype(np.uint8)
            cv2.imshow('Masked Output with Background', cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR))

            # 将 matte_resized 从 2D 数组扩展为 3D 数组以便进行广播
            # matte_resized = np.repeat(matte_resized[:, :, np.newaxis], 3, axis=2)
            # print(matte_resized.shape)
            # view_np = (matte_resized * frame_rgb + (1 - matte_resized) * background_resized).astype(np.uint8)
            # cv2.imshow('Masked Output with Background', cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR))

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
    print('Load pre-trained DeepLabV3+MobileNetV3 model...')
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    # model.classifier[4] = torch.nn.Conv2d(model.classifier[4].in_channels, 1, kernel_size=(1, 1), stride=(1, 1))
    model = torch.nn.DataParallel(model)

    GPU = True if torch.cuda.is_available() else False
    if GPU:
        print('Use GPU...')
        model = model.cuda()

    model.eval()

    background_imgs = ["/Users/liushiwen/Desktop/大四下/tcp.png", "/Users/liushiwen/Desktop/大四下/tcp_tp.png"]
    process_and_display(0, model, background_imgs, gpu=GPU)
