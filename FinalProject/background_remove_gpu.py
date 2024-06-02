import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# 定义U2NET模型
class U2NET(nn.Module):
    # 定义U2NET模型的架构
    # 请从U2NET的官方实现中导入或者手动定义
    pass
def load_model(model_path):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    return net

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def postprocess(output, original_image_size):
    output = output.squeeze().cpu().detach().numpy()
    output = (output - output.min()) / (output.max() - output.min())
    output = (output * 255).astype(np.uint8)
    output = cv2.resize(output, original_image_size, interpolation=cv2.INTER_LINEAR)
    return output

def remove_background(frame, model):
    # 预处理
    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_image_size = original_image.size
    input_image = preprocess(original_image)

    # 获取分割掩码
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = model(input_image)
        mask = d1[:, 0, :, :]

    # 后处理掩码
    mask = postprocess(mask, original_image_size)

    # 应用掩码到原始图像
    mask_inv = cv2.bitwise_not(mask)
    bg_removed = cv2.bitwise_and(frame, frame, mask=mask)

    return bg_removed

def main():
    # 加载模型
    model_path = 'path_to_u2net.pth'
    model = load_model(model_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 移除背景
        result = remove_background(frame, model)

        # 显示结果
        cv2.imshow('Background Removal', result)

        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()