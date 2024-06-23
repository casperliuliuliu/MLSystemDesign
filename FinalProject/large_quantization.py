import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image
import numpy as np
from torch.nn.utils import prune

def change_background_large(frame, background_img, model, transform, device, flag):
    # Resize and prepare the frame
    h, w = frame.shape[:2]
    background_resized = cv2.resize(background_img, (w, h))
    background_resized = cv2.cvtColor(background_resized, cv2.COLOR_BGR2RGB)
    if not flag:
        return background_resized
    rate = w/192
    resized_frame = cv2.resize(frame, (192, int(h/rate)))  # Resize for faster processing
    img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Segment with the model
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)
    mask = (output_predictions == 15)  # 15 is typically the 'person' class

    # Convert mask to numpy and resize to original frame size
    mask = mask.cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8), (w, h))

    # Prepare the background

    # Replace the background
    output_frame = np.where(mask[:, :, None], frame, background_resized)
    return output_frame

if __name__ == "__main__":
    model = deeplabv3_mobilenet_v3_large(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    print(model)
    # model = torch.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {torch.nn.Linear, torch.nn.Conv2d},  # specify which layers to dynamically quantize
    #     dtype=torch.qint8  # specify the dynamic quantization type
    # )

    # for module in model.modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         prune.l1_unstructured(module, name='weight', amount=0.4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    background_img = cv2.imread('DPAML_20240607-021033.png')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Change the background using the DeepLabV3 model
            changed_frame = change_background_large(frame, background_img, model, transform, device)

            # Display the frame with the changed background
            cv2.imshow('Segmented Frame', changed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
