import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image
import numpy as np
model = deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()  # Set the model to evaluation mode
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
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Transform the image
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0)

        # Generate a binary mask for the 'person' class
        person_class_index = 15
        mask = (output_predictions == person_class_index).cpu().numpy()

        # Apply mask to the frame (for example, set background to white)
        frame[~mask] = [255, 255, 255]

        # Display the frame
        cv2.imshow('Segmented Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
