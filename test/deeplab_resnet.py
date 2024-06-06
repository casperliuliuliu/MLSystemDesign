import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

# Load pre-trained DeepLabV3 model
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((720, 1280)),
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply transformations
    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    
    # Create a mask for the foreground
    mask = (output_predictions == 15).cpu().numpy()  # Assuming '15' is the class for person
    # print(output_predictions.shape)
    # Apply mask to frame
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    # print(frame.shape)
    frame[mask == 0] = [255, 255, 255]  # Change background to white
    
    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
