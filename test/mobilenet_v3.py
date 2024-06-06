import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation import fcn
import cv2

# Function to create the MobileNetV3 model adapted for segmentation
def create_mobilenetv3_segmentation_model(num_classes=21, pretrained_backbone=True):
    # Load MobileNetV3
    mobilenetv3 = models.mobilenet_v3_large(pretrained=pretrained_backbone)
    # We select the last layer of the features which is usually good for segmentation tasks
    backbone = create_feature_extractor(mobilenetv3, return_nodes={'features': 'out'})

    # Classifier head
    classifier = fcn.FCNHead(960, num_classes)  # 960 is the number of output channels from MobileNetV3 features

    # Create the FCN model
    model = fcn.FCN(backbone, classifier)
    return model

# Instantiate the segmentation model
model = create_mobilenetv3_segmentation_model(num_classes=21)
model.eval()

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 640)),  # Resize to maintain speed
    transforms.ToTensor(),
])

def process_frame(frame):
    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    
    # Create a mask for the foreground (assuming class '15' is the target class)
    mask = output_predictions == 13
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    
    # Apply mask to frame
    frame[~mask.cpu().numpy()] = [255, 255, 255]  # Change background to white
    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    
    cv2.imshow('Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
