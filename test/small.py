import torch
import torchvision.models as models
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import _deeplabv3_mobilenetv3

import cv2
def create_deeplabv3_mobilenetv3_small(num_classes=21, pretrained_backbone=True):
    # Load MobileNetV3 Small model
    mobilenetv3_small = models.mobilenet_v3_small(pretrained=pretrained_backbone)
    # Remove the last layer as we need to use it as a backbone
    backbone = IntermediateLayerGetter(mobilenetv3_small, 
                                       return_layers={'features': 'out'})

    # Create the DeepLabV3 model using the modified backbone
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    model.backbone = backbone
    
    # Replace the classifier to match the output channels of MobileNetV3 Small
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(576, num_classes)
    return model

def create_small(num_classes=21, pretrained_backbone=True):
    # Load MobileNetV3 Small model
    model = deeplabv3_mobilenet_v3_large(pretrained=True)
    backbone = models.mobilenet_v3_small(pretrained=pretrained_backbone)
    # Remove the last layer as we need to use it as a backbone
    # backbone = IntermediateLayerGetter(mobilenetv3_small, 
    #                                    return_layers={'features': 'out'})
    aux_loss = None
    # backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    return model

model = create_deeplabv3_mobilenetv3_small(num_classes=21)
model.eval()
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
        # frame[~mask] = [255, 255, 255]

        # Display the frame
        cv2.imshow('Segmented Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
