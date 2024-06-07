import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406], std=[0.225])
])

# Paths to your data
train_image_paths = ["/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.jpg","/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.jpg","/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.jpg","/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.jpg",]
train_mask_paths = ["/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.png","/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.png","/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.png","/Users/liushiwen/Desktop/大四下/機器學習系統/MLSystemDesign/FinalProject/precident.png",]

train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_mobilenet_v3_large(pretrained=True)
# model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))  # num_classes is your dataset's number of classes

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        if masks.dim() == 4 and masks.shape[1] >= 1:
            masks = masks.argmax(dim=1)
        optimizer.zero_grad()
        print(images.shape)
        print(masks.shape)
        output = model(images)['out']
        print(output.shape)
        loss = criterion(output, masks)  # Ensure mask is LongTensor
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Training Loss: {epoch_loss}")

# Run training
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, criterion, optimizer, train_loader, device)
