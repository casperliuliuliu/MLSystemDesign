import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from your_dataset_module import YourDataset  # You need to define this
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.mobilenet import mobilenet_v3_small
from torchvision.models._utils import IntermediateLayerGetter

def create_deeplabv3_mobilenetv3_small(num_classes=21, pretrained_backbone=True):
    # Load MobileNetV3 Small as the backbone
    backbone = mobilenet_v3_small(pretrained=pretrained_backbone)
    # Select appropriate layers as feature extractors
    backbone_features = {
        'features': 'out',  # Assuming 'features' is the last feature layer
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=backbone_features)

    # Classifier and aux classifier are specific to the segmentation model
    classifier = deeplabv3_resnet50(pretrained=False).classifier
    aux_classifier = deeplabv3_resnet50(pretrained=False).aux_classifier

    # Assemble the DeepLabV3 model with the custom backbone
    model = deeplabv3_resnet50(pretrained=False)
    model.backbone = backbone
    model.classifier = classifier
    model.aux_classifier = aux_classifier

    return model
device = "cuda"
# Assuming 'create_deeplabv3_mobilenetv3_small' is your function to create the model
model = create_deeplabv3_mobilenetv3_small(num_classes=2)  # Background and person
model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Set up your data loaders
train_loader = DataLoader(YourDataset('train'), batch_size=4, shuffle=True)
val_loader = DataLoader(YourDataset('val'), batch_size=4)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    # Put the model into training mode
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)['out']

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print(f'Batch {i}, Loss {last_loss}')
            tb_writer.add_scalar('Training loss', last_loss, epoch_index * len(train_loader) + i)
            running_loss = 0.0

    return last_loss

# Main training loop
from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter()
epochs = 10  # This is a hyperparameter you can tune
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    train_one_epoch(epoch, tb_writer)

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
