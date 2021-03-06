import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from einops import rearrange
from models.AnomalyResnet import AnomalyResnet
from models.PatchNet import PatchAnomalyNet
from dataloader.dataloader import get_data_loader
from torchvision import transforms, datasets
import settings

EPOCHS = 500
model_name = './ckp/teacher_net.pth'

def distillation_loss(output, target):
    #print("output size", output.size())
    #print("target size", target.size())
    err = torch.norm(output - target, dim=1) ** 2
    loss = torch.mean(err)
    return loss


def compactness_loss(output):
    _, n = output.size()
    avg = torch.mean(output, axis=1)
    std = torch.std(output, axis=1)
    zt = output.T - avg
    zt /= std
    corr = torch.matmul(zt.T, zt) / (n - 1)
    loss = torch.sum(torch.triu(corr, diagonal=1) ** 2)
    return loss


def get_data(train_transforms):
    data_train = datasets.ImageFolder(root='/home/BiomedLab/IHD/png_data_train/', transform=train_transforms)
    #data_val = datasets.ImageFolder(root='../png_data_val/', transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=settings.batch_size, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(data_val, batch_size=settings.batch_size, shuffle=True)

    return train_loader



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    resnet = AnomalyResnet()
    resnet.load_checkpoint('/home/BiomedLab/IHD/intracraneal-hem-detection/checkpoints/best_model_keys.pth', map_location=not torch.cuda.is_available())
    resnet.backbone.fc = nn.Identity()
    resnet.eval().to(device)

    # Teacher
    teacher = PatchAnomalyNet()
    teacher.to(device)

    # Optimizer
    optimizer = optim.Adam(teacher.parameters(), lr=2e-4, weight_decay=1e-5)

    # Data
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((65, 65)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_loader = get_data(data_transform)

    # training
    min_running_loss = np.inf
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, (batch, _) in tqdm(enumerate(data_loader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            inputs = batch.to(device)
            with torch.no_grad():
                targets = resnet(inputs)
            outputs = teacher(inputs)
            loss = distillation_loss(outputs, targets) + compactness_loss(outputs)

            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, iter {i + 1} \t loss: {running_loss}")

        if running_loss < min_running_loss:
            print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
            print(f"Saving model to {model_name}.")
            torch.save(teacher.state_dict(), model_name)

        min_running_loss = min(min_running_loss, running_loss)
        running_loss = 0.0
