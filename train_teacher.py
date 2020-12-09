import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from einops import rearrange
from models.AnomalyResnet import AnomalyResnet
from dataloader.dataloader import get_data_loader
from torchvision import transforms

EPOCHS = 100
model_name = './ckp/teacher_net.pth'

def distillation_loss(output, target):
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    resnet = AnomalyResnet()
    resnet.load_checkpoint('./ckp/best_model.pth', map_location=not torch.cuda.is_available())
    resnet = nn.Sequential(*list(resnet.children())[:-2])
    resnet.eval().to(device)

    # Teacher
    teacher = AnomalyResnet()
    teacher.to(device)

    # Optimizer
    optimizer = optim.Adam(teacher.parameters(), lr=2e-4, weight_decay=1e-5)

    # Data
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_loader = get_data_loader(data_transform)

    # training
    min_running_loss = np.inf
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, (batch, _) in tqdm(enumerate(data_loader)):
            # zero the parameters gradient
            optimizer.zero_grad()

            inputs = batch.to(device)
            with torch.no_grad():
                targets = rearrange(resnet(inputs), 'b vec h w -> b (vec h w)')
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
