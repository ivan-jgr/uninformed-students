import torch
import torch.nn as nn
import numpy as np
from einops import reduce
from models.PatchNet import PatchAnomalyNet
from models.FDFEAnomalyNet import FDEAnomalyNet as FDFEAnomalyNet
from dataloader.dataloader import get_data_loader, get_healthy_data_loader
from utils import increment_mean_and_var
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import settings

pH = 65
pW = 65
imH = 224
imW = 224
sL1, sL2, sL3 = 2, 2, 2
EPOCHS = 15
N_STUDENTS = 3
models_names = ['./ckp/student_net_%d.pth' % i for i in range(N_STUDENTS)]


def student_loss(output, target):
    err = reduce((output - target)**2, 'b h w vec -> b h w', 'sum')
    loss = torch.mean(err)
    return loss

def get_healthy_data(train_transforms, shuffle=True, batch_size=1):
    data_train = datasets.ImageFolder(root='/home/BiomedLab/IHD/png_data_train/positivos/', transform=train_transforms)
    #data_val = datasets.ImageFolder(root='../png_data_val/', transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
    #val_loader = torch.utils.data.DataLoader(data_val, batch_size=settings.batch_size, shuffle=True)

    return train_loader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Used device: {device}")

    teacher_hat = PatchAnomalyNet()
    teacher = FDFEAnomalyNet(base_net=teacher_hat, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
    #teacher = nn.DataParallel(teacher)
    teacher.eval().to(device)

    # Checkpoint
    teacher.load_state_dict(torch.load('./ckp/teacher_net.pth'))

    students_hat = [PatchAnomalyNet() for i in range(N_STUDENTS)]
    students = [FDFEAnomalyNet(base_net=student, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
                for student in students_hat
                ]
    #students = [nn.DataParallel(student) for student in students]
    students = [student.to(device) for student in students]
    #student[0] = students[0].load_state_dict(torch.load('./ckp/student_net_0.pth'))

    optimizers = [optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5) for student in students]

    # Data
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_loader = get_healthy_data(data_transform, shuffle=False)

    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for i, (batch, _) in tqdm(enumerate(data_loader)):
            inputs = batch.to(device)
            t_out = teacher(inputs)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)

    data_loader = get_healthy_data(data_transform)

    for j, student in enumerate(students):
        print(f"Training Student {j} on anomaly-free dataset...")
        min_running_loss = np.inf

        for epoch in range(EPOCHS):
            running_loss = 0.0

            for i, (batch, _) in tqdm(enumerate(data_loader)):
                optimizers[j].zero_grad()

                inputs = batch.to(device)
                with torch.no_grad():
                    targets = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
                outputs = student(inputs)
                loss = student_loss(targets, outputs)

                loss.backward()
                optimizers[j].step()
                running_loss += loss.item()

                if i % 10 == 9:
                    print(f"Epoch {epoch + 1}, iter {i+1} \t loss: {running_loss}")

                    if running_loss < min_running_loss:
                        print(f"Loss decreased: {min_running_loss} -> {running_loss}.")
                        print(f"Saving model to {models_names[j]}")
                        torch.save(student.state_dict(), models_names[j])

                    min_running_loss = min(min_running_loss, running_loss)
                    running_loss = 0.0

