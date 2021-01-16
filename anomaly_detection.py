import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from einops import rearrange, reduce
from models.PatchNet import PatchAnomalyNet
from models.FDFEAnomalyNet import FDEAnomalyNet as FDFEAnomalyNet
from dataloader.dataloader import get_data_loader, get_healthy_data_loader, get_data_loaders
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from utils import increment_mean_and_var

pH = 65
pW = 65
imH = 224
imW = 224
sL1, sL2, sL3 = 2, 2, 2
N_STUDENTS = 3
N_TEST = 200

def get_error_map(students_pred, teacher_pred):
    # student: (batch, student_id, h, w, vector)
    # teacher: (batch, h, w, vector)
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    err = reduce((mu_students - teacher_pred)**2, 'b h w vec -> b h w', 'sum')
    return err


def get_variance_map(students_pred):
    # student: (batch, student_id, h, w, vector)
    sse = reduce(students_pred**2, 'b id h w vec -> b id h w', 'sum')
    msse = reduce(sse, 'b id h w -> b h w', 'mean')
    mu_students = reduce(students_pred, 'b id h w vec -> b h w vec', 'mean')
    var = msse - reduce(mu_students**2, 'b h w vec -> b h w', 'sum')
    return var


def predict_student(students, inputs, n=2):
    s_out = torch.stack([student(inputs) for i in range(n) for student in students], dim=1)
    return s_out

def get_healthy_data(train_transforms, shuffle=True, batch_size=1):
    data_train = datasets.ImageFolder(root='/home/BiomedLab/IHD/png_data_train/positivos/', transform=train_transforms)
    #data_val = datasets.ImageFolder(root='../png_data_val/', transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
    #val_loader = torch.utils.data.DataLoader(data_val, batch_size=settings.batch_size, shuffle=True)

    return train_loader


def get_data(val_transforms):
    #data_train = datasets.ImageFolder(root='/home/BiomedLab/IHD/png_data_train/', transform=train_transforms)
    data_val = datasets.ImageFolder(root='/home/BiomedLab/IHD/png_data_val/', transform=val_transforms)

    #train_loader = torch.utils.data.DataLoader(data_train, batch_size=settings.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=1, shuffle=True)

    return val_loader


if __name__ == '__main__':
    device = torch.device("cuda:1")
    print(f"Device used: {device}")

    #Teacher network
    teacher_hat = PatchAnomalyNet()
    teacher = FDFEAnomalyNet(base_net=teacher_hat, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
    #teacher = nn.DataParallel(teacher)
    teacher.eval().to(device)

    # Load teacher model
    teacher.load_state_dict(torch.load('./ckp/teacher_net.pth'))

    # Students
    students_hat = [PatchAnomalyNet() for i in range(N_STUDENTS)]
    students = [FDFEAnomalyNet(base_net=student, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
                for student in students_hat]
    students = [student.eval().to(device) for student in students]

    # Loading students models
    for i in range(N_STUDENTS):
        model_name = f'./ckp/student_net_{i}.pth'
        students[i].load_state_dict(torch.load(model_name))

    # Calibration on anomaly-free dataset
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_loader = get_healthy_data(data_transform, shuffle=False)
    
    with torch.no_grad():
        print('Callibrating teacher on Student dataset.')
        t_mu, t_var, t_N = 0, 0, 0
        for i, (batch, _) in tqdm(enumerate(data_loader)):
            inputs = batch.to(device)
            t_out = teacher(inputs)
            t_mu, t_var, t_N = increment_mean_and_var(t_mu, t_var, t_N, t_out)

    """
    #print("t_mu", t_mu)
    t_mu = torch.load("t_mu.pth")
    #print("t_var", t_var)
    t_var = torch.load("t_var.pth")
    #print("t_N", t_N)
    t_N = torch.load( "t_N")
    """ 
    with torch.no_grad():
        print("Callibrating scoring parameters on Student dataset.")
        max_err, max_var = 0, 0
        mu_err, var_err, N_err = 0, 0, 0
        mu_var, var_var, N_var = 0, 0, 0
        for i, (batch, _) in tqdm(enumerate(data_loader)):
            inputs = batch.to(device)
            t_out = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student(inputs) for student in students], dim=1)

            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            mu_err, var_err, N_err = increment_mean_and_var(mu_err, var_err, N_err, s_err)
            mu_var, var_var, N_var = increment_mean_and_var(mu_var, var_var, N_var, s_var)

            max_err = max(max_err, torch.max(s_err))
            max_var = max(max_var, torch.max(s_var))
            #print("End of iteración ", i)
    """
    #print("max_err", max_err)
    max_err = torch.load("max_err.pth")
    #print("max_var", max_var)
    max_var = torch.load("max_var.pth")
    #print("mu_err", mu_err)
    mu_err = torch.load("mu_err.pth")
    #print("var_err", var_err)
    var_err = torch.load("var_err.pth")
    #print("N_err", N_err)
    N_err = torch.load("N_err.pth")
    #print("mu_var", mu_var)
    mu_var = torch.load("mu_var.pth")
    #print("var_var", var_var)
    var_var = torch.load("var_var.pth")
    #print("N_var", N_var)
    N_var = torch.load("N_var.pth")
    """
    # Loading test data
    data_loader =  get_data(data_transform)
    test_set = iter(data_loader)

    # Get back to the original
    unorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.409/0.225], std=[1/0.229, 1/0.24, 1/0.225])

    #Build anomaly map
    with torch.no_grad():
        for i in range(N_TEST):
            batch, lbl = next(test_set)
            inputs = batch.to(device)
            label = lbl.cpu()
            anomaly = 'with' if label.item() == 1 else 'without'

            t_out = (teacher(inputs) - t_mu) / torch.sqrt(t_var)
            s_out = torch.stack([student(inputs.to(device)) for student in students], dim=1)

            s_err = get_error_map(s_out, t_out)
            s_var = get_variance_map(s_out)
            score_map = (s_err - mu_err) / torch.sqrt(var_err) + (s_var - mu_var) / torch.sqrt(var_var)

            img_in = unorm(rearrange(inputs, 'b c h w -> c h (b w)').cpu())
            img_in = rearrange(img_in, 'c h w -> h w c')

            score_map = rearrange(score_map, 'b h w -> h (b w)').cpu()
            
            #if anomaly == 'with':
            #    torch.save(inputs, f'./images/{i}-img.png')
            #    torch.save(score_map, f'./images/{i}.pth')
                

            plt.figure(figsize=(13, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(img_in)
            plt.title(f'Original image - {anomaly} anomaly')

            plt.subplot(1, 2, 2)
            plt.imshow(score_map, cmap='jet')
            plt.imshow(img_in, cmap='gray', interpolation='none')
            plt.imshow(score_map, cmap='jet', alpha=0.5, interpolation='none')
            plt.colorbar(extend='both')
            plt.title('Anomaly map')

            max_score = (max_err - mu_err) / torch.sqrt(var_err) + (max_var - mu_var) / torch.sqrt(var_var)
            plt.clim(0, max_score.item())
            plt.savefig(f'./images/{i}.png')
