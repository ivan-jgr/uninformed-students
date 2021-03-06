import torch
from PIL import Image
import settings
import pandas as pd
import numpy as np

from os.path import join
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):

    def __init__(self, root_dir, dataset_type, transform=None):
        """
        Arguments
        ---------
        root_dir:       path del folder con las imagenes png
        dataset_type:   train o val dataset
        transform:      transformaciones que se deben aplicar a las imagenes
        """
        super(ImageDataset, self).__init__()

        self.images_path = join(root_dir, 'img')
        self.dataframe = pd.read_csv(join(root_dir, 'dicom_info.csv'))

        with open(join(root_dir, dataset_type + '.txt'), 'r') as f:
            patients = f.read().splitlines()

        self.dataframe = self.dataframe[self.dataframe['study_instance_uid'].isin(patients)]

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        filename = row.filename
        labels = torch.FloatTensor(np.array(row['any':'subdural'].values, dtype='float'))
        #labels = torch.from_numpy(row['any':'subdural'].values, dtype='float')

        image = Image.open(join(self.images_path, filename)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, labels[0]


def get_data_loader(train_transform):
    train_dataset = ImageDataset('./data', 'train', train_transform)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=settings.batch_size,
                                   shuffle=True,
                                   num_workers=settings.workers,
                                   pin_memory=True,
                                   drop_last=True)
    return train_data_loader


def get_healthy_data_loader(train_transform, shuffle=True):
    train_dataset = ImageDataset('./data', 'healthy_train_sub', train_transform)
    train_data_loader = DataLoader(train_dataset, batch_size=settings.batch_size_healthy, shuffle=shuffle, num_workers=settings.workers, pin_memory=True, drop_last=True)

    return train_data_loader

def get_data_loaders(train_transform, val_transform, batch_size=1):
    train_dataset = ImageDataset('./data', 'train', train_transform)
    val_dataset = ImageDataset('./data', 'val_sub', val_transform)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=settings.workers,
                                   pin_memory=True,
                                   drop_last=True)

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=settings.workers,
                                 pin_memory=True,
                                 drop_last=False)

    data_loaders = {'train': train_data_loader, 'val': val_data_loader}

    return data_loaders


if __name__ == '__main__':
    import settings
    from torchvision import transforms

    t = transforms.Compose([transforms.ToTensor()])
    d = get_healthy_data_loader(t)
    d2 = get_data_loader(t)
    print("Healthy Size", len(d.dataset))
    print("Normal Size", len(d2.dataset))
