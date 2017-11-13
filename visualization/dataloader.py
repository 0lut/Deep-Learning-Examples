from utils import *
import os
from torch.utils import data
from torchvision import transforms
from PIL import Image

class ImageFolfer(data.Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)



def get_loader(image_path, image_size, batch_size, transform, num_workers=2):
    
    dataset = ImageFolfer(image_path, transform)
    data_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers)
    return data_loader

