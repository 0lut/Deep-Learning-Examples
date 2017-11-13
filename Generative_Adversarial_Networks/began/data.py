from os.path import join
from dataset import DataSetFromFolder
import torchvision.transforms as transforms


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")
    
    return DataSetFromFolder(train_dir)


def get_test_set(root_dir):
    
    test_dir = join(root_dir, "test")
    
    return DataSetFromFolder(test_dir)
