import numpy as np
from torch.utils.data import Dataset
import cv2
import torchvision

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        image = cv2.imread(self.image_path)  # (h, w, c)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.mask_path)  # (h, w)
        mask = np.eye(6)[mask]  # (h, w, c)
        to_tensor = torchvision.transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)
        return image, mask