import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset

class TrainDataset(Dataset):
    def __init__(self, annotation_file):
        super().__init__()

        self.train_df = pd.read_csv(file)
        self.image_ids = self.train_df['ImageID'].unique()

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image = cv2.imread(f'train/data/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        image = torchvision.transforms.ToTensor()(image)
        return image

    def __len__(self) -> int:
        return self.image_ids.shape[0]