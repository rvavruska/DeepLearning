import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset
from os import path
import warnings

warnings.filterwarnings("ignore")

class TrainDataset(Dataset):
    def __init__(self, file):
        super().__init__()

       # /m/0k4j = Car
        #print(file)
        self.train_df = pd.read_csv(file)
        #print(self.train_df.columns)
        self.image_ids = self.train_df['ImageID'].unique()
        #print(self.image_ids)
        delete = []

        # Delete images that don't exist from list
        for i, id in enumerate(self.image_ids):
            image_id = id
            if not path.exists(f'train/data/{image_id}.jpg'):
                delete.append(i)

        #print(self.image_ids)
        #print(len(self.image_ids))

        self.image_ids = np.delete(self.image_ids, delete)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        #print(image_id)
        bboxes = self.train_df[self.train_df['ImageID'] == image_id]
        #print(bboxes)
        

        image = cv2.imread(f'train/data/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.resize(image, (416, 416), interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        warnings.filterwarnings("ignore")

        # Fix the box dimensions
        bboxes.loc[:, 'XMin'] = bboxes.loc[:, 'XMin'] * image.shape[1]
        #print(bboxes.loc[:, 'XMin'] * image.shape[1])
        bboxes.loc[:, 'XMax'] = bboxes.loc[:, 'XMax'] * image.shape[1]
        bboxes.loc[:, 'YMin'] = bboxes.loc[:, 'YMin'] * image.shape[0]
        bboxes.loc[:, 'YMax'] = bboxes.loc[:, 'YMax'] * image.shape[0]

        boxes = bboxes[['XMin', 'YMin', 'XMax', 'YMax']].values
        #print(boxes)
        area = (boxes[:, 3] - boxes [:, 1] * boxes[:, 2] - boxes[:, 0])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        image = torchvision.transforms.ToTensor()(image)
        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]