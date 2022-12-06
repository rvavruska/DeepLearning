import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from TestDataSet import TestDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt

def main():
    # Initialize Dataset
    test_dataset = TestDataset('test/labels/detections.csv')
    def collate_fn(batch):
        return tuple(zip(*batch))

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )   

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    images,targets = next(iter(test_data_loader))
    images = list(image.to(device) for image in images)
    outputs = model(images)
    #print(targets[0]['boxes'])
    boxes = outputs[0]['boxes'].data.cpu().numpy().astype(np.int32)
    img = cv2.cvtColor(images[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)
    fig, ax= plt.subplots(1,1, figsize=(12,6))

    for box in boxes:
        print(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 1)

    ax.set_axis_off()
    ax.imshow(img)

    plt.show()

            
if __name__ == "__main__":
    main()