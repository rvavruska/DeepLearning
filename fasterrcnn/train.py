import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from TrainDataset import TrainDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset

def main():
    # Initialize Dataset
    train_dataset = TrainDataset('train/labels/detections.csv')
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )   

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 50

    step = 1
    print("TRAINING")
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model = model.to(device)

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step} loss:{loss_value}")

            step = step + 1

            lr_scheduler.step()
        
        print(f"Epoch {epoch} loss: {loss_value}")

    torch.save(model.state_dict(), 'model.pt')
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}, 'ckpt.pt')

            
if __name__ == "__main__":
    main()