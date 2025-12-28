import argparse
import os 
import wandb
from ultralytics import YOLO, settings
from utils.YOLO_utils.yolo_utils_train_all import create_yolo_labels, create_yolo_config

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    settings.update({"wandb": True})

    print(f"___Data Preparation ___")
    create_yolo_labels(args.dataset_root)
    
    config_path = create_yolo_config(args.dataset_root)
    
    print(f"___Starting training___")
    model = YOLO(args.model)

    model.train(
        data=config_path,
        epochs=args.epochs,
        patience=30,
        batch=args.batch,
        imgsz=640,
        project="linemod-detection",  
        name=f"YOLO_{args.model}_ep{args.epochs}",
        val=True,
        save=True,
        exist_ok=False,
        pretrained=True,
        optimizer='auto',
        verbose=True,
    )

if __name__ == "__main__":
    train()