import argparse
import os 
import wandb
from ultralytics import YOLO, settings
from utils.YOLO_utils.seg_yolo_utils_train_all import create_yolo_labels, create_yolo_config_all

class yolo_seg_trainer:
    def __init__(self,
            dataset_root, 
            train_split, 
            val_split, 
            model="yolov8n-seg.pt",
            epochs=10,
            patience=30,
            batch=16,
            ):
        self.dataset_root = dataset_root
        self.train_split = train_split
        self.val_split = val_split
        self.model = model
        self.epochs = epochs
        self.patience = patience
        self.batch = batch

    def train(self):
        #dataset_root = "C:\Users\gabri\Desktop\AML_project\6D_Pose_Estimation_light\dataset\Linemod_preprocessed" if args["dataset_root"] is None else args["dataset_root"]
        settings.update({"wandb": True})

        print(f"___Data Preparation ___")
        create_yolo_labels(self.dataset_root, self.train_split, self.val_split)
        
        config_path = create_yolo_config_all(self.dataset_root, self.train_split, self.val_split)
        
        print(f"___Starting training___")
        model = YOLO(self.model)

        model.train(
            data=config_path,
            epochs=self.epochs,
            patience=self.patience,
            batch=self.batch,
            imgsz=640,
            project="linemod-segmentation",  
            name=f"YOLO_{self.model}_ep{self.epochs}",
            val=True,
            save=True,
            exist_ok=False,
            pretrained=True,
            optimizer='auto',
            verbose=True,
        )

