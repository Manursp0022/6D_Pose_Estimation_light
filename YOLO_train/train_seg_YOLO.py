from ultralytics import YOLO, settings
from utils.YOLO_utils.seg_yolo_utils_train_all import create_yolo_labels, create_yolo_config_all

class yolo_seg_trainer:
    def __init__(self,
            dataset_root, 
            model="yolov8n-seg.pt",
            epochs=10,
            patience=30,
            batch=16,
            freeze_layers = 5 #about half the backbone
        ):
        self.dataset_root = dataset_root
        self.model = model
        self.epochs = epochs
        self.patience = patience
        self.batch = batch
        self.freeze_layers = freeze_layers

    """def create_labels_and_config(self):
        print(f"___Data Preparation ___")
        train_split = "data\\autosplit_train_ALL.txt"
        val_split = "data\\autosplit_val_ALL.txt"
        create_yolo_labels(self.dataset_root, train_split, val_split)

        config_path = create_yolo_config_all(self.dataset_root, train_split, val_split)
        return config_path"""

    def train(self, config_path=None):
        #dataset_root = "C:\Users\gabri\Desktop\AML_project\6D_Pose_Estimation_light\dataset\Linemod_preprocessed" if args["dataset_root"] is None else args["dataset_root"]
        if config_path is None:
            print(f"[INFO] No config path provided, create it from utils.")
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
            verbose=False,
            freeze=self.freeze_layers,
        )

if __name__ == "__main__":
    dataset_root = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed"
    yolo = yolo_seg_trainer(dataset_root=dataset_root)
    config = yolo.create_labels_and_config()
    yolo.train(config_path=config)