from DFRGBD_Evaluate import DF_RGBD_Net_Evaluator
import os

if __name__ == "__main__":

    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': "/Users/emanuelerosapepe/Desktop/test_YOLO/YOLO_ON_SMALL_TRAIN/autosplit_val_ALL.txt",
        'save_dir': "/Users/emanuelerosapepe/Desktop/test_YOLO",
        'model_dir': "checkpoints/",
        'batch_size': 32
    }

    evaluator = DF_RGBD_Net_Evaluator(config)
    evaluator.run()


"""
if __name__ == "__main__":
    # Assicurati di impostare l'ambiente per Mac
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    trainer = RefineTrainer()
    trainer.run()

"""

"""
self.cfg = {
            'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
            'split_train': "/Users/emanuelerosapepe/Desktop/test_YOLO/YOLO_ON_SMALL_TRAIN/autosplit_train_ALL.txt",
            'save_dir': "checkpoints_refine/",
            'main_weights': "checkpoints/best__DFRGBD.pth",
            'batch_size': 32, 
            'lr': 0.0001,
            'epochs': 30,
            'num_points_mesh': 500 # Quanti punti campioniamo dalla mesh per il refiner
        }

"""