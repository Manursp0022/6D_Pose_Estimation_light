#from DFRGBD_Evaluate import DF_RGBD_Net_Evaluator
#from Refiner_Evaluate import RefinedEvaluator
import os
from Refinement_Section.evaluate_with_ICP_Nogt import ICPEvaluator
import re
import torch
if __name__ == "__main__":
    # Configurazione
    torch.set_num_threads(1)
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': "data/autosplit_val_ALL.txt",
        'save_dir': "checkpoints/", 
        'temperature': 2.0
    }
    
    # Fix Mac issue
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    evaluator = ICPEvaluator(config)
    evaluator.evaluate()
    """
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': "data/autosplit_val_ALL.txt",
        'save_dir': "checkpoints/", 
    }
    evaluator = ZoomLoopEvaluator(config)
    evaluator.run(iterations=2)
    """

    """

    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_train': "data/autosplit_val_ALL.txt",
        'save_dir': "checkpoints/",
        'main_weights': "checkpoints//best_turbo_model_A100.pth", # Baseline weights
        'batch_size': 128, 
        'lr': 0.0001,
        'epochs': 30,     
    }
    
    trainer = RefineTrainerDelta(config)
    trainer.train()
    """

    """
    # CONFIGURAZIONE RAPIDA
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_train': "data/autosplit_train_ALL.txt",
        'split_val': "data/autosplit_val_ALL.txt",
        'save_dir': "checkpoints_turbo/",
        'batch_size': 32,
        'lr': 0.0001,
        'epochs': 10,           # Train corto per debug
        'early_stop_patience': 20,
        'scheduler_patience': 10,
        'num_points_mesh': 500, # Per la Loss
        'temperature': 2.0      # Parametro Confidence (prova 1.0, 2.0, 5.0)
    }

    trainer = DFMdAtt_Trainer(config)
    trainer.run()
    """
    """
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': "/Users/emanuelerosapepe/Desktop/test_YOLO/YOLO_ON_SMALL_TRAIN/autosplit_val_ALL.txt",
        'save_dir': "checkpoints/",
        'save_dir_refine': "checkpoints_refine/",
        'num_points_mesh': 500
    }
    # Se sei su A100, ricordati di cambiare i path!
    evaluator = RefinedEvaluator(config)
    evaluator.run()
    """
    """
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_train': "/Users/emanuelerosapepe/Desktop/test_YOLO/YOLO_ON_SMALL_TRAIN/autosplit_train_ALL.txt",
        'save_dir': "checkpoints_refine/",
        'main_weights': "checkpoints/best__DFRGBD.pth",
        'batch_size': 32, 
        'lr': 0.0001,
        'epochs': 30, # Bastano 30 epoche
        'num_points_mesh': 500
    }

    trainer = RefineTrainer(config)
    trainer.run()
    """


"""
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': "/Users/emanuelerosapepe/Desktop/test_YOLO/YOLO_ON_SMALL_TRAIN/autosplit_val_ALL.txt",
        'save_dir': "checkpoints/",
        'save_dir_refine': "checkpoints_refine/",
        'num_points_mesh': 500
    }
    # Se sei su A100, ricordati di cambiare i path!
    evaluator = RefinedEvaluator(config)
    evaluator.run()
""" # Configurazione
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

"""
if __name__ == "__main__":
    config = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': "/Users/emanuelerosapepe/Desktop/test_YOLO/YOLO_ON_SMALL_TRAIN/autosplit_val_ALL.txt",
        'save_dir': "checkpoints/",
        'save_dir_refine': "checkpoints_refine/",
        'num_points_mesh': 500
    }
    # Se sei su A100, ricordati di cambiare i path!
    evaluator = FinalEvaluator(config)
    evaluator.run()
    """