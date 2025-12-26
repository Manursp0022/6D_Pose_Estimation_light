#from DFRGBD_Evaluate import DF_RGBD_Net_Evaluator
#from Refiner_Evaluate import RefinedEvaluator
import os
import re
import torch
from DAMF_Eval import DAMF_Evaluator
from Refinement_Section.evaluate_with_ICP import ICPEvaluator
from PoseNet_evaluation import PoseNetEvaluator
from Trainer_A100 import DAMFTurboTrainerA100

if __name__ == "__main__":

    config = {
        'dataset_root': '/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed',
        'split_train': 'data/autosplit_train_ALL.txt',
        'split_val': 'data/autosplit_val_ALL.txt',
        'save_dir': 'checkpoints/',

        # Training
        'epochs': 100,
        'batch_size': 32,

        'use_weighted_loss': False,   # ← Da False a True
        'rot_weight': 1,            # ← Aggiungi questo
        'trans_weight': 4,          # ← Aggiungi questo

        # Model
        'temperature': 2.0,
        'num_points_mesh': 500,

        # Optimizer
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6,

        # Regularization
        'early_stop_patience': 30
    }

    trainer = DAMFTurboTrainerA100(config)
    trainer.run()

    """

    config = {
        # Percorsi
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",  # MODIFICA QUESTO
        'split_val': "data/autosplit_val_ALL.txt",  # MODIFICA QUESTO
        'model_dir': 'checkpoints/',  # Directory con best_DAMF.pth
        'save_dir': 'checkpoints_results/',
        
        # Parametri
        'batch_size': 32,
        'num_workers': 12,  # 0 su Mac, 4-8 su Linux
        'temperature': 2.0,  # Stesso usato in training
    }
    
    # Crea directory output
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Esegui valutazione
    evaluator = DAMF_Evaluator(config)
    results = evaluator.run()
    
    print("\n✅ Evaluation completed!")
    print(f"Final Accuracy: {results['accuracy']:.2f}%")
    print(f"Mean ADD: {results['mean_add_cm']:.2f} cm")
    """
    """
    config = {
        # Percorsi
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",  
        'split_val': "data/autosplit_val_ALL.txt", 
        'model_dir': 'checkpoints/',  
        'save_dir': 'checkpoints_results/',
        'batch_size': 32,
    }
    
    evaluator = PoseNetEvaluator(config)
    evaluator.run()
    """
    """
    config = {
    'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
    'split_val': "data/autosplit_val_ALL.txt",
    'save_dir': "checkpoints/", 
    'temperature': 2.0,
    }
    
    evaluator = ICPEvaluator(config)
    evaluator.evaluate()

    """
    """
    config_eval = {
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",
        'split_val': 'data/autosplit_val_ALL.txt',
        
        'checkpoint': 'checkpoints/best_turbo_model_A100.pth', 
        
        'batch_size': 24,
        'temperature': 1.0,
        'num_points_mesh': 500
    }

    # 1. Inizializza
    evaluator = DAMF_Evaluator(config_eval)

    # 2. Lancia valutazione
    mean_error, accuracy = evaluator.validate()
    """
    """
    # Configurazione
    config = {
        'dataset_root': "/content/dataset/Linemod_preprocessed",
        'split_train': "/content/6D_Pose_Estimation_light/data/autosplit_train_ALL.txt",
        'save_dir': "checkpoints/",
        'main_weights': "checkpoints/best_turbo_model_A100.pth", # Assicurati che esista!
        'batch_size': 32, # 32 è sicuro, 64 se hai VRAM
        'lr': 0.0001,
        'epochs': 20
    }
    
    trainer = IterativeRefineTrainer(config)
    trainer.train()
    """
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