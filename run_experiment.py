import os
import re
import torch
from baseline_extension.PWDF_Eval_NoMask import DAMF_Evaluator
from baseline_extension.PWDF_Eval_WMask import DAMF_Evaluator_WMask
from baseline_extension.Trainer_PWDF import DAMFTurboTrainerA100

if __name__ == "__main__":

    config = {
        # Percorsi
        'dataset_root': "",  
        'split_val': "data/autosplit_val_ALL.txt",  
        'model_dir': 'checkpoints/', 
        'save_dir': 'checkpoints_results/',
        'model_old': False,
        'training_mode': 'hard',
        'yolo_model_path': 'checkpoints/best_seg_YOLO.pt' ,
        'temperature': 1.5,
        'batch_size': 1,
        'num_workers': 12,  
    }
    
    evaluator = DAMF_Evaluator_WMask(config)
    results = evaluator.run()
    
    print("\n Evaluation completed!")
    print(f"Final Accuracy: {results['accuracy']:.2f}%")
    print(f"Mean ADD: {results['mean_add_cm']:.2f} cm")


    """
    config = {
        # Percorsi
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",  
        'split_val': "data/autosplit_val_ALL.txt",  
        'model_dir': 'checkpoints/', 
        'save_dir': 'checkpoints_results/',
        'model_old': False,
        'training_mode': 'hard',
        'yolo_model_path': 'checkpoints/bestyolov11.pt' ,
        'temperature': 1.5,
        # Parametri
        'batch_size': 64,
        'num_workers': 12,  # 0 su Mac, 4-8 su Linux
    }

    evaluator = DAMF_Evaluator_WMask(config)
    conf_results = evaluator.analyze_confidence_head(num_batches=32, save_plots=True)
    
    print("\n" + "="*70)
    print("STEP 1: Analyzing confidence head...")
    print("="*70)    
    # Interpreta risultati
    if conf_results['is_working']:
        print("✅ La confidence head sta funzionando bene!")
    else:
        print("⚠️  La confidence head potrebbe non essere utile.")
        print("   Considera di aggiungere la regolarizzazione del paper.")

    """

    """
    config = {
        # Percorsi
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",  
        'split_val': "data/autosplit_val_ALL.txt",  
        'model_dir': 'checkpoints/', 
        'save_dir': 'checkpoints_results/',
        'model_old': False,
        'training_mode': 'easy',
        'yolo_model_path': 'checkpoints/best_seg_YOLO.pt' ,
        'temperature': 2.0,
        # Parametri
        'batch_size': 32,
        'num_workers': 12,  # 0 su Mac, 4-8 su Linux
    }
    
    # Crea directory output
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Esegui valutazione
    evaluator = DAMF_Evaluator_WMask(config)
    results = evaluator.run()
    
    print("\n✅ Evaluation completed!")
    print(f"Final Accuracy: {results['accuracy']:.2f}%")
    print(f"Mean ADD: {results['mean_add_cm']:.2f} cm")
    """

    """
    config = {
        'dataset_root': '/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed',
        'split_train': 'data/autosplit_train_ALL.txt',
        'split_val': 'data/autosplit_val_ALL.txt',

        # SALVA IN UNA CARTELLA DIVERSA per non sovrascrivere il best model precedente!
        'save_dir': 'checkpoints/',
        'training_mode' : 'easy',
        'lr' : 1e-4,

        # === LOSS ESTREMA (Focus Traslazione) ===
        'use_weighted_loss': False,
        'rot_weight': 0.1,      # Mantenimento (Freeze logico)
        'trans_weight': 3.0,    # SPINTA MASSIMA (era 3.0, osiamo 4.0)

        'T_0': 20,    # Un ciclo lungo e calmo
        'T_mult': 2,
        'eta_min': 1e-6, # Scende quasi a zero alla fine

        # === SETTINGS ===
        'batch_size': 32, # Rimaniamo a 32 per stabilità
        'epochs': 150,
        'early_stop_patience': 30,

        'temperature': 2.0, # Temperatura standard
        'num_points_mesh': 500,
    }

    trainer = DAMFTurboTrainerA100(config)
    trainer.run()
    """

    """
    config = {
        # Percorsi
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",  
        'split_val': "data/autosplit_val_ALL.txt",  
        'model_dir': 'checkpoints/', 
        'save_dir': 'checkpoints_results/',
        'model_old': False,
        'training_mode': 'hard',
        'yolo_model_path': 'checkpoints/final_best_seg_YOLO.pt' ,
        'temperature': 1.5,
        # Parametri
        'batch_size': 32,
        'num_workers': 12,  # 0 su Mac, 4-8 su Linux
    }
    
    # Crea directory output
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Esegui valutazione
    evaluator = DAMF_Evaluator_WMask(config)
    results = evaluator.run()
    
    print("\n✅ Evaluation completed!")
    print(f"Final Accuracy: {results['accuracy']:.2f}%")
    print(f"Mean ADD: {results['mean_add_cm']:.2f} cm")
    """
    """
    config = {
        'dataset_root': '/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed',
        'split_train': 'data/autosplit_train_ALL.txt',
        'split_val': 'data/autosplit_val_ALL.txt',
        'save_dir': 'checkpoints/',
        'training_mode': 'easy',

        # Training
        'epochs': 100,
        'batch_size': 32,
        'lr': 1e-4,

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

    """
    config = {
        # Percorsi
        'dataset_root': "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed",  # MODIFICA QUESTO
        'split_val': "data/autosplit_val_ALL.txt",  # MODIFICA QUESTO
        'model_dir': 'checkpoints/',  # Directory con best_DAMF.pth
        'save_dir': 'checkpoints_results/',
        'training_mode': 'easy',
        'yolo_model_path': 'checkpoints/best_seg_YOLO.pt',
        'model_old': False,
        'batch_size': 32,
        'num_workers': 12,  
        'temperature': 2,  # Stesso usato in training
    }

    # Esegui valutazione
    evaluator = DAMF_Evaluator_WYolo(config)
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
    """
