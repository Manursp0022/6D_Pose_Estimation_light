class yolo_seg_trainer:
    def __init__(self,
            dataset_root, 
            model="yolo26s-seg.pt", # 1. Passa a YOLO26 (S è ottimo per Linemod)
            epochs=150,             # Linemod richiede più di 10 epoche per convergere bene
            patience=50,
            batch=32,
            freeze_layers = 0       # Per oggetti difficili come Ape, meglio non freezare troppo
        ):

    def train(self, config_path=None):
        model = YOLO(self.model)

        model.train(
            data=config_path,
            epochs=self.epochs,
            batch=self.batch, # Usa 64 o 128 su A100
            device=0,
            imgsz=640,
            #------
            freeze=0,
            pretrained=True,
            optimizer='MuSGD', 
            #-----
            mask_ratio=1,
            overlap_mask=False,
            # --- AGGRESSIVE AUGMENTATION ---
            degrees=15.0,    # Ruota l'immagine (+/- 15 gradi)
            translate=0.1,   # Sposta l'oggetto nel frame
            scale=0.5,       # Zoom in/out (molto utile per oggetti piccoli)
            shear=2.0,       # Distorsione prospettica leggera
            perspective=0.0, # Se 0.0, non cambia la prospettiva 3D (tienilo basso)
            flipud=0.5,      # Flip verticale (utile in robotica)
            fliplr=0.5,      # Flip orizzontale
            hsv_h=0.015,     # Cambia leggermente la tinta (luce)
            hsv_s=0.7,       # Cambia saturazione
            hsv_v=0.4,       # Cambia luminosità
            mosaic=1.0,      # Combina 4 immagini in una (fondamentale)
            mixup=0.1,       # Sovrappone due immagini (aiuta la generalizzazione)
            # -------------------------------
            close_mosaic=20, # Fondamentale: disabilita mosaic alla fine per pulire i bordi
            project="linemod-segmentation",
            name=f"YOLO26_A100_Ape_StrongAug",
            val=True,
            save=True
        )

if __name__ == "__main__":
    dataset_root = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed"
    config_path = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed\\linemod_yolo_config_standard.yaml"
    trainer = yolo_seg_trainer(dataset_root=dataset_root, epochs=50)
    #config_path = trainer.create_labels_and_config()
    trainer.train(config_path=config_path)