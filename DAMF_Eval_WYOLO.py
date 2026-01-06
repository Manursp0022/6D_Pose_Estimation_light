import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
import cv2
from torchvision import transforms
from ultralytics import YOLO
#from models.DFMasked_DualAtt_NetVar_WRefiner import DenseFusion_Masked_DualAtt_NetVarWRef
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net
from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
from models.DFMasked_DualAtt_NetVarGlobal import DenseFusion_Masked_DualAtt_NetVarGlobal
from models.DFMasked_DualAtt_NetVar_Weighted_WRefiner import DenseFusion_Masked_DualAtt_NetVarWRef

from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.PoseEvaluator import PoseEvaluator 
from utils.Posenet_utils.posenet_dataset_AltMasked import LineModPoseDataset_AltMasked
from utils.Posenet_utils.posenet_dataset_ALLMasked import LineModPoseDatasetMasked
from utils.Posenet_utils.utils_geometric import crop_square_resize

class DAMF_Evaluator_WYolo:
    
    def __init__(self, config):
        self.cfg = config
        self.device = self._get_device()
        print(f"🔧 Initializing DAMF Evaluator on: {self.device}")

        # Diametri oggetti LineMOD (mm)
        self.DIAMETERS = {
            1: 102.09,   # Ape
            2: 247.50,   # Benchvise
            4: 172.49,   # Cam
            5: 201.40,   # Can
            6: 154.54,   # Cat
            8: 261.47,   # Driller
            9: 108.99,   # Duck
            10: 164.62,  # Eggbox (symmetric)
            11: 175.88,  # Glue (symmetric)
            12: 145.54,  # Holepuncher
            13: 278.07,  # Iron
            14: 282.60,  # Lamp
            15: 212.35   # Phone
        }
        
        # Nomi oggetti per i plot
        self.OBJ_NAMES = {
            1: "Ape", 2: "Benchvise", 4: "Cam", 5: "Can", 6: "Cat", 
            8: "Driller", 9: "Duck", 10: "Eggbox", 11: "Glue", 
            12: "Holepuncher", 13: "Iron", 14: "Lamp", 15: "Phone"
        }

        self.LINEMOD_TO_YOLO = {
            1 : 0,   # Ape
            2 : 1,   # Benchvise
            4 : 2,   # Cam
            5 : 3,   # Can
            6 : 4,   # Cat
            8 : 5,   # Driller
            9 : 6,   # Duck
            10 : 7,  # Eggbox
            11 : 8,  # Glue
            12 : 9,  # Holepuncher
            13 : 10, # Iron
            14 : 11, # Lamp
            15 : 12  # Phone
        }

        # Camera intrinsics FISSE (LineMOD)
        self.cam_params_norm = torch.tensor([
            572.4114 / 640,   # fx_norm
            573.57043 / 480,  # fy_norm
            325.2611 / 640,   # cx_norm
            242.04899 / 480   # cy_norm
        ], dtype=torch.float32)

        self.img_h, self.img_w = 480, 640
        self.img_size = 224

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Setup
        self.yolo_model = self._setup_yolo()
        self.val_loader = self._setup_data()
        self.models_3d = self._load_3d_models()
        self.model = self._setup_model()
        
        # Metric calculator (la camera intrinsics non serve per ADD, solo per PnP)
        self.metric_calculator = PoseEvaluator(np.eye(3))

        self.YOLO_CONF = 0.5

    def _setup_yolo(self):
        """Carica il modello YOLO per segmentazione."""
        print("🔍 Loading yolov8n-seg:  Segmentation Model...")
        
        yolo_path = self.cfg['yolo_model_path']
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"❌ YOLO model not found at: {yolo_path}")
        
        model = YOLO(yolo_path)
        print(f"   ✓ Loaded YOLO from: {yolo_path}")
        return model

    def _get_device(self):
        """Selezione automatica del device migliore disponibile."""
        if torch.backends.mps.is_available():
            print("✅ Using Apple MPS acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("✅ Using CUDA")
            return torch.device("cuda")
        else:
            print("⚠️  Using CPU (slower)")
            return torch.device("cpu")

    def _setup_data(self):
        """Carica il dataset di validazione."""
        print("📦 Loading Validation Dataset...")
        if self.cfg['model_old'] :
            val_ds = LineModPoseDataset(
                self.cfg['split_val'], 
                self.cfg['dataset_root'], 
                mode='val'
            )
        else:
            if self.cfg['training_mode'] == "easy" : 
                val_ds = LineModPoseDatasetMasked(
                    self.cfg['split_val'], 
                    self.cfg['dataset_root'], 
                    mode='val'
                )
            elif self.cfg['training_mode'] == "hard":
                val_ds = LineModPoseDataset_AltMasked( 
                    self.cfg['dataset_root'], 
                    mode='val'
                )
        
        # num_workers=0 è più sicuro su Mac, 2-4 su Linux/Windows
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=False, 
            num_workers=self.cfg.get('num_workers', 12),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"   ✓ Loaded {len(val_ds)} validation samples")
        return val_loader
    

    def _setup_model(self):
        """Carica il modello DAMF_Net con i pesi addestrati."""
        print("🧠 Loading Masked_DualAtt_Net model...")
        
        model = DenseFusion_Masked_DualAtt_NetVar(
            pretrained=False,  # Non servono pesi ImageNet, carichiamo i tuoi
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAttNet_Hard1cm.pth')
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVar_Dropout.pth')
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVarRefinerHard.pth')
        weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVar_WOAttention.pth')
        """
        model = DAMF_Net(
            pretrained=False,  # Non servono pesi ImageNet, carichiamo i tuoi
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        weights_path = os.path.join(self.cfg['model_dir'], 'best_DAMF.pth')
        """

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Weights not found at: {weights_path}")
        
        # Carica il checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Controlla se è un checkpoint con metadati o solo state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint con metadati (epoch, optimizer, etc.)
            state_dict = checkpoint['model_state_dict']
            print(f"   ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_loss' in checkpoint:
                print(f"   ✓ Best validation loss: {checkpoint['val_loss']:.4f}")
        else:
            # Solo state_dict
            state_dict = checkpoint
        
        # Rimuovi prefisso '_orig_mod.' se presente (torch.compile)
        state_dict = self._remove_compile_prefix(state_dict)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"   ✓ Loaded weights from: {weights_path}")
        return model
    
    def _load_3d_models(self):
        """Carica i modelli 3D (.ply) per il calcolo ADD."""
        print("📐 Loading 3D Models (.ply)...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        
        for obj_id in self.DIAMETERS.keys():
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            
            if not os.path.exists(path):
                print(f"   ⚠️  Missing: obj_{obj_id:02d}.ply")
                continue
                
            ply = PlyData.read(path)
            vertex = ply['vertex']
            
            # Converti da mm a metri
            pts_mm = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
            models_3d[obj_id] = pts_mm / 1000.0  # mm -> m
        
        print(f"   ✓ Loaded {len(models_3d)} 3D models")
        return models_3d

    def _remove_compile_prefix(self, state_dict):
        """
        Rimuove il prefisso '_orig_mod.' dai pesi salvati con torch.compile().
        
        Args:
            state_dict: State dict con o senza prefisso
            
        Returns:
            State dict pulito senza prefisso
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            # Rimuovi il prefisso '_orig_mod.' se presente
            new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        return new_state_dict

    def _quaternion_to_matrix(self, quats):
        """
        Converte quaternioni in matrici di rotazione.
        
        Args:
            quats: [B, 4] tensor o numpy array in formato [w, x, y, z]
            
        Returns:
            rot_matrices: [B, 3, 3] numpy array
        """
        if isinstance(quats, torch.Tensor):
            quats = quats.cpu().numpy()
        
        # Il tuo modello output [w, x, y, z], scipy vuole [x, y, z, w]
        quats_scipy = np.concatenate([quats[:, 1:], quats[:, 0:1]], axis=1)
        
        return R.from_quat(quats_scipy).as_matrix()
        
    def _process_yolo_batch(self, rgb_paths, depth_paths, class_ids):
            """
            Processa un batch di immagini con YOLO.
            Ritorna rgb, depth, mask, bbox tutti croppati e pronti, più gli indici validi.
            
            Returns:
                rgb_batch, depth_batch, mask_batch, bbox_batch: tensori solo per sample validi
                valid_indices: lista degli indici originali del batch che sono stati processati
            """
            batch_size = len(rgb_paths)
            
            rgb_list = []
            depth_list = []
            mask_list = []
            bbox_list = []
            valid_indices = []  # Traccia quali sample sono validi
            
            # YOLO inference su tutte le immagini del batch
            yolo_results = self.yolo_model(list(rgb_paths), conf=self.YOLO_CONF, verbose=False)
            
            for i in range(batch_size):
                rgb_path = rgb_paths[i]
                depth_path = depth_paths[i]
                obj_id = int(class_ids[i])
                target_yolo_class = self.LINEMOD_TO_YOLO[obj_id]
                
                # Prendi risultato YOLO per questa immagine
                yolo_res = yolo_results[i]
                
                # Trova la detection per la classe target (prendi quella con conf più alta)
                boxes = yolo_res.boxes
                best_idx = None
                best_conf = 0.0
                for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
                    if int(cls) == target_yolo_class and float(conf) > best_conf:
                        best_conf = float(conf)
                        best_idx = j
                
                # CHECK: YOLO ha trovato l'oggetto?
                if best_idx is None:
                    # Skip questo sample
                    continue
                
                # Carica RGB
                rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                
                # Carica Depth originale
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                
                # Estrai bbox [x_min, y_min, w, h]
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                
                # Estrai maschera
                mask_data = yolo_res.masks.data[best_idx].cpu().numpy()
                if mask_data.shape != (self.img_h, self.img_w):
                    mask_data = cv2.resize(mask_data, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                # Crop & Resize
                rgb_crop = crop_square_resize(rgb_img, bbox, self.img_size, is_depth=False)
                depth_crop = crop_square_resize(depth_img, bbox, self.img_size, is_depth=True)
                mask_crop = crop_square_resize(mask, bbox, self.img_size, is_depth=True)
                mask_crop = (mask_crop > 127).astype(np.float32)
                
                # To tensors
                rgb_list.append(self.transform(rgb_crop))
                depth_list.append(torch.from_numpy(depth_crop).float().unsqueeze(0))
                mask_list.append(torch.from_numpy(mask_crop).float().unsqueeze(0))
                
                # Bbox normalizzata
                x, y, w, h = bbox
                bbox_norm = torch.tensor([
                    (x + w/2) / self.img_w,
                    (y + h/2) / self.img_h,
                    w / self.img_w,
                    h / self.img_h
                ], dtype=torch.float32)
                bbox_list.append(bbox_norm)
                
                # Questo sample è valido
                valid_indices.append(i)
            
            # Se nessun sample valido, ritorna None
            if len(valid_indices) == 0:
                return None, None, None, None, []
            
            # Stack tutto
            rgb_batch = torch.stack(rgb_list)      # [N, 3, 224, 224]
            depth_batch = torch.stack(depth_list)  # [N, 1, 224, 224]
            mask_batch = torch.stack(mask_list)    # [N, 1, 224, 224]
            bbox_batch = torch.stack(bbox_list)    # [N, 4]
            
            return rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices

    def run(self):
        """
        Esegue la valutazione completa sul validation set.
        Calcola ADD e ADD-S con threshold 0.1*diameter.
        """
        print("\n" + "="*70)
        print("🚀 STARTING EVALUATION - DAMF_Net")
        print("="*70)
        
        # Statistiche per classe
        class_stats = {
            obj_id: {
                'correct': 0,
                'total': 0,
                'errors': [] , 
                'yolo_missed': 0 
            } 
            for obj_id in self.DIAMETERS.keys()
        }

        all_trans_errors = []
        all_rot_errors_m = []
        all_rot_errors_deg = []
        
        # Statistiche globali
        total_correct = 0
        total_predictions = 0
        total_yolo_missed = 0
        all_errors = []  # in cm

        # --- NUOVO: Liste per X, Y, Z ---
        all_tx_errors = []
        all_ty_errors = []
        all_tz_errors = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                
                # Estrai dati dal batch
                #rgb_batch = batch['image'].to(self.device)
                #mask_batch = batch['mask'].to(self.device)

                paths = batch['path']
                depth_paths = batch['depth_path']
                depth_batch = batch['depth']#.to(self.device)
                cam_params = batch['cam_params']#.to(self.device, non_blocking=True)
                #bb_info = batch['bbox_norm'].to(self.device, non_blocking=True)
                gt_translation = batch['translation'].cpu().numpy()  # [B, 3]
                gt_quats = batch['quaternion']  # [B, 4]
                class_ids = batch['class_id'].numpy()  # [B]

                # Processa con YOLO (ritorna solo sample validi)
                rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices = self._process_yolo_batch(
                    paths, depth_paths, class_ids
                )

                # Conta YOLO misses per questo batch
                batch_size = len(paths)
                num_valid = len(valid_indices)
                num_missed = batch_size - num_valid
                total_yolo_missed += num_missed
                
                # Aggiorna yolo_missed per classe
                for i in range(batch_size):
                    if i not in valid_indices:
                        obj_id = int(class_ids[i])
                        if obj_id in class_stats:
                            class_stats[obj_id]['yolo_missed'] += 1
                
                # Se nessun sample valido, skip batch
                if num_valid == 0:
                    continue

                rgb_batch = rgb_batch.to(self.device)
                depth_batch = depth_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                bbox_batch = bbox_batch.to(self.device)

                B = rgb_batch.shape[0]
                cam_params = self.cam_params_norm.unsqueeze(0).repeat(B, 1).to(self.device)

                
                # 1. INFERENCE del modello
                pred_quats, pred_trans = self.model(rgb_batch, depth_batch, bbox_batch, cam_params, mask_batch)
                
                # 2. Converti quaternioni in matrici di rotazione
                pred_R = self._quaternion_to_matrix(pred_quats)  # [B, 3, 3]

                valid_gt_quats = gt_quats[valid_indices]
                valid_gt_trans = gt_translation[valid_indices]
                valid_class_ids = class_ids[valid_indices]

                gt_R = self._quaternion_to_matrix(valid_gt_quats)  # [B, 3, 3]
                pred_t = pred_trans.cpu().numpy()  # [B, 3]
                
                batch_size = rgb_batch.shape[0]
                
                # 3. Calcola ADD per ogni sample nel batch
                for i in range(B):
                    obj_id = int(valid_class_ids[i])
                    
                    # Salta se non abbiamo il modello 3D
                    if obj_id not in self.models_3d:
                        print(f"   ⚠️  Skipping obj_id={obj_id} (no 3D model)")
                        continue
                    
                    # Prendi i punti 3D dell'oggetto
                    pts_3d = self.models_3d[obj_id]  # [N, 3] in metri
                    
                    # Calcola ADD (o ADD-S per oggetti simmetrici)
                    add_error_m = self.metric_calculator.calculate_metric(
                        pred_R[i], pred_t[i],
                        gt_R[i], gt_translation[i],
                        pts_3d, obj_id
                    )
                    t_err, r_err_m, r_err_deg, tx, ty, tz = self.metric_calculator.calculate_separated_metrics(
                        pred_R[i], pred_t[i], 
                        gt_R[i], gt_translation[i], 
                        pts_3d, obj_id
                    )

                    all_trans_errors.append(t_err * 100.0)      # Convert to cm
                    all_rot_errors_m.append(r_err_m * 100.0)    # Convert to cm
                    all_rot_errors_deg.append(r_err_deg)

                    # Append errori assi (convertiti in cm) ---
                    all_tx_errors.append(tx * 100.0)
                    all_ty_errors.append(ty * 100.0)
                    all_tz_errors.append(tz * 100.0)
                    
                    # Converti errore in cm
                    #add_error_cm = add_error_m * 100.0
                    
                    # Threshold: 10% del diametro
                    diameter_m = self.DIAMETERS[obj_id] / 1000.0  # mm -> m
                    threshold_m = 0.1 * diameter_m
                    
                    is_correct = self.metric_calculator.is_pose_correct(add_error_m, threshold_m)
                    
                    # Aggiorna statistiche globali
                    if is_correct:
                        total_correct += 1
                    total_predictions += 1
                    all_errors.append(add_error_m * 100.0)
                    
                    # Aggiorna statistiche per classe
                    class_stats[obj_id]['total'] += 1
                    class_stats[obj_id]['errors'].append(add_error_m)
                    if is_correct:
                        class_stats[obj_id]['correct'] += 1
        
        # 4. Calcola metriche finali
        accuracy = (total_correct / total_predictions * 100.0) if total_predictions > 0 else 0.0
        mean_add_cm = np.mean(all_errors) if len(all_errors) > 0 else 0.0
        mean_add_cm = np.mean(all_errors) if len(all_errors) > 0 else 0.0
        median_add_cm = np.median(all_errors) if len(all_errors) > 0 else 0.0

        mean_trans = np.mean(all_trans_errors) if len(all_trans_errors) > 0 else 0.0
        mean_rot_cm = np.mean(all_rot_errors_m) if len(all_rot_errors_m) > 0 else 0.0
        mean_rot_deg = np.mean(all_rot_errors_deg) if len(all_rot_errors_deg) > 0 else 0.0

        mean_tx = np.mean(all_tx_errors) if len(all_tx_errors) > 0 else 0.0
        mean_ty = np.mean(all_ty_errors) if len(all_ty_errors) > 0 else 0.0
        mean_tz = np.mean(all_tz_errors) if len(all_tz_errors) > 0 else 0.0
        
        # 5. Stampa report
        self._print_report(accuracy, mean_add_cm, mean_trans, mean_rot_cm, mean_rot_deg, 
                           mean_tx, mean_ty, mean_tz,  # <--- Nuovi argomenti
                           total_predictions,total_yolo_missed, class_stats) # <--- Fix class_stats        
        # 6. Genera plot
        self._plot_results(class_stats) 
        
        return {
            'accuracy': accuracy,
            'mean_add_cm': mean_add_cm,
            'median_add_cm': median_add_cm,
            'mean_trans': mean_trans,
            'mean_rot_cm': mean_rot_cm,
            'mean_rot_deg': mean_rot_deg,
            'class_stats': class_stats
        }
    def _print_report(self, accuracy, mean_add, mean_trans, mean_rot_cm, mean_rot_deg, 
                        mean_tx, mean_ty, mean_tz,
                        total, total_yolo_missed, class_stats):
            
            print("\n" + "="*60)
            print("FINAL REPORT (Detailed Breakdown)")
            print("="*60)
            print(f" Samples Evaluated: {total}")
            print(f" YOLO Missed: {total_yolo_missed}")
            total_samples = total + total_yolo_missed
            yolo_det_rate = (total / total_samples * 100.0) if total_samples > 0 else 0.0
            print(f" YOLO Detection Rate: {yolo_det_rate:.2f}%")
            print("-" * 40)
            print(f" TOTAL ACCURACY:    {accuracy:.2f} %")
            print(f" COMBINED ADD Error:{mean_add:.2f} cm")
            print("-" * 40)
            print(" ERROR BREAKDOWN:")
            print(f" -> Translation (Total): {mean_trans:.2f} cm")
            print(f"    |-> Err X: {mean_tx:.2f} cm")
            print(f"    |-> Err Y: {mean_ty:.2f} cm")
            print(f"    |-> Err Z: {mean_tz:.2f} cm (Depth)")
            print(f" -> Rotation (Mesh):     {mean_rot_cm:.2f} cm")
            print(f" -> Rotation (Angle):    {mean_rot_deg:.2f} deg")
            print("="*60)
            
            # Loop per classe con YOLO missed
            print(f"{'OBJECT':<12} {'COUNT':<8} {'MISSED':<8} {'ACCURACY':<10} {'MEAN ERR (cm)':<12}")
            print("-" * 60)
            for obj_id in sorted(list(class_stats.keys())):
                stats = class_stats[obj_id]
                if stats['total'] == 0 and stats.get('yolo_missed', 0) == 0:
                    continue
                    
                obj_name = self.OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
                acc = (stats['correct'] / stats['total']) * 100.0 if stats['total'] > 0 else 0.0
                avg_err = np.mean(stats['errors']) * 100 if len(stats['errors']) > 0 else 0.0
                missed = stats.get('yolo_missed', 0)
                
                print(f"{obj_name:<12} {stats['total']:<8} {missed:<8} {acc:>6.2f}%      {avg_err:>8.2f}")
            
            print("="*70 + "\n")

    def _plot_results(self, class_stats):
        """Genera i plot dei risultati per classe."""
        print("📈 Generating plots...")
        
        # Prepara dati per i plot
        obj_ids = sorted([oid for oid in class_stats.keys() if class_stats[oid]['total'] > 0])
        labels = [self.OBJ_NAMES.get(oid, str(oid)) for oid in obj_ids]
        
        accuracies = []
        mean_errors = []
        thresholds_cm = []
        
        for oid in obj_ids:
            stats = class_stats[oid]
            acc = (stats['correct'] / stats['total']) * 100.0 if stats['total'] > 0 else 0.0
            avg_err = np.mean(stats['errors']) if len(stats['errors']) > 0 else 0.0
            threshold_cm = (self.DIAMETERS[oid] / 1000.0 * 0.1) * 100.0  # 10% diameter in cm
            
            accuracies.append(acc)
            mean_errors.append(avg_err)
            thresholds_cm.append(threshold_cm)
        
        # Crea figura con 2 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # --- PLOT 1: Accuracy per classe ---
        bars = ax1.bar(labels, accuracies, color='#4CAF50', edgecolor='black', alpha=0.8)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Per-Class Accuracy (ADD < 0.1*diameter)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
        ax1.legend()
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Ruota labels se necessario
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # --- PLOT 2: ADD Error vs Threshold ---
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, mean_errors, width, label='Mean ADD Error', 
                        color='#FF5722', alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x + width/2, thresholds_cm, width, label='Threshold (10% Diam)', 
                        color='#2196F3', alpha=0.8, edgecolor='black')
        
        ax2.set_ylabel('Distance (cm)', fontsize=12, fontweight='bold')
        ax2.set_title('Mean ADD Error vs Acceptance Threshold', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Aggiungi valori sopra le barre
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Salva
        save_path = os.path.join(self.cfg['save_dir'], 'DAMF_evaluation_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Plot saved to: {save_path}")
        
        plt.show()
