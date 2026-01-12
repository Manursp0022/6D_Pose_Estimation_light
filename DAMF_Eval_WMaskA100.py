import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import autocast
from torchvision import transforms
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData

# --- IMPORT MODELLI ---
# Assicurati che i percorsi di import siano corretti per il tuo progetto
from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
from utils.Posenet_utils.posenet_dataset_ALLMasked import LineModPoseDatasetMasked
from utils.Posenet_utils.utils_geometric import crop_square_resize

# --- OTTIMIZZAZIONE KERNEL ---
torch.backends.cudnn.benchmark = True 

class DAMF_Evaluator_WMask:
    
    def __init__(self, config):
        self.cfg = config
        self.device = self._get_device()
        print(f"🔧 Initializing DAMF Evaluator Turbo A100 on: {self.device}")

        # Configurazione YOLO
        self.YOLO_CONF = 0.5
        self.img_size = config.get('img_size', 224) # Dimensione crop
        self.img_w = 640
        self.img_h = 480
        
        # Mapping Classi
        self.LINEMOD_TO_YOLO = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 
            9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }
        
        # Diametri Oggetti (mm)
        self.DIAMETERS = {
            1: 102.09, 2: 247.50, 4: 203.40, 5: 167.36, 6: 172.49, 8: 152.89, 
            9: 121.42, 10: 246.87, 11: 102.37, 12: 240.17, 13: 192.98, 14: 197.50, 15: 124.35
        }
        self.sym_list = [10, 11] # Oggetti simmetrici (Eggbox, Glue)

        # Trasformazioni Input
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Parametri Camera (Normalizzati)
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # Normalizziamo dividendo cx/w, cy/h etc.
        # Esempio LineMOD standard (modifica se usi camera diversa)
        fx, fy, cx, cy = 572.4114, 573.57043, 325.2611, 242.04899
        self.cam_params_norm = torch.tensor([
            cx / self.img_w, cy / self.img_h, fx, fy 
        ], dtype=torch.float32).to(self.device)

        # Caricamento Modelli
        self.yolo_model = self._setup_yolo()
        self.model = self._setup_model()
        
        # Dataset Validation
        self.val_loader = self._setup_dataloader()
        
        # Caricamento Mesh 3D per calcolo ADD
        self.models_3d = self._load_3d_models()

        # --- STATISTICHE LATENZA ---
        self.latencies = {
            'yolo': [],
            'pose': []
        }

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_yolo(self):
        print("🧠 Loading YOLOv8-Seg...")
        # Usa il peso 'yolov8x-seg.pt' o il tuo custom path
        yolo_path = os.path.join(self.cfg['model_dir'], 'bestyolov11.pt') 
        if not os.path.exists(yolo_path):
            print(f"⚠️ YOLO weights not found at {yolo_path}, using standard yolov8n-seg.pt")
            return YOLO("yolov8n-seg.pt")
        return YOLO(yolo_path)

    def _setup_model(self):
        print("🧠 Loading Masked_DualAtt_Net model (A100 Optimized)...")
        # Inizializza architettura
        model = DenseFusion_Masked_DualAtt_NetVar(
            pretrained=False, 
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)

        # Carica Pesi
        weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVar_WOAttention_Hard.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Pose Model weights not found at: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Gestione checkpoint (intero vs state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Rimuovi prefissi di compilazione precedenti se presenti
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # --- A100 OTTIMIZZAZIONE: TORCH COMPILE ---
        if hasattr(torch, 'compile'):
            print("   🚀 Compiling Pose Model with torch.compile() for A100...")
            try:
                model = torch.compile(model, mode='reduce-overhead')
            except Exception as e:
                print(f"   ⚠️ Compile failed (ignoring): {e}")
        
        return model

    def _setup_dataloader(self):
        print("📂 Setting up Validation Loader...")
        val_ds = LineModPoseDataset_AltMasked( 
            self.cfg['dataset_root'], 
            mode='val'
        )
        # Num workers alto per saturare la banda
        return DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=False, 
                          num_workers=16, pin_memory=True)


    def _load_3d_models(self):
        print("📦 Loading 3D Meshes...")
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        loaded_models = {}
        for obj_id in self.DIAMETERS.keys():
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                v = ply['vertex']
                pts = np.stack([v['x'], v['y'], v['z']], axis=-1) / 1000.0
                loaded_models[obj_id] = torch.from_numpy(pts).float().to(self.device)
        return loaded_models

    def _load_depth_worker(self, path):
        """Helper per caricare depth in parallelo (IO Bound)"""
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def _process_yolo_batch(self, rgb_paths, depth_paths, class_ids):
        """
        Versione TURBO: Parallel I/O + Latency Measurement + Zero-Copy RGB
        """
        batch_size = len(rgb_paths)
        
        # 1. Caricamento Depth in Parallelo (mentre YOLO lavora)
        depth_futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for path in depth_paths:
                depth_futures.append(executor.submit(self._load_depth_worker, path))

        # --- MISURAZIONE LATENZA YOLO (Start) ---
        torch.cuda.synchronize()
        start_yolo = time.time()
        
        # 2. YOLO Inference (Batch)
        yolo_results = self.yolo_model(list(rgb_paths), conf=self.YOLO_CONF, verbose=False)
        
        torch.cuda.synchronize()
        end_yolo = time.time()
        self.latencies['yolo'].append((end_yolo - start_yolo) * 1000.0) # ms
        # --- MISURAZIONE LATENZA YOLO (End) ---

        rgb_list = []
        depth_list = []
        mask_list = []
        bbox_list = []
        valid_indices = []

        # 3. Assemblaggio Batch
        for i in range(batch_size):
            try:
                depth_img = depth_futures[i].result() # Wait for depth
            except Exception as e:
                continue

            obj_id = int(class_ids[i])
            target_yolo_class = self.LINEMOD_TO_YOLO.get(obj_id, -1)
            yolo_res = yolo_results[i]
            
            # Trova detection migliore
            best_idx = None
            best_conf = -1.0
            
            if len(yolo_res.boxes) > 0:
                cls_tensor = yolo_res.boxes.cls
                conf_tensor = yolo_res.boxes.conf
                
                # Check veloce classi
                mask_cls = (cls_tensor == target_yolo_class)
                if mask_cls.any():
                    # Prendi quella con confidenza maggiore
                    valid_confs = conf_tensor[mask_cls]
                    max_conf_val, max_conf_idx_local = torch.max(valid_confs, 0)
                    
                    # Dobbiamo ritrovare l'indice globale
                    # Un modo rapido senza impazzire con gli indici:
                    indices = torch.where(mask_cls)[0]
                    best_idx = indices[max_conf_idx_local].item()
            
            if best_idx is None:
                continue # Nessun oggetto trovato

            # Estrazione Dati
            # Usa orig_img di YOLO (BGR) -> Converti RGB
            rgb_img = cv2.cvtColor(yolo_res.orig_img, cv2.COLOR_BGR2RGB)
            
            box = yolo_res.boxes.xyxy[best_idx].cpu().numpy() # x1, y1, x2, y2
            bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]] # x, y, w, h
            
            # Maschera
            if yolo_res.masks is not None:
                mask_data = yolo_res.masks.data[best_idx].cpu().numpy()
                if mask_data.shape != (self.img_h, self.img_w):
                    mask_data = cv2.resize(mask_data, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_data > 0.5).astype(np.uint8) * 255
            else:
                # Fallback box mask
                mask_bin = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                mask_bin[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 255

            # Cropping & Resizing (CPU Bound)
            rgb_crop = crop_square_resize(rgb_img, bbox, self.img_size, is_depth=False)
            depth_crop = crop_square_resize(depth_img, bbox, self.img_size, is_depth=True)
            mask_crop = crop_square_resize(mask_bin, bbox, self.img_size, is_depth=True)
            mask_crop = (mask_crop > 127).astype(np.float32)

            # Tensori
            rgb_list.append(self.transform(rgb_crop))
            depth_list.append(torch.from_numpy(depth_crop).float().unsqueeze(0))
            mask_list.append(torch.from_numpy(mask_crop).float().unsqueeze(0))
            
            # BBox Norm per il modello [cx, cy, w, h] normalizzati
            x, y, w, h = bbox
            bbox_norm = torch.tensor([
                (x + w/2) / self.img_w,
                (y + h/2) / self.img_h,
                w / self.img_w,
                h / self.img_h
            ], dtype=torch.float32)
            bbox_list.append(bbox_norm)
            
            valid_indices.append(i)

        if not valid_indices:
            return None

        # Stack su GPU
        return (
            torch.stack(rgb_list).to(self.device),
            torch.stack(depth_list).to(self.device),
            torch.stack(mask_list).to(self.device),
            torch.stack(bbox_list).to(self.device),
            valid_indices
        )

    def evaluate(self):
        """Loop Principale di Valutazione"""
        print(f"🚀 Starting Evaluation on {len(self.val_loader)} batches...")
        
        # Metriche ADD
        metrics = {obj_id: {'add': [], 'count': 0} for obj_id in self.DIAMETERS.keys()}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Eval"):
                # Dati dal loader
                paths = batch['path'] # RGB paths
                depth_paths = batch['depth_path']
                class_ids = batch['class_id'].numpy()
                gt_r = batch['quaternion'].to(self.device)
                gt_t = batch['translation'].to(self.device)
                
                # --- PROCESSING YOLO ---
                processed_data = self._process_yolo_batch(paths, depth_paths, class_ids)
                
                if processed_data is None: 
                    continue
                
                rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices = processed_data
                
                # Prepara parametri camera batchati
                B = rgb_batch.shape[0]
                cam_params_batch = self.cam_params_norm.unsqueeze(0).repeat(B, 1)

                # --- MISURAZIONE LATENZA POSE MODEL (Start) ---
                torch.cuda.synchronize()
                start_pose = time.time()
                
                # Inferenza in Mixed Precision (A100 loves this)
                with autocast():
                    # return_dense=False -> Inferenza standard (Voto pesato)
                    pred_r, pred_t = self.model(
                        rgb_batch, depth_batch, bbox_batch, cam_params_batch, 
                        mask=mask_batch, return_dense=False
                    )
                
                torch.cuda.synchronize()
                end_pose = time.time()
                # Dividiamo per B perché il tempo è per l'intero batch
                self.latencies['pose'].append(((end_pose - start_pose) * 1000.0) / B)
                # --- MISURAZIONE LATENZA POSE MODEL (End) ---

                # --- CALCOLO ERRORI ADD ---
                # Filtra GT solo per gli indici validi (quelli trovati da YOLO)
                gt_r_valid = gt_r[valid_indices]
                gt_t_valid = gt_t[valid_indices]
                class_ids_valid = class_ids[valid_indices]

                for i in range(B):
                    oid = int(class_ids_valid[i])
                    
                    # Pose Predetta
                    pr = pred_r[i]
                    pt = pred_t[i]
                    # Pose GT
                    gr = gt_r_valid[i]
                    gt = gt_t_valid[i]
                    
                    # Calcola ADD
                    model_pts = self.models_3d[oid]
                    
                    # Converti Quat -> Matrix
                    # (Qui uso una funzione interna o scipy se preferisci, 
                    # ma per velocità teniamo tutto su torch)
                    pred_R = self._quat_to_mat(pr)
                    gt_R = self._quat_to_mat(gr)
                    
                    pred_pts = torch.mm(model_pts, pred_R.T) + pt
                    gt_pts = torch.mm(model_pts, gt_R.T) + gt
                    
                    if oid in self.sym_list:
                        # ADD-S
                        # cdist pesante, ma su A100 va bene per eval
                        dists = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0))
                        min_dists = torch.min(dists, dim=2)[0]
                        add_error = torch.mean(min_dists).item()
                    else:
                        # ADD
                        add_error = torch.mean(torch.norm(pred_pts - gt_pts, dim=1)).item()
                    
                    metrics[oid]['add'].append(add_error)
                    metrics[oid]['count'] += 1

        self._print_results(metrics)

    def _quat_to_mat(self, q):
        """Helper veloce quaternione -> matrice (senza batch)"""
        # q = [x, y, z, w] o [w, x, y, z] a seconda del training.
        # Assumo formato standard [x, y, z, w] dal tuo codice precedente
        # Se usavi R.from_quat di scipy ricorda che vuole [x,y,z,w]
        # Qui implementazione torch nativa
        r, i, j, k = q
        s = 1.0 / (q.dot(q))
        
        return torch.tensor([
            [1-2*s*(j*j+k*k), 2*s*(i*j-k*r),   2*s*(i*k+j*r)],
            [2*s*(i*j+k*r),   1-2*s*(i*i+k*k), 2*s*(j*k-i*r)],
            [2*s*(i*k-j*r),   2*s*(j*k+i*r),   1-2*s*(i*i+j*j)]
        ], device=self.device)

    def _print_results(self, metrics):
        print("\n" + "="*50)
        print("📊 FINAL RESULTS")
        print("="*50)
        
        total_add = []
        
        print(f"{'Object':<15} | {'Mean ADD (m)':<15} | {'< 2cm (%)':<10}")
        print("-" * 50)
        
        for oid, data in metrics.items():
            if len(data['add']) == 0: continue
            
            mean_add = np.mean(data['add'])
            # Percentuale sotto i 2cm (soglia comune)
            acc_2cm = np.mean(np.array(data['add']) < 0.02) * 100
            
            total_add.extend(data['add'])
            
            print(f"Obj {oid:<11} | {mean_add:.4f}          | {acc_2cm:.1f}%")

        if total_add:
            print("-" * 50)
            print(f"🌍 GLOBAL MEAN ADD: {np.mean(total_add):.4f} m")
        
        print("\n" + "="*50)
        print("⏱️  LATENCY STATISTICS (ms)")
        print("="*50)
        
        if self.latencies['yolo']:
            y_times = np.array(self.latencies['yolo'])
            print(f"YOLOv8 (Batch) : Avg {np.mean(y_times):.2f} ms | Min {np.min(y_times):.2f} ms | Max {np.max(y_times):.2f} ms")
        
        if self.latencies['pose']:
            p_times = np.array(self.latencies['pose'])
            print(f"PoseNet (Per Img): Avg {np.mean(p_times):.2f} ms | Min {np.min(p_times):.2f} ms | Max {np.max(p_times):.2f} ms")
        print("="*50 + "\n")