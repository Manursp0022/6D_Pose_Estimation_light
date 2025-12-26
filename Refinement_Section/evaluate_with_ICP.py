import torch
import numpy as np
import cv2
#import open3d as o3d
import os
import scipy.spatial.transform
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData
# Import dei tuoi moduli custom
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset

class ICPEvaluator:
    def __init__(self, config):
        self.cfg = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.num_workers = 8
            print(">>> Using CUDA (NVIDIA)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.num_workers = 4 
            print(">>> Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            print(">>> Using CPU")

        # 1. CARICAMENTO MODELLO TRAINATO
        self.model = DenseFusion_Masked_DualAtt_Net(
            pretrained=False, 
            temperature=self.cfg['temperature']
        ).to(self.device)

        self.DRS = {
            1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54, 8: 261.47,
            9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54, 13: 278.07, 14: 282.60, 15: 212.35
        }
        
        ckpt_path = os.path.join(self.cfg['save_dir'], 'best_turbo_model_A100.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
            
        print(f"Loading Checkpoint: {ckpt_path}")
        
        # --- NUOVO CODICE CORRETTO ---
        state_dict = torch.load(ckpt_path, map_location=self.device)
        
        # Dizionario pulito per rimuovere il prefisso "_orig_mod." Dato dal train con torch.compile
        new_state_dict = {}
        for k, v in state_dict.items():
            # Rimuove "_orig_mod." se presente all'inizio della chiave
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        # -----------------------------
        
        self.model.eval()
        # 2. CARICAMENTO DATASET DI VALIDAZIONE
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        # Batch size 1 è obbligatorio per ICP (processiamo una nuvola alla volta)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # 3. CARICAMENTO MESH 3D (DATABASE)
        # Le carichiamo in memoria RAM come array numpy per Open3D
        print("Loading 3D Meshes for ICP...")
        self.models_db = self._load_3d_models()

    def _load_3d_models(self):
        print(" Loading 3D Models (.ply)...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        for obj_id in self.DRS.keys():
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                vertex = ply['vertex']
                pts_mm = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
                models_3d[obj_id] = pts_mm / 1000.0 
        return models_3d

    def evaluate(self):
        add_errors_basic = []
        add_errors_icp = []
        
        print(f"Starting Evaluation on {len(self.val_loader)} samples...")
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # A. PREPARAZIONE DATI
            img = batch['image'].to(self.device)
            depth = batch['depth'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Parametri CPU numpy per geometria
            gt_t = batch['translation'].numpy()[0]
            gt_q = batch['quaternion'].numpy()[0]
            obj_id = int(batch['class_id'].item())
            
            # Se l'oggetto non ha una mesh caricata, saltiamo
            if obj_id not in self.models_db:
                continue

            # B. INFERENZA RETE (Stima "Coarse")
            with torch.no_grad():
                pred_r, pred_t = self.model(img, depth, mask)
            
            pred_t = pred_t.cpu().numpy()[0]
            pred_q = pred_r.cpu().numpy()[0]
            
            # Ricostruzione Matrice 4x4 Predetta
            # Nota: scipy usa (x,y,z,w), pytorch spesso (w,x,y,z). 
            # Nel tuo utils_geometric.py avevi riordinato. Assicurati che combaci.
            # Qui assumo che la rete sputi quaternioni normalizzati.
            R_pred = scipy.spatial.transform.Rotation.from_quat([pred_q[1], pred_q[2], pred_q[3], pred_q[0]]).as_matrix()
            
            # C. GENERAZIONE NUVOLE DI PUNTI
            model_pts = self.models_db[obj_id] # Punti modello a riposo
            
            # 1. Nuvola Predetta (World Space) = R_pred * P + t_pred
            pred_pts_world = (R_pred @ model_pts.T).T + pred_t
            
            # 2. Nuvola GT (World Space) = R_gt * P + t_gt
            R_gt = scipy.spatial.transform.Rotation.from_quat([gt_q[1], gt_q[2], gt_q[3], gt_q[0]]).as_matrix()
            gt_pts_world = (R_gt @ model_pts.T).T + gt_t
            
            # D. CALCOLO ERRORE BASE (ADD)
            # Media delle distanze punto-punto (senza matching intelligente per ora, ADD semplice)
            # Per ADD-S (simmetrici) servirebbe KDTree, qui facciamo ADD standard asimmetrico.
            err_base = np.mean(np.linalg.norm(pred_pts_world - gt_pts_world, axis=1))
            add_errors_basic.append(err_base)

        
        # F. STATISTICHE FINALI
        mean_base = np.mean(add_errors_basic)
        #mean_icp = np.mean(add_errors_icp)
        
        print("\n" + "="*40)
        print(f" FINAL EVALUATION RESULTS")
        print("="*40)
        print(f" Mean ADD Error (Baseline):  {mean_base*100:.2f} cm")
        #print(f" Mean ADD Error (With ICP):  {mean_icp*100:.2f} cm")
        #print(f" Improvement:                {(mean_base - mean_icp)*100:.2f} cm")
        print("="*40)
        
        return mean_base