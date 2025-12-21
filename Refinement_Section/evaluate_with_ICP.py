import torch
import numpy as np
import cv2
import open3d as o3d
import os
import scipy.spatial.transform
from tqdm import tqdm
from torch.utils.data import DataLoader

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
            self.num_workers = 0 
            print(">>> Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            print(">>> Using CPU")

        # 1. CARICAMENTO MODELLO TRAINATO
        self.model = DenseFusion_Masked_DualAtt_Net(
            pretrained=False, 
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
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
        self.models_db = {}
        mesh_path = os.path.join(self.cfg['dataset_root'], 'models')
        obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        
        for oid in obj_ids:
            ply_path = os.path.join(mesh_path, f"obj_{oid:02d}.ply")
            if os.path.exists(ply_path):
                mesh = o3d.io.read_point_cloud(ply_path)
                # Downsample per velocità (opzionale, ma consigliato)
                # mesh = mesh.uniform_down_sample(every_k_points=5) 
                self.models_db[oid] = np.asarray(mesh.points) / 1000.0 # Converti in Metri
            else:
                print(f"[WARNING] Mesh {ply_path} not found!")


    def _run_icp(self, gt_points, pred_points):
        """
        Esegue ICP Point-to-Point tra nuvola predetta (Source) e target (GT/Depth).
        """
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(pred_points)
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(gt_points)
        
        # Parametri ICP
        threshold = 0.02 # Cerca corrispondenze entro 2cm
        trans_init = np.identity(4) # Partiamo dalla posa predetta (nessuno spostamento iniziale)
        
        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            return reg_p2p.transformation
        except Exception as e:
            # Fallback se Open3D fallisce (raro)
            return np.identity(4)

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
            
            # E. REFINEMENT CON ICP
            # Allineiamo la predizione (Source) alla GT (Target).
            # Nella realtà useresti la nuvola estratta dalla Depth Map reale come Target.
            # Qui usiamo i punti GT come proxy di una "Depth Map perfetta" per vedere il potenziale massimo.
            
            refined_transform = self._run_icp(gt_pts_world, pred_pts_world)
            
            # Applica la correzione ICP
            # Punti_Nuovi = T_icp * Punti_Vecchi_Homogenei
            ones = np.ones((pred_pts_world.shape[0], 1))
            pred_pts_homo = np.hstack((pred_pts_world, ones)) # [N, 4]
            pred_pts_refined = (refined_transform @ pred_pts_homo.T).T[:, :3] # [N, 3]
            
            err_icp = np.mean(np.linalg.norm(pred_pts_refined - gt_pts_world, axis=1))
            add_errors_icp.append(err_icp)

        # F. STATISTICHE FINALI
        mean_base = np.mean(add_errors_basic)
        mean_icp = np.mean(add_errors_icp)
        
        print("\n" + "="*40)
        print(f" FINAL EVALUATION RESULTS")
        print("="*40)
        print(f" Mean ADD Error (Baseline):  {mean_base*100:.2f} cm")
        print(f" Mean ADD Error (With ICP):  {mean_icp*100:.2f} cm")
        print(f" Improvement:                {(mean_base - mean_icp)*100:.2f} cm")
        print("="*40)
        
        return mean_base, mean_icp