import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

# Import dei tuoi moduli
from models.DAMF_DNet import DAMF_Net
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.PoseEvaluator import PoseEvaluator 


class DAMF_Evaluator:
    
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

        # Setup
        self.val_loader = self._setup_data()
        self.models_3d = self._load_3d_models()
        self.model = self._setup_model()
        
        # Metric calculator (la camera intrinsics non serve per ADD, solo per PnP)
        self.metric_calculator = PoseEvaluator(np.eye(3))

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
        val_ds = LineModPoseDataset(
            self.cfg['split_val'], 
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
        
        model = DenseFusion_Masked_DualAtt_Net(
            pretrained=False,  # Non servono pesi ImageNet, carichiamo i tuoi
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        weights_path = os.path.join(self.cfg['model_dir'], 'best_DuallAtt_noDecoder.pth')

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
                'errors': []  # in cm
            } 
            for obj_id in self.DIAMETERS.keys()
        }

        all_trans_errors = []
        all_rot_errors_m = []
        all_rot_errors_deg = []
        
        # Statistiche globali
        total_correct = 0
        total_predictions = 0
        all_errors = []  # in cm

        # --- NUOVO: Liste per X, Y, Z ---
        all_tx_errors = []
        all_ty_errors = []
        all_tz_errors = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                
                # Estrai dati dal batch
                rgb_batch = batch['image'].to(self.device)
                depth_batch = batch['depth'].to(self.device)
                mask_batch = batch['mask'].to(self.device)
                cam_params = batch['cam_params'].to(self.device, non_blocking=True)
                bb_info = batch['bbox_norm'].to(self.device, non_blocking=True)
                gt_translation = batch['translation'].cpu().numpy()  # [B, 3]
                gt_quats = batch['quaternion']  # [B, 4]
                class_ids = batch['class_id'].numpy()  # [B]
                
                # 1. INFERENCE del modello
                pred_quats, pred_trans = self.model(rgb_batch, depth_batch, bb_info, cam_params, mask_batch)
                
                # 2. Converti quaternioni in matrici di rotazione
                pred_R = self._quaternion_to_matrix(pred_quats)  # [B, 3, 3]
                gt_R = self._quaternion_to_matrix(gt_quats)  # [B, 3, 3]
                
                pred_t = pred_trans.cpu().numpy()  # [B, 3]
                
                batch_size = rgb_batch.shape[0]
                
                # 3. Calcola ADD per ogni sample nel batch
                for i in range(batch_size):
                    obj_id = int(class_ids[i])
                    
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

                    # --- NUOVO: Append errori assi (convertiti in cm) ---
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
                           total_predictions, class_stats) # <--- Fix class_stats        
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
                    mean_tx, mean_ty, mean_tz, # <--- Nuovi parametri
                    total, class_stats):       # <--- Fix parametro mancante
        
        print("\n" + "="*60)
        print("FINAL REPORT (Detailed Breakdown)")
        print("="*60)
        print(f" Samples Evaluated: {total}")
        print("-" * 40)
        print(f" TOTAL ACCURACY:    {accuracy:.2f} %")
        print(f" COMBINED ADD Error:{mean_add:.2f} cm")
        print("-" * 40)
        print(" ERROR BREAKDOWN:")
        print(f" -> Translation (Total): {mean_trans:.2f} cm")
        # --- NUOVO: Stampa dettagliata X, Y, Z ---
        print(f"    |-> Err X: {mean_tx:.2f} cm")
        print(f"    |-> Err Y: {mean_ty:.2f} cm")
        print(f"    |-> Err Z: {mean_tz:.2f} cm (Depth)")
        # -----------------------------------------
        print(f" -> Rotation (Mesh):     {mean_rot_cm:.2f} cm")
        print(f" -> Rotation (Angle):    {mean_rot_deg:.2f} deg")
        print("="*60)
        
        # Loop per classe (Fix name error)
        print(f"{'OBJECT':<15} {'COUNT':<10} {'ACCURACY':<10} {'MEAN ERR (cm)':<15}")
        print("-" * 60)
        for obj_id in sorted(list(class_stats.keys())):
            stats = class_stats[obj_id]
            if stats['total'] == 0:
                continue
                
            obj_name = self.OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
            acc = (stats['correct'] / stats['total']) * 100.0
            avg_err = np.mean(stats['errors']) * 100
            
            print(f"{obj_name:<15} {stats['total']:<10} {acc:>6.2f}%      {avg_err:>8.2f}")
        
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
