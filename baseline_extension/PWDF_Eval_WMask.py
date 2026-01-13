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
from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
from models.DFMasked_DualAtt_NetVarGlobal import DenseFusion_Masked_DualAtt_NetVarGlobal
import torch.nn.functional as F
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.PoseEvaluator import PoseEvaluator 
from utils.Posenet_utils.posenet_dataset_AltMasked import LineModPoseDataset_AltMasked
from utils.Posenet_utils.posenet_dataset_ALLMasked import LineModPoseDatasetMasked
from utils.Posenet_utils.utils_geometric import crop_square_resize

class DAMF_Evaluator_WMask:
    
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
        
        self.metric_calculator = PoseEvaluator(np.eye(3))
        self.YOLO_CONF = 0.5

    def _setup_yolo(self):
        """Carica il modello YOLO per segmentazione."""
        print("🔍 Loading yolov8n-seg:  Segmentation Model...")
        
        yolo_path = self.cfg['yolo_model_path']
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found at: {yolo_path}")
        
        model = YOLO(yolo_path)
        print(f" Loaded YOLO from: {yolo_path}")
        return model

    def _get_device(self):
        """Selezione automatica del device migliore disponibile."""
        if torch.backends.mps.is_available():
            print(" Using Apple MPS acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print(" Using CUDA")
            return torch.device("cuda")
        else:
            print("  Using CPU (slower)")
            return torch.device("cpu")

    def _setup_data(self):
        """Carica il dataset di validazione."""
        print(" Loading Validation Dataset...")
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
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=False, 
            num_workers=self.cfg.get('num_workers', 12),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f" Loaded {len(val_ds)} validation samples")
        return val_loader
    

    def _setup_model(self):
        print(" Loading model...")
        model = DenseFusion_Masked_DualAtt_NetVar(
            pretrained=False, 
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)

        """
        model = DenseFusion_Masked_DualAtt_NetVarGlobal(
            pretrained=False  # Non servono pesi ImageNet, carichiamo i tuoi
        ).to(self.device)
        """

        
        
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAttNet_Hard1cm.pth')
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVar_Dropout.pth')
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVarRefinerHard.pth')
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVar_WOAttention.pth')
        weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVar_WOAttention_Hard.pth')
        #weights_path = os.path.join(self.cfg['model_dir'], 'DenseFusion_Masked_DualAtt_NetVarGlobaleasy.pth')


        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Weights not found at: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Controlla se è un checkpoint con metadati o solo state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_loss' in checkpoint:
                print(f"   ✓ Best validation loss: {checkpoint['val_loss']:.4f}")
        else:
            state_dict = checkpoint
        
        # removing state dict if present
        state_dict = self._remove_compile_prefix(state_dict)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f" Loaded weights from: {weights_path}")
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

    def _calculate_iou(self, mask1, mask2):
        """
        Calcola Intersection over Union tra due maschere binarie (numpy arrays).
        """
        # Assicuriamoci che siano booleani o 0/1
        m1 = (mask1 > 0).astype(bool)
        m2 = (mask2 > 0).astype(bool)
        
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        if union == 0:
            return 0.0
        return (intersection / union) * 100.0
    
    def analyze_confidence_head(self, num_batches=30, save_plots=True):
        """
        Analizza la distribuzione dei pesi della confidence head.
        
        Questa analisi serve a capire se la confidence head sta effettivamente
        imparando a distinguere regioni informative (pesi concentrati) o se
        produce pesi quasi uniformi (non sta aiutando).
        
        Args:
            num_batches: Numero di batch da analizzare (default: 30)
            save_plots: Se True, salva i plot nella save_dir
            
        Returns:
            dict con statistiche e interpretazione
        """
        print("\n" + "="*70)
        print("🔍 CONFIDENCE HEAD ANALYSIS")
        print("="*70)
        
        all_weights = []
        all_max_weights = []
        all_entropies = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Analyzing confidence", total=num_batches)):
                if batch_idx >= num_batches:
                    break
                
                paths = batch['path']
                depth_paths = batch['depth_path']
                class_ids = batch['class_id'].numpy()
                
                # Process with YOLO
                result = self._process_yolo_batch_wIoU(paths, depth_paths, class_ids)
                
                # ============================================
                # FIX: Unpack correttamente tutti i 6 valori
                # ============================================
                rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices, batch_ious = result
                
                if len(valid_indices) == 0:
                    continue
                
                rgb_batch = rgb_batch.to(self.device)
                depth_batch = depth_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)  # Aggiungi anche la mask
                bbox_batch = bbox_batch.to(self.device)
                
                B = rgb_batch.shape[0]
                cam_params = self.cam_params_norm.unsqueeze(0).repeat(B, 1).to(self.device)
                
                # === EXTRACT CONFIDENCE WEIGHTS ===
                # Replica la forward del modello fino ai pesi
                
                # Forward fusion - PASSA ANCHE LA MASK!
                fused_feat, rgb_enh, depth_enh = self.model._forward_fusion(rgb_batch, depth_batch, mask_batch)
                
                # Prepara input per confidence head
                bb_spatial = bbox_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
                cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
                
                # Calcola confidence logits
                conf_input = torch.cat([fused_feat, rgb_enh, bb_spatial, cam_spatial], dim=1)
                conf_logits = self.model.conf_head(conf_input)  # [B, 1, 7, 7]
                conf_logits = conf_logits.view(B, 1, -1)  # [B, 1, 49]
                
                # Applica softmax con temperature
                weights = F.softmax(conf_logits / self.model.temperature, dim=2)  # [B, 1, 49]
                weights = weights.squeeze(1)  # [B, 49]
                
                # Salva per analisi
                all_weights.append(weights.cpu().numpy())
        
        # Concatena tutti i pesi
        all_weights = np.concatenate(all_weights, axis=0)  # [N, 49]
        
        # === CALCOLA STATISTICHE ===
        uniform = 1.0 / 49  # ~0.0204
        
        max_w = all_weights.max(axis=1).mean()
        min_w = all_weights.min(axis=1).mean()
        std_w = all_weights.std(axis=1).mean()
        
        # Entropia (misura di uniformità)
        # Max entropy = log(49) ≈ 3.89 per distribuzione uniforme
        entropies = -np.sum(all_weights * np.log(all_weights + 1e-10), axis=1)
        max_entropy = np.log(49)
        norm_entropy = (entropies / max_entropy).mean()
        
        # Pixel dominanti (peso > 2x uniforme)
        num_dominant = (all_weights > 2 * uniform).sum(axis=1).mean()
        
        # Effective number of pixels (exp(entropy))
        effective_n = np.exp(entropies).mean()
        
        # === STAMPA REPORT ===
        print(f"\n📊 STATISTICS ({len(all_weights)} samples analyzed):")
        print(f"   Uniform reference:     {uniform:.4f} (= 1/49)")
        print(f"   Max weight (avg):      {max_w:.4f}  {'✅' if max_w > 2*uniform else '⚠️'}")
        print(f"   Min weight (avg):      {min_w:.6f}")
        print(f"   Std weight (avg):      {std_w:.4f}")
        print(f"\n   Normalized entropy:    {norm_entropy:.3f}  ", end="")
        
        if norm_entropy > 0.95:
            print("⚠️  QUASI UNIFORME!")
        elif norm_entropy > 0.85:
            print("⚡ Moderatamente concentrata")
        else:
            print("✅ Ben concentrata!")
            
        print(f"   Effective #pixels:     {effective_n:.1f} / 49")
        print(f"   Dominant pixels (>2x): {num_dominant:.1f}")
        
        # === INTERPRETAZIONE ===
        print(f"\n💡 INTERPRETAZIONE:")
        if norm_entropy > 0.95:
            print("   ⚠️  La confidence head produce pesi QUASI UNIFORMI.")
            print("   → NON sta aiutando a selezionare regioni informative.")
            print("   → Il weighted pooling è praticamente un average pooling.")
            print("   → Considera di aggiungere la loss del paper DenseFusion:")
            print("      L = (1/N) * Σ(L_add * c - w*log(c))")
            print("   → Questo termine forza la rete a 'scommettere' su alcune regioni.")
        elif norm_entropy > 0.85:
            print("   ⚡ La confidence mostra una LEGGERA concentrazione.")
            print("   → Sta iniziando a differenziare, ma potrebbe migliorare.")
            print("   → La regolarizzazione potrebbe aiutare.")
        else:
            print("   ✅ La confidence head sta FUNZIONANDO!")
            print("   → I pesi sono concentrati su alcune regioni.")
            print("   → Il weighted pooling sta effettivamente selezionando.")
        
        print("="*70 + "\n")
        
        # === GENERA PLOT ===
        if save_plots:
            self._plot_confidence_analysis(all_weights, norm_entropy, uniform)
        
        return {
            'norm_entropy': norm_entropy,
            'max_weight': max_w,
            'min_weight': min_w,
            'std_weight': std_w,
            'effective_n': effective_n,
            'num_dominant': num_dominant,
            'all_weights': all_weights,
            'is_working': norm_entropy < 0.85
        }
    
    def _plot_confidence_analysis(self, all_weights, norm_entropy, uniform):
        """Genera e salva i plot dell'analisi confidence."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Confidence Head Analysis (Normalized Entropy: {norm_entropy:.3f})', 
                     fontsize=14, fontweight='bold')
        
        # 1. Istogramma dei pesi
        ax1 = axes[0, 0]
        ax1.hist(all_weights.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=uniform, color='red', linestyle='--', linewidth=2, label=f'Uniform (1/49 = {uniform:.4f})')
        ax1.set_xlabel('Weight Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of All Confidence Weights', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Heatmap media dei pesi 7x7
        ax2 = axes[0, 1]
        mean_weight_map = all_weights.mean(axis=0).reshape(7, 7)
        im = ax2.imshow(mean_weight_map, cmap='hot', interpolation='nearest')
        ax2.set_title('Mean Confidence Weight Map (7x7)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Weight')
        ax2.set_xlabel('Spatial X')
        ax2.set_ylabel('Spatial Y')
        
        # Aggiungi valori nelle celle
        for i in range(7):
            for j in range(7):
                val = mean_weight_map[i, j]
                color = 'white' if val > mean_weight_map.mean() else 'black'
                ax2.text(j, i, f'{val:.3f}', ha='center', va='center', 
                        fontsize=7, color=color, fontweight='bold')
        
        # 3. Box plot per alcune posizioni spaziali
        ax3 = axes[1, 0]
        # Seleziona alcune posizioni interessanti: angoli e centro
        positions_to_show = [0, 3, 6, 21, 24, 27, 42, 45, 48]  # angoli + centro
        position_labels = ['TL', 'TM', 'TR', 'ML', 'C', 'MR', 'BL', 'BM', 'BR']
        
        data_to_plot = [all_weights[:, p] for p in positions_to_show]
        bp = ax3.boxplot(data_to_plot, labels=position_labels, showfliers=False)
        ax3.axhline(y=uniform, color='red', linestyle='--', linewidth=1, label='Uniform')
        ax3.set_xlabel('Spatial Position', fontsize=12)
        ax3.set_ylabel('Weight', fontsize=12)
        ax3.set_title('Weight Distribution at Key Positions', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Distribuzione dell'entropia normalizzata
        ax4 = axes[1, 1]
        entropies = -np.sum(all_weights * np.log(all_weights + 1e-10), axis=1)
        normalized_entropies = entropies / np.log(49)
        
        ax4.hist(normalized_entropies, bins=30, color='green', edgecolor='black', alpha=0.7)
        ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Max (uniform)')
        ax4.axvline(x=norm_entropy, color='blue', linestyle='-', linewidth=2, 
                    label=f'Mean: {norm_entropy:.3f}')
        ax4.axvline(x=0.85, color='orange', linestyle=':', linewidth=2, label='Good threshold')
        ax4.set_xlabel('Normalized Entropy', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Entropy Distribution (1.0 = Uniform, <0.85 = Good)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Salva
        save_path = os.path.join(self.cfg['save_dir'], 'confidence_head_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Confidence analysis plot saved to: {save_path}")
        
        plt.show()
        
    def _process_yolo_batch(self, rgb_paths, depth_paths, class_ids):
            batch_size = len(rgb_paths)
            
            rgb_list = []
            depth_list = []
            mask_list = []
            bbox_list = []
            valid_indices = []  
            
            yolo_results = self.yolo_model(list(rgb_paths), conf=self.YOLO_CONF, verbose=False)
            
            for i in range(batch_size):
                rgb_path = rgb_paths[i]
                depth_path = depth_paths[i]
                obj_id = int(class_ids[i])
                target_yolo_class = self.LINEMOD_TO_YOLO[obj_id]
                
                # take results for ony this image
                yolo_res = yolo_results[i]
                
                # take detection with higher conf
                boxes = yolo_res.boxes
                best_idx = None
                best_conf = 0.0
                for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
                    if int(cls) == target_yolo_class and float(conf) > best_conf:
                        best_conf = float(conf)
                        best_idx = j
                
                #If yolo doesn't found the object skip it
                if best_idx is None:
                    # Skip questo sample
                    continue
                
                rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                
                mask_data = yolo_res.masks.data[best_idx].cpu().numpy()
                if mask_data.shape != (self.img_h, self.img_w):
                    mask_data = cv2.resize(mask_data, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                mask = (mask_data > 0.5).astype(np.uint8) * 255
                
                rgb_crop = crop_square_resize(rgb_img, bbox, self.img_size, is_depth=False)
                depth_crop = crop_square_resize(depth_img, bbox, self.img_size, is_depth=True)
                mask_crop = crop_square_resize(mask, bbox, self.img_size, is_depth=True)
                mask_crop = (mask_crop > 127).astype(np.float32)
                
                rgb_list.append(self.transform(rgb_crop))
                depth_list.append(torch.from_numpy(depth_crop).float().unsqueeze(0))
                mask_list.append(torch.from_numpy(mask_crop).float().unsqueeze(0))
                
                x, y, w, h = bbox
                bbox_norm = torch.tensor([
                    (x + w/2) / self.img_w,
                    (y + h/2) / self.img_h,
                    w / self.img_w,
                    h / self.img_h
                ], dtype=torch.float32)
                bbox_list.append(bbox_norm)
                
                # this sample is valid
                valid_indices.append(i)
            
            #If non sample valid
            if len(valid_indices) == 0:
                return None, None, None, None, []
            
            rgb_batch = torch.stack(rgb_list)      # [N, 3, 224, 224]
            depth_batch = torch.stack(depth_list)  # [N, 1, 224, 224]
            mask_batch = torch.stack(mask_list)    # [N, 1, 224, 224]
            bbox_batch = torch.stack(bbox_list)    # [N, 4]
            
            return rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices

    def _process_yolo_batch_wIoU(self, rgb_paths, depth_paths, class_ids):
        batch_size = len(rgb_paths)
        
        rgb_list = []
        depth_list = []
        mask_list = []
        bbox_list = []
        valid_indices = []
        
        # Lista per salvare le IoU di questo batch
        batch_ious = [] 
        
        yolo_results = self.yolo_model(list(rgb_paths), conf=self.YOLO_CONF, verbose=False)
        
        for i in range(batch_size):
            rgb_path = rgb_paths[i]
            depth_path = depth_paths[i]
            obj_id = int(class_ids[i])
            target_yolo_class = self.LINEMOD_TO_YOLO[obj_id]
            
            # --- CARICAMENTO MASK GT (Per calcolo IoU) ---
            # Assuming standard LineMOD strcture : .../rgb/1234.png -> .../mask/1234.png
            mask_gt_path = rgb_path.replace('rgb', 'mask') 
            
            mask_gt = None
            if os.path.exists(mask_gt_path):
                 # Carica in scala di grigi
                mask_gt = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)
                mask_gt = (mask_gt > 127).astype(np.uint8) # Binarizza

            yolo_res = yolo_results[i]
            
            boxes = yolo_res.boxes
            best_idx = None
            best_conf = 0.0
            for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
                if int(cls) == target_yolo_class and float(conf) > best_conf:
                    best_conf = float(conf)
                    best_idx = j
            
            if best_idx is None:
                continue
            
            rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            
            mask_data = yolo_res.masks.data[best_idx].cpu().numpy()
            if mask_data.shape != (self.img_h, self.img_w):
                mask_data = cv2.resize(mask_data, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            mask_yolo = (mask_data > 0.5).astype(np.uint8) * 255
            
            # --- CALCOLO IOU QUI ---
            if mask_gt is not None:
                iou = self._calculate_iou(mask_yolo, mask_gt)
                batch_ious.append(iou)
            else:
                batch_ious.append(0.0) # Fallback se manca GT
            # -----------------------

            rgb_crop = crop_square_resize(rgb_img, bbox, self.img_size, is_depth=False)
            depth_crop = crop_square_resize(depth_img, bbox, self.img_size, is_depth=True)
            mask_crop = crop_square_resize(mask_yolo, bbox, self.img_size, is_depth=True)
            mask_crop = (mask_crop > 127).astype(np.float32)
            
            rgb_list.append(self.transform(rgb_crop))
            depth_list.append(torch.from_numpy(depth_crop).float().unsqueeze(0))
            mask_list.append(torch.from_numpy(mask_crop).float().unsqueeze(0))
            
            x, y, w, h = bbox
            bbox_norm = torch.tensor([(x + w/2) / self.img_w, (y + h/2) / self.img_h, w / self.img_w, h / self.img_h], dtype=torch.float32)
            bbox_list.append(bbox_norm)
            valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return None, None, None, None, [], [] 
        
        rgb_batch = torch.stack(rgb_list)
        depth_batch = torch.stack(depth_list)
        mask_batch = torch.stack(mask_list)
        bbox_batch = torch.stack(bbox_list)
        
        return rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices, batch_ious 

    def run(self):
        """
        Esegue la valutazione completa sul validation set.
        Calcola ADD e ADD-S con threshold 0.1*diameter.
        """
        print("\n" + "="*70)
        print(" STARTING EVALUATION - DAMF_Net")
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
        
        total_correct = 0
        total_predictions = 0
        total_yolo_missed = 0
        all_errors = []  # in cm

        # list to analyze X,YZ error
        all_tx_errors = []
        all_ty_errors = []
        all_tz_errors = []

        all_ious = [] 
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                
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

                rgb_batch, depth_batch, mask_batch, bbox_batch, valid_indices, batch_ious = self._process_yolo_batch_wIoU(
                    paths, depth_paths, class_ids
                )

                # yolo misses
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
                
                if num_valid == 0:
                    continue

                all_ious.extend(batch_ious)

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
        median_add_cm = np.median(all_errors) if len(all_errors) > 0 else 0.0

        mean_trans = np.mean(all_trans_errors) if len(all_trans_errors) > 0 else 0.0
        mean_rot_cm = np.mean(all_rot_errors_m) if len(all_rot_errors_m) > 0 else 0.0
        mean_rot_deg = np.mean(all_rot_errors_deg) if len(all_rot_errors_deg) > 0 else 0.0

        mean_tx = np.mean(all_tx_errors) if len(all_tx_errors) > 0 else 0.0
        mean_ty = np.mean(all_ty_errors) if len(all_ty_errors) > 0 else 0.0
        mean_tz = np.mean(all_tz_errors) if len(all_tz_errors) > 0 else 0.0

        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
        print(f"\n🎭 MASK QUALITY REPORT:")
        print(f"   Mean Mask IoU: {mean_iou:.2f}%")
        if mean_iou < 85.0:
            print("   ⚠️ WARNING: Mask Quality is low! This explains the accuracy drop.")
        
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
