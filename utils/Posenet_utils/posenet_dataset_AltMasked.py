import torch
from torch.utils.data import Dataset
import yaml
import os
import cv2
import numpy as np
from torchvision import transforms
from utils.Posenet_utils.utils_geometric import matrix_to_quaternion, crop_square_resize
from tqdm import tqdm

OBJID_TO_MASK_PIXEL = {
    1: 21,
    2: 43,
    5: 106,
    6: 128,
    8: 170,
    9: 191,
    10: 213,
    11: 234,
    12: 255
}

class LineModPoseDataset_AltMasked(Dataset):
    def __init__(self, dataset_root, mode='train', img_size=224, noise_factor=0.10, aug_intensity='aggressive'):
        """
        Args:
            dataset_root (str): Percorso alla root del dataset (es. .../Linemod_preprocessed)
                                Si aspetta che dentro ci sia la cartella 'data/01', 'data/02', ecc.
            mode (str): 'train' o 'val'. Se 'train' legge train.txt, se 'val' legge test.txt.
            img_size (int): Dimensione output.
            noise_factor (float): Intensità data augmentation geometrica.
        """
        self.dataset_root = dataset_root
        self.mode = mode
        self.img_size = img_size
        self.noise_factor = noise_factor
        self.img_h, self.img_w = 480, 640 
        self.samples = [] 

        self._setup_aug_params(aug_intensity)

        obj_ids = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
        
        target_filename = 'train.txt' if mode == 'train' else 'test.txt'
        
        print(f"[{mode.upper()}] Scanning LineMOD folders for '{target_filename}'...")

        image_paths_raw = []
        
        for oid in obj_ids:
            txt_path = os.path.join(self.dataset_root, 'data', oid, target_filename)
            
            if not os.path.exists(txt_path):
                print(f"⚠️ Warning: Split file not found: {txt_path}")
                continue
            
            with open(txt_path, 'r') as f:
                # Legge gli ID (es. "0000", "0023") pulendo spazi e a capo
                frame_ids = [line.strip() for line in f.readlines() if line.strip()]
            
            # Ricostruisce il path assoluto per ogni immagine
            for fid in frame_ids:
                # Struttura LineMOD standard: data/<id>/rgb/<frame>.png
                # Nota: Assumiamo estensione .png. Se è .jpg cambiala qui.
                full_path = os.path.join(self.dataset_root, 'data', oid, 'rgb', f"{fid}.png")
                image_paths_raw.append(full_path)

        print(f"[{mode.upper()}] Found {len(image_paths_raw)} total images across {len(obj_ids)} objects.")
        
        loaded_gts = {} 
        cam_infos_cache = {}

        for img_path_abs in tqdm(image_paths_raw, desc="Indexing Dataset"):
            # Parsing del path
            parts = img_path_abs.split(os.sep)
            # Assumiamo struttura: .../data/{folder_id}/rgb/{filename}
            # Se il path nel txt è relativo o diverso, aggiusta questi indici!
            folder_id = parts[-3] 
            img_name = parts[-1]
            img_id_num = int(img_name.replace('.png', '')) 

            d_img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'depth', img_name)
            mask_img_path = os.path.join(self.dataset_root,'data', folder_id, 'mask' ,img_name)

            # Gestione Path Assoluto per il caricamento immagine dopo
            if not os.path.exists(img_path_abs):
                img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'rgb', img_name)
            
            # GT Loading in cache , speed improvement
            if folder_id not in loaded_gts:
                gt_path = os.path.join(self.dataset_root, 'data', folder_id, 'gt.yml')
                cam_info_path = os.path.join(self.dataset_root, 'data', folder_id, 'info.yml')
                if os.path.exists(gt_path) and os.path.exists(cam_info_path):
                    with open(gt_path, 'r') as f:
                        loaded_gts[folder_id] = yaml.safe_load(f)
                    with open(cam_info_path, 'r') as c:
                        cam_infos_cache[folder_id] = yaml.safe_load(c)
                    
                    
                else:
                    loaded_gts[folder_id] = {} 
                    cam_infos_cache[folder_id] = {}


            gt_data_folder = loaded_gts[folder_id]
            cam_infos = cam_infos_cache[folder_id][img_id_num]

            cam_K_list = cam_infos['cam_K']
            fx = cam_K_list[0]
            fy = cam_K_list[4]
            cx = cam_K_list[2]
            cy = cam_K_list[5]
            #cam_params = [fx,fy,cx,cy]

            cam_params_norm = [
                fx / self.img_w,  # focal_x normalizzata
                fy / self.img_h,  # focal_y normalizzata
                cx / self.img_w,  # principal point x normalizzato [0-1]
                cy / self.img_h   # principal point y normalizzato [0-1]
            ]
            
            if img_id_num in gt_data_folder:
                objs_in_frame = gt_data_folder[img_id_num]
                
                # iterating on all objects in the image
                for obj in objs_in_frame:
                    obj_id = obj['obj_id']
                    
                    # FILTER LOGIC:
                    # If we are in TRAIN: We take EVERYTHING (Augmentation)
                    # If we are in VAL: We take ONLY the "master" object of the folder 
                    # (to keep the validation set standard and comparable)
                    target_id_standard = int(folder_id)
                    
                    if mode == 'train':
                        #Accept everything
                        pass 
                    else:
                        if obj_id != target_id_standard:
                            continue

                    x_min, y_min, w_box, h_box = obj['obj_bb']

                    # === GESTIONE MASCHERE ===
                    """
                    if folder_id == '02':
                        mask_img_path = os.path.join(self.dataset_root, 'data', folder_id, 'mask_all', img_name)
                        use_mask_all = True
                    else:
                        mask_img_path = os.path.join(self.dataset_root, 'data', folder_id, 'mask', img_name)
                        use_mask_all = False
                    """
                    if folder_id == '02':
                        if mode == 'train':
                            # Training: usa mask_all (tutti gli oggetti per augmentation)
                            mask_img_path = os.path.join(self.dataset_root, 'data', folder_id, 'mask_all', img_name)
                            use_mask_all = True
                        else:
                            # Validation: usa mask/ standard (solo benchvise, obj_id=2)
                            mask_img_path = os.path.join(self.dataset_root, 'data', folder_id, 'mask', img_name)
                            use_mask_all = False
                    else:
                        mask_img_path = os.path.join(self.dataset_root, 'data', folder_id, 'mask', img_name)
                        use_mask_all = False

                    sample = {
                        'img_path': img_path_abs,
                        'depth_path': d_img_path_abs,
                        'mask_path': mask_img_path,
                        'obj_id': obj_id,
                        'bbox': [x_min, y_min, w_box, h_box],
                        'R': obj['cam_R_m2c'],
                        't': obj['cam_t_m2c'],
                        'position_input': cam_params_norm,
                        'use_mask_all': use_mask_all,
                    }
                    self.samples.append(sample)

        print(f"[{mode.upper()}] Generated {len(self.samples)} samples from {len(image_paths_raw)} images.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def _setup_aug_params(self, intensity):
        """Configura parametri augmentation in base all'intensità"""
        if intensity == 'light':
            self.aug_prob = 0.3
            self.brightness_range = (0.9, 1.1)
            self.contrast_range = (0.9, 1.1)
            self.blur_kernel_max = 3
            self.noise_std = 5
            self.hue_shift = 5
            self.sat_range = (0.95, 1.05)
            self.depth_noise_std = 0.005
        elif intensity == 'medium':
            self.aug_prob = 0.5
            self.brightness_range = (0.8, 1.2)
            self.contrast_range = (0.8, 1.2)
            self.blur_kernel_max = 5
            self.noise_std = 10
            self.hue_shift = 10
            self.sat_range = (0.9, 1.1)
            self.depth_noise_std = 0.01
        else:  # aggressive
            self.aug_prob = 0.7
            self.brightness_range = (0.6, 1.4)
            self.contrast_range = (0.7, 1.3)
            self.blur_kernel_max = 7
            self.noise_std = 15
            self.hue_shift = 15
            self.sat_range = (0.8, 1.2)
            self.depth_noise_std = 0.02

    def _extract_clean_mask(self, mask_raw, obj_id):
        """Estrae maschera pulita usando la mappatura pixel corretta"""
        if obj_id in OBJID_TO_MASK_PIXEL:
            target_pixel_value = OBJID_TO_MASK_PIXEL[obj_id]
            return np.where(mask_raw == target_pixel_value, 255, 0).astype(np.uint8)
        else:
            # Fallback
            return np.ones_like(mask_raw) * 255

    def _extract_contaminated_mask(self, mask_raw):
        """Usa la maschera così com'è (tutti gli oggetti visibili nel crop)"""
        # Binarizza: tutto ciò che non è sfondo (0) diventa 255
        return np.where(mask_raw > 0, 255, 0).astype(np.uint8)

    def _augment_rgb(self, img):
        """
        Augmentation RGB aggressiva.
        Input: numpy array RGB (H, W, 3) uint8
        Output: numpy array RGB (H, W, 3) uint8
        """
        img = img.copy()
        
        # 1. Brightness & Contrast
        if np.random.rand() < self.aug_prob:
            alpha = np.random.uniform(*self.contrast_range)  # contrast
            beta = np.random.uniform(-30, 30) * (self.brightness_range[1] - 1) / 0.2  # brightness scaled
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 2. Gaussian Blur
        if np.random.rand() < self.aug_prob * 0.6:
            k = np.random.choice([3, 5, self.blur_kernel_max])
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)
        
        # 3. Gaussian Noise
        if np.random.rand() < self.aug_prob * 0.5:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 4. HSV Perturbation (Hue & Saturation)
        if np.random.rand() < self.aug_prob * 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Hue shift
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-self.hue_shift, self.hue_shift)) % 180
            # Saturation scale
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(*self.sat_range), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 5. Random Shadow (simula illuminazione non uniforme)
        if np.random.rand() < self.aug_prob * 0.3:
            h, w = img.shape[:2]
            # Crea gradiente casuale
            x1, x2 = np.random.randint(0, w, 2)
            shadow_mask = np.linspace(0.5, 1.0, w).reshape(1, -1)
            if np.random.rand() < 0.5:
                shadow_mask = shadow_mask[:, ::-1]
            shadow_mask = np.tile(shadow_mask, (h, 1))
            img = (img * shadow_mask[:, :, np.newaxis]).astype(np.uint8)
        
        # 6. Channel Shuffle (raro, per robustezza)
        if np.random.rand() < 0.1:
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            img = img[:, :, channels]
        
        return img

    def _augment_depth(self, depth):
        """
        Augmentation Depth.
        Input: numpy array (H, W) float32 in metri
        Output: numpy array (H, W) float32 in metri
        """
        depth = depth.copy()
        
        # 1. Depth noise (simula rumore sensore)
        if np.random.rand() < self.aug_prob:
            noise = np.random.normal(0, self.depth_noise_std, depth.shape).astype(np.float32)
            depth = depth + noise
            depth = np.maximum(depth, 0)  # No negative depth
        
        # 2. Depth scale perturbation (simula calibrazione imperfetta)
        if np.random.rand() < self.aug_prob * 0.3:
            scale = np.random.uniform(0.98, 1.02)
            depth = depth * scale
        
        # 3. Missing depth simulation (dropout casuale)
        if np.random.rand() < self.aug_prob * 0.2:
            mask = np.random.rand(*depth.shape) > 0.05  # 5% dropout
            depth = depth * mask
        
        return depth

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. LOAD RGB
        img = cv2.imread(sample['img_path'])
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. LOAD DEPTH
        d_img = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
        if d_img is None: 
            d_img = np.zeros((480, 640), dtype=np.float32)
        d_img = d_img.astype(np.float32) / 1000.0
        
        # 3. LOAD MASK
        mask_raw = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask_raw is None: 
            mask = np.ones((480, 640), dtype=np.uint8) * 255
        else:
            if sample['use_mask_all']:
                # === FOLDER 02: Decidi se usare maschera pulita o contaminata ===
                use_contaminated = (self.mode == 'train')
                
                if use_contaminated:
                    # Maschera "sporca": include tutti gli oggetti nel crop
                    mask = self._extract_contaminated_mask(mask_raw)
                else:
                    # Maschera pulita: solo l'oggetto target
                    mask = self._extract_clean_mask(mask_raw, sample['obj_id'])
            else:
                # Altri folder: maschera standard
                mask = mask_raw

        """
        depth_stats = np.array([
            valid_depth.min(),
            valid_depth.max(),
            valid_depth.mean(),
            valid_depth.std()
        ])
        depth_stats_tensor = torch.tensor(depth_stats, dtype=torch.float32)
        """

        bbox = sample['bbox']
        R_matrix = np.array(sample['R']).reshape(3, 3)
        t_vector = np.array(sample['t'])
        target_obj_id = sample['obj_id']


        # --- TRAIN AUGMENTATION ---
        if self.mode == 'train':
            x, y, w, h = bbox

            center_x = x + w / 2
            center_y = y + h / 2

            noise_x = np.random.uniform(-self.noise_factor, self.noise_factor) * w
            noise_y = np.random.uniform(-self.noise_factor, self.noise_factor) * h

            scale = np.random.uniform(0.95, 1.05)

            new_w = w * scale
            new_h = h * scale

            new_cx = center_x + noise_x
            new_cy = center_y + noise_y

            new_x = new_cx - new_w / 2
            new_y = new_cy - new_h / 2
            
            final_bbox = [new_x, new_y, new_w, new_h]

            bbox_norm = [
                np.clip(new_cx / self.img_w, 0, 1),
                np.clip(new_cy / self.img_h, 0, 1),
                np.clip(new_w / self.img_w, 0, 1),
                np.clip(new_h / self.img_h, 0, 1)
            ]
            
            # MASK AUGMENTATION (erosion/dilation)
            chance = np.random.rand()
            #kernel_size = np.random.randint(3, 8) #kernel changes from 3 to 8 and determine how aggressive is the erosion/dilatation
            kernel_size = np.random.randint(3, 15)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if chance < 0.4: 
                mask = cv2.erode(mask, kernel, iterations=1)
            elif chance < 0.8:
                mask = cv2.dilate(mask, kernel, iterations=1)
            
        else:
            final_bbox = bbox
            x, y, w, h = bbox
            bbox_norm = [
                (x + w / 2) / self.img_w,
                (y + h / 2) / self.img_h,
                w / self.img_w,
                h / self.img_h
            ]

        # 4. CROP & RESIZE
        img_crop = crop_square_resize(img, final_bbox, self.img_size, is_depth=False)
        d_img_crop = crop_square_resize(d_img, final_bbox, self.img_size, is_depth=True)
        mask_crop = crop_square_resize(mask, final_bbox, self.img_size, is_depth=True)
        mask_crop = (mask_crop > 127).astype(np.float32)

        if self.mode == 'train':
            img_crop = self._augment_rgb(img_crop)
            d_img_crop = self._augment_depth(d_img_crop)
        
        # 5. TO TENSOR
        img_tensor = self.transform(img_crop)
        depth_tensor = torch.from_numpy(d_img_crop).float().unsqueeze(0) 
        mask_tensor = torch.from_numpy(mask_crop).float().unsqueeze(0)

        # Labels
        quaternion = matrix_to_quaternion(R_matrix)
        quat_tensor = torch.from_numpy(quaternion).float()
        trans_tensor = torch.from_numpy(t_vector).float() / 1000.0

        params = sample['position_input'] 
        cam_params = torch.tensor(params, dtype=torch.float32)


        return {
            'image': img_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor, 
            'quaternion': quat_tensor,
            'translation': trans_tensor,
            'class_id': target_obj_id,
            'path': sample['img_path'],
            'depth_path': sample['depth_path'],
            'bbox_norm': torch.tensor(bbox_norm, dtype=torch.float32),
            'cam_params': cam_params
        }
