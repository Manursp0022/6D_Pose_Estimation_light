import torch
from torch.utils.data import Dataset
import yaml
import os
import cv2
import numpy as np
from torchvision import transforms
from utils.Posenet_utils.utils_geometric import matrix_to_quaternion, crop_square_resize
from tqdm import tqdm


# === MAPPATURA PIXEL -> OBJ_ID PER FOLDER 02 ===
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


class LineModPoseDatasetMasked(Dataset):
    def __init__(self, split_file, dataset_root, mode='train', img_size=224, noise_factor=0.08):
        """
        Args:
            mask_contamination_prob: Probabilità di usare maschera "sporca" (crop bbox) 
                                     invece di quella pulita. Solo in training.
                                     0.0 = sempre pulita, 1.0 = sempre sporca
        """
        self.dataset_root = dataset_root
        self.mode = mode
        self.img_size = img_size
        self.noise_factor = noise_factor
        self.img_h, self.img_w = 480, 640 
        self.samples = []
        
        with open(split_file, 'r') as f:
            image_paths_raw = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[{mode.upper()}] Dataset indexing (Scene-Level Augmentation)...")
        
        loaded_gts = {} 
        cam_infos_cache = {}

        for img_path_abs in tqdm(image_paths_raw):
            parts = img_path_abs.split(os.sep)
            folder_id = parts[-3] 
            img_name = parts[-1]
            img_id_num = int(img_name.replace('.png', '')) 

            d_img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'depth', img_name)
            
            if not os.path.exists(img_path_abs):
                img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'rgb', img_name)
            
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
            
            cam_params_norm = [
                fx , fy , cx , cy
            ]

            """
            cam_params_norm = [
                fx ,
                fy ,
                cx ,
                cy
            ]
            """

            
            if img_id_num in gt_data_folder:
                objs_in_frame = gt_data_folder[img_id_num]
                
                for obj_idx, obj in enumerate(objs_in_frame):
                    obj_id = obj['obj_id']
                    
                    target_id_standard = int(folder_id)
                    
                    if mode == 'train':
                        pass 
                    else:
                        if obj_id != target_id_standard:
                            continue

                    x_min, y_min, w_box, h_box = obj['obj_bb']

                    # === GESTIONE MASCHERE ===
                    if folder_id == '02':
                        mask_img_path = os.path.join(self.dataset_root, 'data', folder_id, 'mask_all', img_name)
                        use_mask_all = True
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
            kernel_size = np.random.randint(3, 8)
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