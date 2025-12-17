import torch
from torch.utils.data import Dataset
import yaml
import os
import cv2
import numpy as np
from torchvision import transforms
from utils.Posenet_utils.utils_geometric import matrix_to_quaternion, crop_square_resize
from tqdm import tqdm


class LineModPoseDataset(Dataset):
    def __init__(self, split_file, dataset_root, mode='train', img_size=224):
        self.dataset_root = dataset_root
        self.mode = mode
        self.img_size = img_size
        self.samples = [] #The flat list of ALL possible crops
        
        #split file is ...train_ALL.txt or ...val_ALL.txt
        with open(split_file, 'r') as f:
            image_paths_raw = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[{mode.upper()}] Dataset indexing (Scene-Level Augmentation)...")
        
        # We preload the GTs so we don't have to reopen them a thousand times.
        #    And we build the list of samples.
        #    We use a temporary dictionary to load the yamls once per folder.
        loaded_gts = {} 
        cam_infos_cache = {}

        for img_path_abs in tqdm(image_paths_raw):
            # Parsing del path
            parts = img_path_abs.split(os.sep)
            # Assumiamo struttura: .../data/{folder_id}/rgb/{filename}
            # Se il path nel txt è relativo o diverso, aggiusta questi indici!
            folder_id = parts[-3] 
            img_name = parts[-1]
            img_id_num = int(img_name.replace('.png', '')) 

            d_img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'depth', img_name)

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
            cam_params = [fx,fy,cx,cy]
            
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

                    sample = {
                        'img_path': img_path_abs,
                        'depth_path': d_img_path_abs,
                        'obj_id': obj_id,
                        'bbox': obj['obj_bb'],
                        'R': obj['cam_R_m2c'],
                        't': obj['cam_t_m2c'],
                        'position_input': cam_params,
                    }
                    self.samples.append(sample)

        print(f"[{mode.upper()}] Generated {len(self.samples)} samples from {len(image_paths_raw)} images.")

        # Compute max_depth from the dataset
        print(f"[{mode.upper()}] Computing max depth from dataset...")
        self.max_depth = self._compute_max_depth()
        print(f"[{mode.upper()}] Using max_depth: {self.max_depth:.2f} mm")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.d_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _compute_max_depth(self, sample_ratio=0.1):
        """
        Compute max depth (99th percentile) from a sample of the dataset
        """
        import random
        
        # Sample 10% of dataset for speed (min 50 samples)
        n_samples = max(50, int(len(self.samples) * sample_ratio))
        sampled = random.sample(self.samples, min(n_samples, len(self.samples)))
        
        max_values = []
        
        for sample in tqdm(sampled, desc="Sampling depth values"):
            depth = cv2.imread(sample['depth_path'], cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                # Get valid depth values (exclude zeros)
                valid_depths = depth[depth > 0]
                if len(valid_depths) > 0:
                    max_values.append(valid_depths.max())
        
        if len(max_values) == 0:
            print("Warning: No valid depth values found! Using default 2000mm")
            return 2000.0
        
        max_values = np.array(max_values)
        
        # Use 99th percentile to avoid outliers
        p99 = np.percentile(max_values, 99)
        
        print(f"  Depth statistics from {len(max_values)} samples:")
        print(f"    Absolute max: {max_values.max():.2f} mm")
        print(f"    99th percentile: {p99:.2f} mm")
        print(f"    Mean: {max_values.mean():.2f} mm")
        
        return p99

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        img = cv2.imread(sample['img_path'])
        if img is None:
                # Brutal fallback to avoid stopping training: generate black image
                img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load Depth image
        d_img = cv2.imread(sample['depth_path'], cv2.IMREAD_ANYDEPTH)
        if d_img is None:
            d_img = np.zeros((480, 640), dtype=np.uint16)
        
        bbox = sample['bbox']
        R_matrix = np.array(sample['R']).reshape(3, 3)
        t_vector = np.array(sample['t'])
        target_obj_id = sample['obj_id']
        
        final_bbox = bbox
        d_img = d_img.astype(np.float32)
        d_img = np.clip(d_img, 0, self.max_depth) / self.max_depth

        if self.mode == 'train':
            x, y, w, h = final_bbox
            
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Casual noise (+/- 5% della dimensione)
            noise_x = np.random.uniform(-0.05, 0.05) * w
            noise_y = np.random.uniform(-0.05, 0.05) * h
            
            # casual scale (+/- 5%)
            scale = np.random.uniform(0.95, 1.05)
            
            # Jitter
            new_w = w * scale
            new_h = h * scale
            new_cx = center_x + noise_x
            new_cy = center_y + noise_y
            
            new_x = new_cx - new_w / 2
            new_y = new_cy - new_h / 2
            
            # This is the dirty box
            final_bbox = [new_x, new_y, new_w, new_h]
            
            # Crop both RGB and depth with the same bbox
            img = crop_square_resize(img, final_bbox, self.img_size, is_depth=False)
            d_img = crop_square_resize(d_img, final_bbox, self.img_size, is_depth=True)
            
            # Transform RGB
            img_tensor = self.transform(img)
            
            # Transform depth to tensor
            depth_tensor = torch.from_numpy(d_img).float().unsqueeze(0)
            
            final_bbox = torch.tensor(final_bbox, dtype=torch.float32)
        else:
            # Validation/Test mode
            final_bbox = torch.tensor(bbox, dtype=torch.float32)
            img = cv2.resize(img, (224, 224))
            d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            # Transform RGB
            img_tensor = self.transform(img)
            
            # Transform depth to tensor
            depth_tensor = torch.from_numpy(d_img).float().unsqueeze(0)

        quaternion = matrix_to_quaternion(R_matrix)
        
        quat_tensor = torch.from_numpy(quaternion).float()
        trans_tensor = torch.from_numpy(t_vector).float() 

        params = sample['position_input'] 
        cam_params = torch.tensor([params[0], params[1], params[2], params[3]], dtype=torch.float32)

        if self.mode == 'train':
            return {
                'image': img_tensor,
                'depth': depth_tensor,
                'quaternion': quat_tensor,
                'translation': trans_tensor,
                'class_id': target_obj_id,
                'path': sample['img_path'],
                'bbox': final_bbox,
                'cam_params': cam_params
            }
        else:
            return {

                'quaternion': quat_tensor,
                'translation': trans_tensor,
                'class_id': target_obj_id,
                'path': sample['img_path'],
                'bbox': final_bbox,
                'cam_params': cam_params
            }
