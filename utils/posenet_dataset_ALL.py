import torch
from torch.utils.data import Dataset
import yaml
import os
import cv2
import numpy as np
from torchvision import transforms
from utils.utils_geometric import matrix_to_quaternion, crop_square_resize
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

        print(f"[{mode.upper()}] Indicizzazione dataset (Scene-Level Augmentation)...")
        
        # We preload the GTs so we don't have to reopen them a thousand times.
        #    And we build the list of samples.
        #    We use a temporary dictionary to load the yamls once per folder.
        loaded_gts = {} 

        for img_path_abs in tqdm(image_paths_raw):
            # Parsing del path
            parts = img_path_abs.split(os.sep)
            # Assumiamo struttura: .../data/{folder_id}/rgb/{filename}
            # Se il path nel txt è relativo o diverso, aggiusta questi indici!
            folder_id = parts[-3] 
            img_name = parts[-1]
            img_id_num = int(img_name.replace('.png', '')) 

            # Gestione Path Assoluto per il caricamento immagine dopo
            if not os.path.exists(img_path_abs):
                 img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'rgb', img_name)
            
            # GT Loading in cache , speed improvement
            if folder_id not in loaded_gts:
                gt_path = os.path.join(self.dataset_root, 'data', folder_id, 'gt.yml')
                if os.path.exists(gt_path):
                    with open(gt_path, 'r') as f:
                        loaded_gts[folder_id] = yaml.safe_load(f)
                else:
                    loaded_gts[folder_id] = {} # Empty dict if no GT

            # Retrieving all objects for this image
            gt_data_folder = loaded_gts[folder_id]
            
            if img_id_num in gt_data_folder:
                objs_in_frame = gt_data_folder[img_id_num]
                
                # iterating on all objects in the image
                for obj in objs_in_frame:
                    obj_id = obj['obj_id']
                    
                    # FILTER LOGIC:
                    # If we are in TRAIN: We take EVERYTHING (Augmentation)
                    # If we are in VAL: We take ONLY the “master” object of the folder 
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
                        'obj_id': obj_id,
                        'bbox': obj['obj_bb'],
                        'R': obj['cam_R_m2c'],
                        't': obj['cam_t_m2c']
                    }
                    self.samples.append(sample)

        print(f"[{mode.upper()}] Generated {len(self.samples)} samples from {len(image_paths_raw)} images.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = cv2.imread(sample['img_path'])
        if img is None:
             # Brutal fallback to avoid stopping training: generate black image
             img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox = sample['bbox']
        R_matrix = np.array(sample['R']).reshape(3, 3)
        t_vector = np.array(sample['t'])
        target_obj_id = sample['obj_id']

        # Crop & Jitter
        do_jitter = (self.mode == 'train')
        img_crop = crop_square_resize(img, bbox, self.img_size, jitter=do_jitter)

        quaternion = matrix_to_quaternion(R_matrix)
        
        img_tensor = self.transform(img_crop)
        quat_tensor = torch.from_numpy(quaternion).float()
        trans_tensor = torch.from_numpy(t_vector).float()
        
        return {
            'image': img_tensor,
            'quaternion': quat_tensor,
            'translation': trans_tensor,
            'class_id': target_obj_id,
            'path': sample['img_path']
        }