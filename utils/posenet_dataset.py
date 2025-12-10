import torch
from torch.utils.data import Dataset
import yaml
import os
from torchvision import transforms
from utils_geometric import matrix_to_quaternion,crop_square_resize

class LineModPoseDataset(Dataset):
    def __init__(self, split_file, dataset_root, mode='train', img_size=224):
        """
        Args:
            split_file (str): Path to autosplit_train_ALL.txt or autosplit_val_ALL.txt
            dataset_root (str): Root folder (Linemod_preprocessed)
            mode (str): 'train' or ‘val’. If ‘train’, enable Jitter.
            img_size (int): ResNet input size (default 224).
        """
        self.dataset_root = dataset_root
        self.mode = mode
        self.img_size = img_size
        
        # Carichiamo la lista delle immagini dal file txt
        with open(split_file, 'r') as f:
            # .strip() rimuove spazi vuoti e a capo
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"[{mode.upper()}] Caricate {len(self.image_paths)} immagini da {split_file}")

        # ALL ground truths into memory (gt Cache).
        # This avoids opening yaml files 1000 times per second during training.
        self.gt_cache = {} # {'01': data_dict, '02': data_dict...}
        self._preload_gt()

        # Standard transformations for PyTorch (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(), # [0,255] -> [0,1] e HWC -> CHW
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preload_gt(self):
        """Carica i file gt.yml di tutte le cartelle in memoria."""
        all_folders = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
        print("Pre-loading Ground Truth data...")
        for folder in all_folders:
            path = os.path.join(self.dataset_root, 'data', folder, 'gt.yml')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.gt_cache[folder] = yaml.safe_load(f)
        print("GT Loaded.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path_abs = self.image_paths[idx] 
        

        # path: .../data/01/rgb/0055.png
        parts = img_path_abs.split(os.sep)
        # let's assume standard structure: .../data/{folder_id}/rgb/{filename}
        folder_id = parts[-3] 
        img_name = parts[-1]
        img_id_num = int(img_name.replace('.png', ''))
        
        # (If the path in the txt file does not exist on the current disk, try rebuilding it with dataset_root)
        if not os.path.exists(img_path_abs):
             img_path_abs = os.path.join(self.dataset_root, 'data', folder_id, 'rgb', img_name)

        img = cv2.imread(img_path_abs)
        if img is None:
            raise ValueError(f"Immagine non trovata o corrotta: {img_path_abs}")
        # BGR (OpenCV) -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # retrieve Ground Truth from Cache
        gt_data = self.gt_cache[folder_id][img_id_num]
        
        # In the LINEMOD dataset, date/02 has multiple objects.
        # We must ONLY take the object that corresponds to the current folder_id.
        target_obj_id = int(folder_id)
        
        target_obj = None
        # gt_data is a list of objects for that photo
        for obj in gt_data:
            if obj['obj_id'] == target_obj_id:
                target_obj = obj
                break
        
        if target_obj is None:
            # Caso raro: l'oggetto target non è nel gt di questa foto (es. troppo occluso?)
            # Saltiamo o ritorniamo un errore. Qui gestiamo semplice.
            raise ValueError(f"Oggetto {target_obj_id} non trovato nel GT di {img_name}")

        # D. Estrazione Dati Posa
        bbox = target_obj['obj_bb'] # [x, y, w, h]
        R_matrix = np.array(target_obj['cam_R_m2c']).reshape(3, 3)
        t_vector = np.array(target_obj['cam_t_m2c']) # [tx, ty, tz]

        # E. Elaborazione Immagine (Crop & Jitter)
        # Se self.mode == 'train', jitter è True
        do_jitter = (self.mode == 'train')
        img_crop = crop_square_resize(img, bbox, self.img_size, jitter=do_jitter)

        # F. Elaborazione Label (Matrix -> Quaternion)
        quaternion = matrix_to_quaternion(R_matrix) # [w, x, y, z]

        # G. Conversione in Tensori
        img_tensor = self.transform(img_crop) # Normalizza e permuta
        
        # Convertiamo label in FloatTensor
        quat_tensor = torch.from_numpy(quaternion).float()
        trans_tensor = torch.from_numpy(t_vector).float()
        
        # Ritorniamo tutto ciò che serve alla Loss Function
        return {
            'image': img_tensor,
            'quaternion': quat_tensor,   # Target Rotazione
            'translation': trans_tensor, # Target Traslazione
            'class_id': target_obj_id,   # Utile per debug
            'path': img_path_abs         # Utile per debug
        }