import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import trimesh

def load_object_points(ply_path):
    """
    Loads vertices from a .ply file.
    """
    mesh = trimesh.load(ply_path)
    # Extract the vertices (points) of the model
    return np.array(mesh.vertices)

def matrix_to_quaternion(R_matrix):
    r = R.from_matrix(R_matrix)
    # Scipy returns [x, y, z, w], but PyTorch/Standard often uses [w, x, y, z]    
    quat_scipy = r.as_quat() 
    # Let's reorder in [w, x, y, z]
    quat_wxyz = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
    return quat_wxyz

def quaternion_to_matrix(quat):
    # quat is in [w, x, y, z]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # reorder to [x, y, z, w]
    R_matrix = r.as_matrix()
    return R_matrix

def crop_square_resize(img, bbox, target_size=224, jitter=True):
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    
    # --- 1. JITTERING (Solo se jitter=True) ---
    if jitter:
        # Spostiamo il centro del box casualmente (+/- 5%)
        center_x = x + w / 2
        center_y = y + h / 2
        noise_x = np.random.uniform(-0.05, 0.05) * w
        noise_y = np.random.uniform(-0.05, 0.05) * h
        
        # Scaliamo il box casualmente (+/- 5%)
        scale = np.random.uniform(0.95, 1.05)        
        # Nuove dimensioni e centro
        w = w * scale
        h = h * scale
        new_cx = center_x + noise_x
        new_cy = center_y + noise_y
        
        # Ricalcoliamo x, y (top-left)
        x = new_cx - w / 2
        y = new_cy - h / 2

    # --- 2. SQUARE PADDING CALCULATION ---
    # Vogliamo un crop quadrato. Prendiamo il lato maggiore.
    # Aggiungiamo un margine (es. 1.2x) per dare contesto alla rete
    side = max(w, h) * 1.1 
    
    # Coordinate del crop quadrato (possono uscire dall'immagine)
    center_x = x + w / 2
    center_y = y + h / 2
    
    x1 = int(center_x - side / 2)
    y1 = int(center_y - side / 2)
    x2 = int(center_x + side / 2)
    y2 = int(center_y + side / 2)

    # --- 3. CROP CON PADDING (Gestione bordi) ---
    # Calcoliamo quanto "sbordiamo" per aggiungere il padding nero
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)

    # Coordinate sicure per il ritaglio su immagine reale
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w_img, x2)
    crop_y2 = min(h_img, y2)

    # Ritaglio
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Aggiungiamo il bordo nero se necessario
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])

    # --- 4. RESIZE ---
    # A questo punto 'crop' è quadrato. Lo ridimensioniamo a 224x224.
    try:
        final_img = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # Fallback di sicurezza se il crop è venuto vuoto (raro)
        final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
    return final_img