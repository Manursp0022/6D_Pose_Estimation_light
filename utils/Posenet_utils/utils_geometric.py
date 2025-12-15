import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R

def matrix_to_quaternion(R_matrix):
    r = R.from_matrix(R_matrix)
    # Scipy returns [x, y, z, w], but PyTorch/Standard often uses [w, x, y, z]    
    quat_scipy = r.as_quat() 
    # Let's reorder in [w, x, y, z]
    quat_wxyz = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
    return quat_wxyz

def crop_square_resize(img, bbox, target_size=224):
    """
    Esegue un ritaglio quadrato centrato sul bbox fornito e ridimensiona.
    NON applica jittering casuale (deve essere fatto prima se necessario).
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    
    # --- 1. SQUARE PADDING CALCULATION ---
    # Vogliamo un crop quadrato. Prendiamo il lato maggiore.
    # Aggiungiamo un margine fisso (es. 1.2x) per dare contesto alla rete
    # Questo NON è jitter, è context padding necessario per la ResNet
    pad_factor = 1.2 
    side = max(w, h) * pad_factor
    
    # Coordinate del crop quadrato centrato sul bbox fornito
    center_x = x + w / 2
    center_y = y + h / 2
    
    x1 = int(center_x - side / 2)
    y1 = int(center_y - side / 2)
    x2 = int(center_x + side / 2)
    y2 = int(center_y + side / 2)

    # --- 2. CROP CON PADDING (Gestione bordi) ---
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

    # Aggiungiamo il bordo nero se necessario (zero padding)
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])

    # --- 3. RESIZE ---
    # A questo punto 'crop' è quadrato. Lo ridimensioniamo a 224x224.
    try:
        final_img = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # Fallback di sicurezza se il crop è venuto vuoto (es. bbox invalido)
        final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
    return final_img


def solve_pinhole_diameter(bboxes, intrinsics, diameters_batch):
    # Calculate diagonal pixels of the bbox
    w_pixel = bboxes[:, 2]
    h_pixel = bboxes[:, 3]
    d_pixel = torch.sqrt(w_pixel**2 + h_pixel**2)
    
    fx = intrinsics[:, 0]  # Primo valore
    fy = intrinsics[:, 1]  # Secondo valore
    cx = intrinsics[:, 2]  # Terzo valore
    cy = intrinsics[:, 3]  # Quarto valore
    
    # Average focal length calculation
    f_avg = (fx + fy) / 2

    # Z = (f * Actual_Diameter) / Pixel_Diagonal
    Z = (f_avg * diameters_batch) / d_pixel

    # X and Y
    # (u_center and v_center are the bbox center coordinates)
    u_center, v_center = bboxes[:, 0], bboxes[:, 1]
    
    X = ((u_center - cx) * Z) / fx
    Y = ((v_center - cy) * Z) / fy

    return torch.stack([X, Y, Z], dim=1)