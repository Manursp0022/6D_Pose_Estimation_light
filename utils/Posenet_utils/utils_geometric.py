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

def crop_square_resize(img, bbox, target_size=224, is_depth=False):
    """
    Esegue un ritaglio quadrato centrato sul bbox fornito e ridimensiona.
    Supporta sia immagini RGB che depth.
    
    Args:
        img: Immagine RGB (H, W, 3) o depth (H, W) o (H, W, 1)
        bbox: Bounding box [x, y, w, h]
        target_size: Dimensione target (default 224)
        is_depth: True se è un'immagine depth
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    
    # --- 1. SQUARE PADDING CALCULATION ---
    pad_factor = 1.2 
    side = max(w, h) * pad_factor
    
    center_x = x + w / 2
    center_y = y + h / 2
    
    x1 = int(center_x - side / 2)
    y1 = int(center_y - side / 2)
    x2 = int(center_x + side / 2)
    y2 = int(center_y + side / 2)

    # --- 2. CROP CON PADDING ---
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w_img, x2)
    crop_y2 = min(h_img, y2)

    # Ritaglio
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Padding con zero (va bene per depth, zero = nessuna profondità)
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        if is_depth:
            # Per depth: padding value = 0 (2D array)
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, 
                                     cv2.BORDER_CONSTANT, value=0)
        else:
            # Per RGB: padding value = [0,0,0]
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, 
                                     cv2.BORDER_CONSTANT, value=[0,0,0])

    # --- 3. RESIZE ---
    try:
        if is_depth:
            # Per depth: usa INTER_NEAREST per preservare i valori esatti
            # oppure INTER_LINEAR se vuoi interpolazione smooth
            final_img = cv2.resize(crop, (target_size, target_size), 
                                  interpolation=cv2.INTER_NEAREST)
        else:
            final_img = cv2.resize(crop, (target_size, target_size), 
                                  interpolation=cv2.INTER_LINEAR)
    except Exception:
        if is_depth:
            final_img = np.zeros((target_size, target_size), dtype=crop.dtype)
        else:
            final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
    return final_img


def solve_pinhole_diameter(bboxes, intrinsics, diameters_batch):
    # Calcola diagonale pixel del bbox
    w_pixel = bboxes[:, 2]
    h_pixel = bboxes[:, 3]
    d_pixel = torch.sqrt(w_pixel**2 + h_pixel**2)

    # --- MODIFICA QUI ---
    # Il dataset ci dà un vettore [fx, fy, cx, cy], non una matrice 3x3.
    # Quindi accediamo direttamente agli indici 0, 1, 2, 3.
    
    fx = intrinsics[:, 0]  # Primo valore
    fy = intrinsics[:, 1]  # Secondo valore
    cx = intrinsics[:, 2]  # Terzo valore
    cy = intrinsics[:, 3]  # Quarto valore
    
    # Calcolo focale media
    f_avg = (fx + fy) / 2

    # Z = (f * Diametro_Reale) / Diagonale_Pixel
    Z = (f_avg * diameters_batch) / d_pixel

    # X e Y
    # (u_center e v_center sono le coordinate centro bbox)
    u_center, v_center = bboxes[:, 0], bboxes[:, 1]
    
    X = ((u_center - cx) * Z) / fx
    Y = ((v_center - cy) * Z) / fy

    return torch.stack([X, Y, Z], dim=1)