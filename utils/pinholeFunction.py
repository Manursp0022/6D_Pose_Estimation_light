import numpy as np

def get_3d_coordinates(bbox, K, h_real):
    """
    Trasforma le coordinate 2D di YOLO in coordinate spaziali 3D.
    
    Parametri:
    - bbox: list/tuple (u_c, v_c, w_pix, h_pix) -> Yolo bounding box
    - K: np.array 3x3 -> Camera intrinsics matrix
    - h_real: float -> Object real height in meters (es. Persona = 1.75)
    
    Ritorna:
    - np.array([X, Y, Z]): 3D coordinates in meters
    """
    u_c, v_c, w_pix, h_pix = bbox
    
    # Estrarre i parametri dalla matrice K
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 1. Stima della distanza (Z) basata sull'altezza reale dell'oggetto
    # Formula: Z = (focale_y * altezza_reale) / altezza_pixel
    Z = (fy * h_real) / h_pix
    
    # 2. Trasformazione da pixel a coordinate spaziali (X, Y)
    # Formula derivata dal modello pinhole: u = (fx * X)/Z + cx  => X = (u - cx) * Z / fx
    X = (u_c - cx) * Z / fx
    Y = (v_c - cy) * Z / fy
    
    return np.array([X, Y, Z])
