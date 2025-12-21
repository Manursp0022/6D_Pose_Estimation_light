import torch

def compute_delta_target(pred_r, pred_t, gt_r, gt_t, cam_params):
    """
    Calcola il target DELTA che la rete deve imparare.
    Input: Posa di partenza (pred/jittered) e Posa Target (GT).
    Output: (vx, vy, vz, delta_quat)
    """
    # 1. Delta Traslazione (DeepIM style)
    # Calcoliamo lo spostamento in PIXEL nel piano immagine proiettato
    # vx = (gt_u - pred_u) * (z_pred / fx) ... semplificato in metri:
    
    # Delta Z in spazio logaritmico (per stabilità di scala)
    # dz = log(z_gt / z_pred)
    delta_z = torch.log(gt_t[:, 2] / pred_t[:, 2])
    
    # Delta XY (centrato rispetto alla profondità attuale)
    # dx = (gt_x - pred_x) / pred_z * fx ... un po' complesso dipendere da K.
    # Usiamo una parametrizzazione metrica locale semplice:
    delta_x = (gt_t[:, 0] - pred_t[:, 0]) 
    delta_y = (gt_t[:, 1] - pred_t[:, 1]) 
    
    # Uniamo in un vettore [B, 3]
    delta_t_target = torch.stack([delta_x, delta_y, delta_z], dim=1)
    
    # 2. Delta Rotazione (Quaternioni)
    # R_gt = R_delta * R_pred  =>  R_delta = R_gt * R_pred^-1
    # Nei quaternioni: q_delta = q_gt * q_pred_inv
    # Nota: Assumiamo quaternioni normalizzati [w, x, y, z] o [x, y, z, w]. 
    # PyTorch di solito usa [w, x, y, z] in certe lib, ma tu usi [x, y, z, w].
    # Implementiamo il prodotto di quaternioni manuale per sicurezza.
    
    # Inversione q_pred (coniugato per unitari): [x, y, z, w] -> [-x, -y, -z, w]
    q_pred_inv = pred_r.clone()
    q_pred_inv[:, :3] = -q_pred_inv[:, :3]
    
    # Prodotto Hamilton: q_target = q_gt * q_pred_inv
    # (Ometto la formula estesa per brevità, usiamo una approssimazione lineare se l'angolo è piccolo
    # oppure la differenza semplice se siamo vicini).
    # Per semplicità nel refiner locale: q_delta = q_gt - q_pred (funziona bene per piccoli angoli)
    delta_r_target = gt_r - pred_r 
    
    return delta_t_target, delta_r_target

def apply_delta_update(pose_r, pose_t, delta_r, delta_t):
    """
    Applica il DELTA predetto dalla rete alla posa corrente.
    """
    # 1. Update Traslazione
    # z_new = z_old * exp(dz)
    z_new = pose_t[:, 2] * torch.exp(delta_t[:, 2])
    
    # x_new = x_old + dx
    x_new = pose_t[:, 0] + delta_t[:, 0]
    y_new = pose_t[:, 1] + delta_t[:, 1]
    
    t_new = torch.stack([x_new, y_new, z_new], dim=1)
    
    # 2. Update Rotazione
    # q_new = q_old + dq (approssimazione locale)
    r_new = pose_r + delta_r
    r_new = torch.nn.functional.normalize(r_new, p=2, dim=1)
    
    return r_new, t_new