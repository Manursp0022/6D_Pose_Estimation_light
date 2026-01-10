import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseFusionLoss_Conf(nn.Module):
    def __init__(self, rot_weight=1.0, trans_weight=0.3, w_rate=0.01):
        super().__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.w_rate = w_rate 
        self.sym_list = [10, 11] # Eggbox, Glue (Oggetti simmetrici)
        self.eps = 1e-8

    def quaternion_to_matrix(self, quaternions):
        """
        Converte batch di quaternioni [N, 4] in matrici di rotazione [N, 3, 3]
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
            ), -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def forward(self, pred_r, pred_t, pred_c, gt_r, gt_t, model_points, obj_ids, return_metrics=False):
        """
        Calcola la Loss DenseFusion esatta (Eq. 2 e 3 del paper) su tutte le patch.
        Args:
            pred_r: [B, 4, 49]
            pred_t: [B, 3, 49]
            pred_c: [B, 1, 49]
            gt_r:   [B, 4]
            gt_t:   [B, 3]
            model_points: [B, N_PTS, 3]
            obj_ids: [B]
            return_metrics: bool, se True ritorna dizionario per logging
        """
        batch_size, _, num_patches = pred_r.shape
        num_points = model_points.shape[1]
        
        total_loss = 0.0
        
        # Accumulatori per le metriche (calcolate sulla posa finale pesata)
        meta_rot_loss = 0.0
        meta_trans_loss = 0.0
        meta_add_loss = 0.0

        # Iteriamo sul batch
        for i in range(batch_size):
            idx = int(obj_ids[i])
            
            # --- 1. PREPARAZIONE DATI PATCH ---
            r_patches = pred_r[i].permute(1, 0) # [49, 4]
            t_patches = pred_t[i].permute(1, 0) # [49, 3]
            conf_patches = pred_c[i].view(-1)   # [49]

            gt_r_i = gt_r[i] 
            gt_t_i = gt_t[i] 
            points_i = model_points[i] 

            # --- 2. CALCOLO DELLA LOSS DI TRAINING (Su tutte le patch) ---
            
            # Trasformazione GT
            gt_R_mat = self.quaternion_to_matrix(gt_r_i)
            target_points = torch.mm(points_i, gt_R_mat.T) + gt_t_i 
            
            # Trasformazione Predizioni (Batchata per 49 patch)
            pred_R_mats = self.quaternion_to_matrix(r_patches) 
            points_exp = points_i.unsqueeze(0).expand(num_patches, -1, -1)
            
            # [49, N_PTS, 3] -> Punti ruotati e traslati per ogni patch
            pred_points_transformed = torch.bmm(points_exp, pred_R_mats.permute(0, 2, 1)) + t_patches.unsqueeze(1)
            
            # Calcolo Distanze (ADD / ADD-S)
            target_exp = target_points.unsqueeze(0).expand(num_patches, -1, -1)
            
            if idx in self.sym_list:
                # ADD-S: Distanza dal punto più vicino
                dist_matrix = torch.cdist(pred_points_transformed, target_exp) 
                min_dists, _ = torch.min(dist_matrix, dim=2) 
                loss_per_patch = torch.mean(min_dists, dim=1) 
            else:
                # ADD: Distanza punto-a-punto
                dists = torch.norm(pred_points_transformed - target_exp, dim=2)
                loss_per_patch = torch.mean(dists, dim=1) 

            # DenseFusion Loss Formula
            weighted_loss = loss_per_patch * conf_patches - self.w_rate * torch.log(conf_patches + self.eps)
            total_loss += torch.mean(weighted_loss)

            # --- 3. CALCOLO METRICHE DI LOGGING (Opzionale) ---
            # Calcoliamo l'errore della "Posa Finale" (media pesata) per i grafici
            if return_metrics:
                with torch.no_grad():
                    # 1. Calcola Posa Finale (Weighted Average)
                    # Normalizza pesi confidenza per sommare a 1
                    w_norm = conf_patches / (torch.sum(conf_patches) + self.eps)
                    
                    # Media pesata Rotazione e Traslazione
                    r_final = torch.sum(r_patches * w_norm.unsqueeze(1), dim=0)
                    r_final = F.normalize(r_final, p=2, dim=0) # Rinormalizza quaternione
                    t_final = torch.sum(t_patches * w_norm.unsqueeze(1), dim=0)
                    
                    # 2. Calcola Metriche Fisiche Reali (ADD della posa finale)
                    final_R_mat = self.quaternion_to_matrix(r_final)
                    final_pts = torch.mm(points_i, final_R_mat.T) + t_final
                    
                    if idx in self.sym_list:
                        # ADD-S finale
                        dist_m = torch.cdist(final_pts.unsqueeze(0), target_points.unsqueeze(0))
                        add_error = torch.mean(torch.min(dist_m, dim=2)[0])
                        # Rot Loss approssimata per simmetrici (distanza punti solo ruotati)
                        final_pts_rot = torch.mm(points_i, final_R_mat.T)
                        target_pts_rot = torch.mm(points_i, gt_R_mat.T)
                        dist_m_rot = torch.cdist(final_pts_rot.unsqueeze(0), target_pts_rot.unsqueeze(0))
                        rot_error = torch.mean(torch.min(dist_m_rot, dim=2)[0])
                    else:
                        # ADD finale
                        add_error = torch.mean(torch.norm(final_pts - target_points, dim=1))
                        # Rot Loss (ADD solo rotazione)
                        final_pts_rot = torch.mm(points_i, final_R_mat.T)
                        target_pts_rot = torch.mm(points_i, gt_R_mat.T)
                        rot_error = torch.mean(torch.norm(final_pts_rot - target_pts_rot, dim=1))

                    # Trans Loss (Distanza euclidea vettori)
                    trans_error = torch.norm(t_final - gt_t_i)

                    meta_add_loss += add_error.item()
                    meta_rot_loss += rot_error.item()
                    meta_trans_loss += trans_error.item()

        avg_loss = total_loss / batch_size
        
        if return_metrics:
            return avg_loss, {
                'rot_loss': meta_rot_loss / batch_size,   # Errore Rotazione (in metri sulla superficie)
                'trans_loss': meta_trans_loss / batch_size, # Errore Traslazione (in metri centro-centro)
                'add_loss': meta_add_loss / batch_size    # Errore Totale ADD reale (metri)
            }
        
        return avg_loss