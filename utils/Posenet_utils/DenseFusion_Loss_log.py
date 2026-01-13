import torch
import torch.nn as nn
import torch.optim as optim

class DenseFusionLoss(nn.Module):
    """
    Loss migliorata con opzione di weighting e metriche separate per debugging.
    - use_weighted: False = ADD standard (come paper), True = weighted loss
    """
    def __init__(self, device, rot_weight=1.0, trans_weight=0.3, use_weighted=False):
        super().__init__()
        self.device = device
        self.sym_list = [10, 11]  # Eggbox, Glue
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.use_weighted = use_weighted

    def quaternion_to_matrix(self, quaternions):
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
            ), -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def forward(self, pred_r, pred_t, gt_r, gt_t, model_points, obj_ids, return_metrics=False):
        bs = pred_r.shape[0]
        
        pred_R_mat = self.quaternion_to_matrix(pred_r)
        gt_R_mat = self.quaternion_to_matrix(gt_r)

        # Trasforma punti con rotazione predetta e GT
        pred_pts = torch.bmm(pred_R_mat, model_points.permute(0, 2, 1)) + pred_t.unsqueeze(2)
        gt_pts = torch.bmm(gt_R_mat, model_points.permute(0, 2, 1)) + gt_t.unsqueeze(2)
        
        pred_pts = pred_pts.permute(0, 2, 1)
        gt_pts = gt_pts.permute(0, 2, 1)

        total_loss = 0.0
        rot_loss_sum = 0.0
        trans_loss_sum = 0.0
        
        for i in range(bs):
            idx = int(obj_ids[i])
            
            if idx in self.sym_list:
                # ADD-S per oggetti simmetrici (standard)
                dist_matrix = torch.cdist(pred_pts[i].unsqueeze(0), gt_pts[i].unsqueeze(0))
                min_dist, _ = torch.min(dist_matrix, dim=2)
                loss_add = torch.mean(min_dist)
            else:
                # ADD standard (come nel paper)
                diff = pred_pts[i] - gt_pts[i]
                loss_add = torch.mean(torch.norm(diff, dim=1))
            
            if return_metrics or self.use_weighted:
                pred_pts_rot = torch.bmm(pred_R_mat[i:i+1], model_points[i:i+1].permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)
                gt_pts_rot = torch.bmm(gt_R_mat[i:i+1], model_points[i:i+1].permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)
                
                if idx in self.sym_list:
                    dist_matrix_rot = torch.cdist(pred_pts_rot.unsqueeze(0), gt_pts_rot.unsqueeze(0))
                    min_dist_rot, _ = torch.min(dist_matrix_rot, dim=2)
                    loss_rot = torch.mean(min_dist_rot)
                else:
                    diff_rot = pred_pts_rot - gt_pts_rot
                    loss_rot = torch.mean(torch.norm(diff_rot, dim=1))
                
                # Translation loss
                loss_trans = torch.norm(pred_t[i] - gt_t[i])
                
                rot_loss_sum += loss_rot.item()
                trans_loss_sum += loss_trans.item()
            

            if self.use_weighted:
                # Loss pesata (sperimentale)
                loss_i = self.rot_weight * loss_rot + self.trans_weight * loss_trans
            else:
                # Loss standard ADD (come nel paper)
                loss_i = loss_add
            
            total_loss += loss_i

        avg_loss = total_loss / bs
        
        if return_metrics:
            return avg_loss, {
                'rot_loss': rot_loss_sum / bs,
                'trans_loss': trans_loss_sum / bs,
                'add_loss': avg_loss.item()  # La loss ADD standard
            }
        
        return avg_loss