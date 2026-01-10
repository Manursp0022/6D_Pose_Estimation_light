import torch
from torch import nn

class DenseFusionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sym_list = [10, 11] # Eggbox, Glue

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

    def forward(self, pred_r, pred_t, gt_r, gt_t, model_points, obj_ids):
        bs = pred_r.shape[0]
        total_loss = 0.0

        pred_R_mat = self.quaternion_to_matrix(pred_r)
        gt_R_mat = self.quaternion_to_matrix(gt_r)

        # Trasforma punti modello: (R * P) + t
        pred_pts = torch.bmm(pred_R_mat, model_points.permute(0, 2, 1)) + pred_t.unsqueeze(2) 
        gt_pts = torch.bmm(gt_R_mat, model_points.permute(0, 2, 1)) + gt_t.unsqueeze(2)       
        
        pred_pts = pred_pts.permute(0, 2, 1)
        gt_pts = gt_pts.permute(0, 2, 1)

        for i in range(bs):
            idx = int(obj_ids[i])
            if idx in self.sym_list:
                dist_matrix = torch.cdist(pred_pts[i].unsqueeze(0), gt_pts[i].unsqueeze(0))
                min_dist, _ = torch.min(dist_matrix, dim=2) 
                loss_i = torch.mean(min_dist)
            else:
                diff = pred_pts[i] - gt_pts[i]
                dis = torch.norm(diff, dim=1)
                loss_i = torch.mean(dis)
            
            total_loss += loss_i

        return total_loss / bs
