import torch
import torch.optim as optim
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import pynvml
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.RGB_D_ResNet import PoseResNetRGBD
from utils.Posenet_utils.quaternion_Loss import QuaternionLoss
from utils.Posenet_utils.utils_geometric import crop_square_resize, image_transformation
import ultralytics
import cv2
import torchvision.transforms as transforms
import random
import numpy as np

if __name__ == "__main__":
    dataset_root= 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed'
    split_val= 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed\\autosplit_val_ALL.txt'
    yolo_weights= 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\checkpoints\\best_YOLO.pt'

    # Use the validation dataset (we need original paths + GT bbox)
    val_dataset = LineModPoseDataset(split_file=split_val, dataset_root=dataset_root, mode='val', img_size=224)

    # Load YOLO (CPU fallback if CUDA unavailable)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = ultralytics.YOLO(yolo_weights).to(device)

    # Mapping from YOLO class indices to Linemod folder ids (same as PipelineTrainer)
    yolo_to_folder = {
        0: '01', 1: '02', 2: '04', 3: '05', 4: '06', 5: '08',
        6: '09', 7: '10', 8: '11', 9: '12', 10: '13', 11: '14', 12: '15'
    }

    out_dir = os.path.join('.', 'debug_gt_vs_yolo')
    os.makedirs(out_dir, exist_ok=True)

    # Number of examples to visualize
    n_examples = 8
    # Pick random sample indices from validation set
    rng = random.Random(42)
    indices = rng.sample(range(len(val_dataset)), min(n_examples, len(val_dataset)))

    for ii, idx in enumerate(indices):
        sample = val_dataset.samples[idx]
        img_path = sample['img_path']
        gt_bbox = sample['bbox']  # [x, y, w, h]
        gt_obj_id = sample['obj_id']

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image {img_path}; skipping")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create GT crop
        gt_crop = crop_square_resize(img_rgb, gt_bbox, target_size=224, is_depth=False)

        # Run YOLO on the single image path
        results = yolo(img_path, verbose=False)
        res = results[0]
        detections = res.boxes

        yolo_crop = None
        yolo_info = None

        if len(detections) == 0:
            print(f"YOLO: no detections for {img_path}")
        else:
            # Find candidate detections matching GT class
            candidates = []
            for k in range(len(detections)):
                try:
                    cls_idx = int(detections.cls[k].item()) if hasattr(detections.cls[k], 'item') else int(detections.cls[k])
                except Exception:
                    cls_idx = int(detections.cls[k])

                mapped_folder = yolo_to_folder.get(cls_idx, None)
                mapped_int = int(mapped_folder) if mapped_folder is not None else cls_idx

                if mapped_int == int(gt_obj_id):
                    candidates.append(k)

            if len(candidates) == 0:
                print(f"YOLO: no detection matching GT class {gt_obj_id} for {img_path}")
            else:
                # pick highest-confidence among candidates
                confidences = detections.conf
                best_idx = candidates[0]
                best_conf = confidences[best_idx]
                for c in candidates[1:]:
                    if confidences[c] > best_conf:
                        best_idx = c
                        best_conf = confidences[c]

                # YOLO returns xywh as (cx,cy,w,h) in pixels — convert to top-left x,y,w,h
                bbox_xywh = detections.xywh[best_idx]
                cx = float(bbox_xywh[0])
                cy = float(bbox_xywh[1])
                w = float(bbox_xywh[2])
                h = float(bbox_xywh[3])
                x_tl = cx - (w / 2.0)
                y_tl = cy - (h / 2.0)
                yolo_bbox_tl = [x_tl, y_tl, w, h]

                yolo_crop = crop_square_resize(img_rgb, yolo_bbox_tl, target_size=224, is_depth=False)
                yolo_info = {
                    'conf': float(best_conf), 'cls_idx': int(detections.cls[best_idx])
                }

        # Compose visualization: original with boxes, GT crop, YOLO crop (if present)
        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        # Original image with drawn boxes
        img_disp = img_rgb.copy()
        # draw GT box (green)
        xg, yg, wg, hg = map(int, gt_bbox)
        cv2.rectangle(img_disp, (xg, yg), (xg+wg, yg+hg), (0,255,0), 2)
        cv2.putText(img_disp, f"GT: {gt_obj_id}", (xg, max(yg-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # draw YOLO box (red) if available
        if yolo_crop is not None and yolo_info is not None:
            xr, yr, wr, hr = map(int, yolo_bbox_tl)
            cv2.rectangle(img_disp, (xr, yr), (xr+wr, yr+hr), (255,0,0), 2)
            cv2.putText(img_disp, f"YOLO: {yolo_info['conf']:.2f}", (xr, max(yr-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        axs[0].imshow(img_disp)
        axs[0].axis('off')
        axs[0].set_title('Original (GT green, YOLO red)')

        axs[1].imshow(gt_crop)
        axs[1].axis('off')
        axs[1].set_title('GT Crop')

        if yolo_crop is not None:
            axs[2].imshow(yolo_crop)
            axs[2].set_title('YOLO Crop')
        else:
            axs[2].imshow(255 * np.ones((224,224,3), dtype=np.uint8))
            axs[2].set_title('YOLO Crop (none)')
        axs[2].axis('off')

        plt.suptitle(f"Sample {ii+1} - Obj {gt_obj_id}")
        out_path = os.path.join(out_dir, f"sample_{ii+1:02d}_obj{gt_obj_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {out_path}")
        plt.show()

    print('Done')

