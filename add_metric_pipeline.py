
import os
import yaml
import numpy as np
import cv2
import torch
import open3d as o3d

from ultralytics import YOLO

from utils.Posenet_utils.PoseEvaluator import PoseEvaluator
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.utils_geometric import crop_square_resize
from models.Posenet import PoseResNet
from torch.utils.data import DataLoader


def quaternion_to_rotmat(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y- z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x + y*y)]
    ])
    return R

def retrieve_model_points(models3d_dir, class_id):
    # Expect a single PLY named with zero-padded id for single digits, e.g. obj_01.ply, obj_10.ply
    cid = int(class_id)
    filename = f"obj_{cid:02d}.ply"
    path = os.path.join(models3d_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected model PLY not found: {path}")
    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points)
    if pts.size == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    return pts


def add_metric_evaluation(dataset_root,
                          yolo_weights,
                          posenet_ckpt,
                          models_info_yaml,
                          models3d_dir,
                          device='cuda',
                          out_csv=None):
    """
    Evaluate ADD using YOLO detections and PoseNet rotation predictions.

    - YOLO provides bbox and class.
    - PoseEvaluator.estimate_translation uses bbox + object diameter to compute translation.
    - PoseResNet predicts quaternion (w,x,y,z) used to compute rotation matrix.
    - ADD computed between model points transformed by predicted and GT poses.
    """

    device = torch.device(device)

    # load models info (diameters)
    with open(models_info_yaml, 'r') as f:
        models_info = yaml.safe_load(f)

    # default mapping from YOLO class index to LINEMOD folder id (string)
    yolo_to_folder = {
        0: '01', 1: '02', 2: '04', 3: '05', 4: '06', 5: '08',
        6: '09', 7: '10', 8: '11', 9: '12', 10: '13', 11: '14', 12: '15'
    }

    # camera intrinsics: default LINEMOD-like values (user should adapt if needed)
    K = np.array([[572.4114, 0.0, 319.5], [0.0, 573.57043, 239.5], [0.0, 0.0, 1.0]])

    evaluator = PoseEvaluator(K)

    # load Posenet
    posenet = PoseResNet(pretrained=False)
    if os.path.exists(posenet_ckpt):
        state = torch.load(posenet_ckpt, map_location=device)
        # allow state dict or raw
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        posenet.load_state_dict(state)
    posenet.to(device).eval()

    # load YOLO
    yolo = YOLO(yolo_weights)

    results = []

    # Build evaluation dataset using LineModPoseDataset and autosplit_val_ALL.txt
    split_file = os.path.join(dataset_root, 'autosplit_val_ALL.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Validation split file not found: {split_file}")

    val_ds = LineModPoseDataset(split_file, dataset_root, mode='val', img_size=224)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    for batch in val_loader:
        # batch contains: 'image'(tensor preprocessed), 'quaternion'(gt), 'translation'(gt), 'class_id', 'path'
        img_tensor = batch['image'].to(device)  # preprocessed crop (but we'll re-crop from YOLO bbox)
        gt_quat = batch['quaternion'].numpy()[0]
        gt_t = batch['translation'].numpy()[0]
        class_id = int(batch['class_id'].numpy()[0])
        img_path = batch['path'][0]

        # run YOLO on original image path to get bbox and class
        yres = yolo(img_path)
        detections = []
        if len(yres) > 0:
            boxes = yres[0].boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
            try:
                confs = boxes.conf.cpu().numpy()
            except Exception:
                confs = np.zeros(len(xyxy))
            try:
                clss = boxes.cls.cpu().numpy()
            except Exception:
                clss = np.zeros(len(xyxy))
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i]) if len(confs) > i else 0.0
                cls = int(clss[i]) if len(clss) > i else 0
                detections.append(np.array([x1, y1, x2, y2, conf, cls]))

        if len(detections) == 0:
            continue

        # prefer detection matching expected class (map class_id->yolo cls)
        target_folder = f"{class_id:02d}"
        # invert yolo_to_folder
        folder_to_yolo = {v: k for k, v in yolo_to_folder.items()}
        target_yolo_cls = folder_to_yolo.get(target_folder, None)

        chosen_det = None
        # require a YOLO detection matching the GT class; skip otherwise
        if target_yolo_cls is None:
            # couldn't map dataset class to YOLO class, skip sample
            continue
        for det in detections:
            if int(det[5]) == int(target_yolo_cls):
                if chosen_det is None or det[4] > chosen_det[4]:
                    chosen_det = det

        if chosen_det is None:
            # no detection matching the GT class -> skip this sample
            continue

        x1, y1, x2, y2, conf, cls = chosen_det.tolist()
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        # load model points for this class
        model_points = retrieve_model_points(models3d_dir, class_id)
        if model_points is None:
            continue

        # diameter from models_info (models_info uses integer keys)
        try:
            diam_mm = float(models_info[int(class_id)]['diameter'])
        except Exception:
            # fallback: try string key
            diam_mm = float(models_info.get(f"{class_id:02d}", {}).get('diameter', 0.0))

        pred_t = evaluator.estimate_translation(bbox_xywh, diam_mm)

        # crop and run posenet
        img_bgr = cv2.imread(img_path)
        crop = crop_square_resize(img_bgr, bbox_xywh, target_size=224, jitter=False)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_tensor = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        crop_tensor = (crop_tensor - mean) / std

        with torch.no_grad():
            pred_q = posenet(crop_tensor).cpu().numpy().reshape(4,)
        pred_q = pred_q / np.linalg.norm(pred_q)
        pred_R = quaternion_to_rotmat(pred_q)

        # get GT rotation from dataset's quaternion
        gt_q = gt_quat
        gt_R = quaternion_to_rotmat(gt_q)

        add_val = evaluator.calculate_metric(pred_R, pred_t, gt_R, gt_t, model_points, cls)

        results.append({'folder': f"{class_id:02d}", 'image': os.path.basename(img_path), 'add': add_val})

    adds = [r['add'] for r in results]
    if len(adds) > 0:
        mean_add = float(np.mean(adds))
        print(f"Mean ADD over {len(adds)} samples: {mean_add:.4f}")
    else:
        print("No results computed.")

    if out_csv:
        import csv
        with open(out_csv, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=['folder','image','add'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    return results