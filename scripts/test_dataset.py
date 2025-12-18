import os
import sys

# Ensure project root is on sys.path so imports like `utils...` work when
# running this script directly from the project root or from other folders.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
val = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\autosplit_train_ALL.txt"
train = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\autosplit_val_ALL.txt"  
# Adjust this path to your local dataset folder (the parent that contains autosplit_train_ALL.txt)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'dataset', 'Linemod_preprocessed')
DATA_ROOT = os.path.abspath(DATA_ROOT)

print('Using dataset root:', DATA_ROOT)

# Try with empty split -> loader will look for autosplit files under dataset root
try:
    ds = LineModPoseDataset('', DATA_ROOT, mode='train')
    print('Total samples:', len(ds))
    if len(ds) > 0:
        s = ds[0]
        print('Sample keys:', list(s.keys()))
        print('Sample img path:', s.get('path'))
except Exception as e:
    print('Error instantiating dataset:', e)
