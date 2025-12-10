import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import posenet_dataset
import posenet
import quaternion_Loss
from posenet_dataset import LineModPoseDataset
from posenet import PoseResNet
from quaternion_loss import QuaternionLoss
# Parametri


BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Standard per fine-tuning ResNet
EPOCHS = 20          # Bastano poche epoche se il dataset è piccolo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/content/drive/MyDrive/Linemod_Pose_Training_Data/weights"
os.makedirs(SAVE_DIR, exist_ok=True)


# Assicurati di aver definito la classe LineModPoseDataset nella cella precedente!
dataset_root = "/content/drive/MyDrive/Linemod_Pose_Training_Data" # Dove hai salvato i crop
# Nota: Usiamo gli stessi file txt di split di YOLO per coerenza
split_train = "/content/drive/MyDrive/Linemod_preprocessed/autosplit_train_ALL.txt"
split_val = "/content/drive/MyDrive/Linemod_preprocessed/autosplit_val_ALL.txt"

# Inizializza Dataset
train_ds = LineModPoseDataset(split_train, dataset_root, mode='train')
val_ds = LineModPoseDataset(split_val, dataset_root, mode='val')

# DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Dati caricati: {len(train_ds)} Train, {len(val_ds)} Val")

# Assicurati di aver definito PoseResNet e QuaternionLoss nella cella precedente!
model = PoseResNet(pretrained=True).to(DEVICE)
criterion = QuaternionLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

print("Starting Training PoseNet...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for batch in progress_bar:
        images = batch['image'].to(DEVICE)
        gt_quats = batch['quaternion'].to(DEVICE)

        optimizer.zero_grad()

        pred_quats = model(images) # Ritorna solo i quaternioni ora!

        loss = criterion(pred_quats, gt_quats)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(DEVICE)
            gt_quats = batch['quaternion'].to(DEVICE)

            pred_quats = model(images)
            loss = criterion(pred_quats, gt_quats)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f" New Best :  Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(" Best Model Saved!")


model.load_state_dict(best_model_state)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_posenet.pth'))

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("PoseNet Training (Rotation Only)")
plt.show()