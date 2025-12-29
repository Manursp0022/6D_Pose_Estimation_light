import cv2
import numpy as np
import os

# === CONFIGURA QUESTI PATH ===
DATASET_ROOT = "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed"  # <-- MODIFICA QUESTO
FOLDER_ID = "02"
IMG_NAME = "0000.png"
TARGET_OBJ_ID = 1  # Oggetto da isolare

# Mappatura completa
OBJID_TO_MASK_PIXEL = {
    1: 21,
    2: 43,
    5: 106,
    6: 128,
    8: 170,
    9: 191,
    10: 213,
    11: 234,
    12: 255
}

# Path
mask_all_path = os.path.join(DATASET_ROOT, "data", FOLDER_ID, "mask_all", IMG_NAME)
rgb_path = os.path.join(DATASET_ROOT, "data", FOLDER_ID, "rgb", IMG_NAME)
output_dir = "./test_mask_output"

os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print(f"TEST ESTRAZIONE MASCHERA - Oggetto {TARGET_OBJ_ID}")
print("=" * 60)

# Carica immagini
mask_raw = cv2.imread(mask_all_path, cv2.IMREAD_GRAYSCALE)
rgb = cv2.imread(rgb_path)

if mask_raw is None:
    print(f"[ERRORE] Non riesco a caricare: {mask_all_path}")
    exit(1)

if rgb is None:
    print(f"[WARNING] Non riesco a caricare RGB: {rgb_path}")
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

print(f"\nMaschera caricata: {mask_all_path}")
print(f"Shape: {mask_raw.shape}")
print(f"Valori unici: {np.unique(mask_raw).tolist()}")

# Estrai maschera per l'oggetto target
target_pixel_value = OBJID_TO_MASK_PIXEL[TARGET_OBJ_ID]
print(f"\nOggetto {TARGET_OBJ_ID} -> Valore pixel: {target_pixel_value}")

# Crea maschera binaria
mask_isolated = np.where(mask_raw == target_pixel_value, 255, 0).astype(np.uint8)

# Conta pixel
num_pixels = np.sum(mask_isolated == 255)
print(f"Pixel isolati: {num_pixels}")

# Salva risultati
# 1. Maschera originale (tutti gli oggetti)
cv2.imwrite(os.path.join(output_dir, "1_mask_all_original.png"), mask_raw)

# 2. Maschera isolata (solo oggetto target)
cv2.imwrite(os.path.join(output_dir, f"2_mask_obj{TARGET_OBJ_ID}_isolated.png"), mask_isolated)

# 3. RGB originale
cv2.imwrite(os.path.join(output_dir, "3_rgb_original.png"), rgb)

# 4. RGB con overlay della maschera isolata (per verifica visiva)
rgb_overlay = rgb.copy()
# Colora in verde i pixel dell'oggetto
rgb_overlay[mask_isolated == 255] = [0, 255, 0]  # Verde
# Blend con originale per trasparenza
rgb_blended = cv2.addWeighted(rgb, 0.6, rgb_overlay, 0.4, 0)
cv2.imwrite(os.path.join(output_dir, f"4_rgb_overlay_obj{TARGET_OBJ_ID}.png"), rgb_blended)

# 5. Crop dell'oggetto (solo la regione dell'oggetto)
# Trova bounding box della maschera
coords = np.where(mask_isolated == 255)
if len(coords[0]) > 0:
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Aggiungi padding
    pad = 10
    y_min = max(0, y_min - pad)
    y_max = min(mask_isolated.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(mask_isolated.shape[1], x_max + pad)
    
    crop_rgb = rgb[y_min:y_max, x_min:x_max]
    crop_mask = mask_isolated[y_min:y_max, x_min:x_max]
    
    cv2.imwrite(os.path.join(output_dir, f"5_crop_rgb_obj{TARGET_OBJ_ID}.png"), crop_rgb)
    cv2.imwrite(os.path.join(output_dir, f"6_crop_mask_obj{TARGET_OBJ_ID}.png"), crop_mask)
    
    print(f"\nBounding box oggetto: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

print(f"\n{'=' * 60}")
print(f"Output salvati in: {output_dir}/")
print(f"{'=' * 60}")
print("""
File generati:
  1_mask_all_original.png     - Maschera originale con tutti gli oggetti
  2_mask_obj{N}_isolated.png  - Maschera binaria solo oggetto target
  3_rgb_original.png          - Immagine RGB originale
  4_rgb_overlay_obj{N}.png    - RGB con overlay verde sull'oggetto
  5_crop_rgb_obj{N}.png       - Crop RGB dell'oggetto
  6_crop_mask_obj{N}.png      - Crop maschera dell'oggetto

Verifica visivamente che l'overlay verde corrisponda all'oggetto 9!
""")

# === TEST TUTTI GLI OGGETTI ===
print("\n" + "=" * 60)
print("TEST RAPIDO SU TUTTI GLI OGGETTI")
print("=" * 60)

for obj_id, pixel_val in OBJID_TO_MASK_PIXEL.items():
    mask_obj = np.where(mask_raw == pixel_val, 255, 0).astype(np.uint8)
    num_px = np.sum(mask_obj == 255)
    print(f"  Obj {obj_id:2d} (pixel={pixel_val:3d}): {num_px:6d} pixel")
    
    # Salva anche queste
    cv2.imwrite(os.path.join(output_dir, f"all_obj{obj_id:02d}_mask.png"), mask_obj)

print(f"\nTutte le maschere individuali salvate in {output_dir}/")
