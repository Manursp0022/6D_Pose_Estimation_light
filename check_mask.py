"""
Script per analizzare i valori dei pixel nelle maschere mask_all di LineMOD.
Obiettivo: capire la mappatura tra valori pixel e obj_id.
"""

import cv2
import numpy as np
import yaml
import os
from collections import Counter

# === CONFIGURA QUESTI PATH ===
DATASET_ROOT = "/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed"  # <-- MODIFICA QUESTO
FOLDER_ID = "02"

# Path alle cartelle
mask_all_dir = os.path.join(DATASET_ROOT, "data", FOLDER_ID, "mask_all")
gt_path = os.path.join(DATASET_ROOT, "data", FOLDER_ID, "gt.yml")

# Carica il ground truth
print("=" * 60)
print("ANALISI MASCHERE MASK_ALL - FOLDER 02")
print("=" * 60)

with open(gt_path, 'r') as f:
    gt_data = yaml.safe_load(f)

# Analizza alcune immagini campione
sample_images = [0, 1, 2, 5, 6, 10, 100]  # Indici immagini da analizzare

for img_id in sample_images:
    mask_path = os.path.join(mask_all_dir, f"{img_id:04d}.png")
    
    if not os.path.exists(mask_path):
        print(f"\n[!] Maschera {mask_path} non trovata, skip...")
        continue
    
    # Carica maschera
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"\n[!] Errore nel caricare {mask_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"IMMAGINE {img_id:04d}.png")
    print(f"{'='*60}")
    
    # Trova tutti i valori unici nella maschera (escluso 0 = sfondo)
    unique_values = np.unique(mask)
    unique_non_zero = unique_values[unique_values > 0]
    
    print(f"\nValori pixel unici nella maschera: {unique_values.tolist()}")
    print(f"Valori non-zero (oggetti): {unique_non_zero.tolist()}")
    
    # Conta pixel per ogni valore
    print("\nDistribuzione pixel per valore:")
    value_counts = Counter(mask.flatten())
    for val in sorted(value_counts.keys()):
        count = value_counts[val]
        percentage = (count / mask.size) * 100
        label = "sfondo" if val == 0 else f"oggetto?"
        print(f"  Valore {val:3d}: {count:7d} pixel ({percentage:5.2f}%) - {label}")
    
    # Confronta con GT
    if img_id in gt_data:
        objs_in_frame = gt_data[img_id]
        gt_obj_ids = [obj['obj_id'] for obj in objs_in_frame]
        print(f"\nObj_id dal GT per questa immagine: {gt_obj_ids}")
        
        # Verifica corrispondenza
        print("\n--- VERIFICA CORRISPONDENZA ---")
        mask_values_set = set(unique_non_zero.tolist())
        gt_ids_set = set(gt_obj_ids)
        
        if mask_values_set == gt_ids_set:
            print("✓ PERFETTO! I valori pixel corrispondono esattamente agli obj_id")
        else:
            print(f"  Valori maschera: {sorted(mask_values_set)}")
            print(f"  Obj_id nel GT:   {sorted(gt_ids_set)}")
            
            in_mask_not_gt = mask_values_set - gt_ids_set
            in_gt_not_mask = gt_ids_set - mask_values_set
            
            if in_mask_not_gt:
                print(f"  [!] Valori in maschera ma NON in GT: {in_mask_not_gt}")
            if in_gt_not_mask:
                print(f"  [!] Obj_id in GT ma NON in maschera: {in_gt_not_mask}")
                
            # Potrebbe essere che i valori siano indici (0,1,2...) invece di obj_id
            if len(unique_non_zero) == len(gt_obj_ids):
                print("\n  [?] IPOTESI: I valori potrebbero essere INDICI (0-based o 1-based)")
                print(f"      Valori maschera ordinati: {sorted(unique_non_zero.tolist())}")
                print(f"      Obj_id GT ordinati:       {sorted(gt_obj_ids)}")
                
                # Prova mappatura per indice
                sorted_mask_vals = sorted(unique_non_zero.tolist())
                sorted_gt_ids = sorted(gt_obj_ids)
                print("\n      Possibile mappatura per posizione:")
                for i, (mv, gid) in enumerate(zip(sorted_mask_vals, sorted_gt_ids)):
                    print(f"        mask_val {mv} -> obj_id {gid}")
    else:
        print(f"\n[!] Immagine {img_id} non presente nel GT")

print("\n" + "=" * 60)
print("ANALISI COMPLETATA")
print("=" * 60)
print("""
INTERPRETAZIONE:
- Se i valori pixel == obj_id: usa `mask = np.where(mask_raw == obj_id, 255, 0)`
- Se i valori sono indici: dovrai mappare indice -> obj_id
- Se i valori sono livelli di grigio arbitrari: serve analisi più approfondita
""")