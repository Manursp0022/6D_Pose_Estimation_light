import glob
import os
import random
import yaml
from tqdm import tqdm

def create_yolo_labels(dataset_root):
    """
    Generate .txt files for YOLO starting from LINEMOD's gt.yml.
    """
    # Instead of creating a separate “labels” folder, we put the txt files 
    # in the same folder as the images (“rgb”). YOLO will definitely find them.

    all_folders = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']

    for folder_id in all_folders:

        labels_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
        images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
        yaml_path = os.path.join(dataset_root, 'data', folder_id, 'gt.yml')

        # Quick check: let's count the .txt files in the rgb folder
        txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        if len(txt_files) > 100:
            print(f"[INFO] Labels già presenti in {labels_dir}. Salto generazione.")
        else:
            print(f"[INFO] Generazione labels YOLO da {yaml_path}...")
            
            # Map ID LINEMOD -> ID YOLO
            id_map = {
                1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 
                9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
            }

            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            count = 0
            for img_id, obj_list in tqdm(data.items(), desc="Creazione .txt"):
                filename = f"{img_id:04d}"
                
                # Paths
                img_path = os.path.join(images_dir, filename + ".png")
                txt_path = os.path.join(labels_dir, filename + ".txt")
                
                if not os.path.exists(img_path): continue
                #hard coded to save time
                w_img, h_img = 640, 480 

                yolo_lines = []
                target_id = int(folder_id)

                for obj in obj_list:
                    if obj['obj_id'] != target_id: continue

                    obj_id = obj['obj_id']
                    if obj_id not in id_map: continue

                    x_min, y_min, w_box, h_box = obj['obj_bb']

                    x_center = (x_min + w_box / 2) / w_img
                    y_center = (y_min + h_box / 2) / h_img
                    w_norm = w_box / w_img
                    h_norm = h_box / h_img

                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))

                    yolo_lines.append(f"{id_map[obj_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    break

                if yolo_lines:
                            with open(txt_path, 'w') as f_out:
                                f_out.write("\n".join(yolo_lines))
                                count += 1
            print(f"Labels generated in {labels_dir}.")

def create_yolo_config_all(dataset_root):
    """
    Strategia STRATIFICATA:
    1. Itera su ogni cartella.
    2. Divide 80/20 le immagini DI QUELLA CARTELLA.
    3. Accumula nelle liste globali Train e Val.
    4. Mischia le liste globali alla fine.
    
    Garantisce che ogni oggetto sia rappresentato equamente sia in Train che in Val.
    """
    # Lista di tutte le cartelle
    all_folders = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
    
    # Liste globali (i "sacchi")
    global_train_list = []
    global_val_list = []

    print(f"[INFO] Inizio splitting stratificato su {len(all_folders)} cartelle...")

    for folder_id in all_folders:
        images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
        
        # Trova immagini png
        imgs = glob.glob(os.path.join(images_dir, "*.png"))
        
        # Filtra solo quelle che hanno la label .txt generata (sicurezza)
        valid_imgs = [img for img in imgs if os.path.exists(img.replace('.png', '.txt'))]
        
        if not valid_imgs:
            print(f"Warning: Nessuna immagine valida in {folder_id}")
            continue

        # --- STEP CRUCIALE: Mischia e Taglia ORA, per questa singola cartella ---
        # Usiamo un seed fisso per riproducibilità
        random.seed(42) 
        random.shuffle(valid_imgs)
        
        # Calcola il punto di taglio per questa cartella (es. su 1200 immagini -> 960 train, 240 val)
        split_idx = int(len(valid_imgs) * 0.8)
        
        # Aggiungi ai sacchi globali
        global_train_list.extend(valid_imgs[:split_idx])
        global_val_list.extend(valid_imgs[split_idx:])
        
        print(f"   Folder {folder_id}: {len(valid_imgs)} imgs -> {split_idx} Train / {len(valid_imgs)-split_idx} Val")

    # --- STEP FINALE: Mischia i sacchi globali ---
    # Ora che abbiamo raccolto tutto, mischiamo l'ordine per il training
    # così il modello non vede "tutte le api" poi "tutte le morse".
    random.shuffle(global_train_list)
    random.shuffle(global_val_list)

    print(f"[INFO] TOTALE: {len(global_train_list)} Train, {len(global_val_list)} Validation.")

    # Scrittura su file
    train_list_path = os.path.join(dataset_root, 'autosplit_train_ALL.txt')
    val_list_path = os.path.join(dataset_root, 'autosplit_val_ALL.txt')
    
    with open(train_list_path, 'w') as f:
        f.write('\n'.join(global_train_list))
        
    with open(val_list_path, 'w') as f:
        f.write('\n'.join(global_val_list))
        
    # Config YAML
    config = {
        'path': dataset_root, 
        'train': train_list_path,
        'val': val_list_path,
        'names': {
            0: 'Ape', 1: 'Benchvise', 2: 'Cam', 3: 'Can', 4: 'Cat', 
            5: 'Driller', 6: 'Duck', 7: 'Eggbox', 8: 'Glue', 
            9: 'Holepuncher', 10: 'Iron', 11: 'Lamp', 12: 'Phone'
        }
    }
    
    config_path = os.path.join(dataset_root, 'linemod_yolo_config_ALL.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"[INFO] Configurazione salvata in: {config_path}")
    return config_path