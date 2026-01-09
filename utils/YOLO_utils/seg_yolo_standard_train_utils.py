import glob
import os
import random
import yaml
from tqdm import tqdm



import cv2
import numpy as np
import os

def mask_to_yolo_polygons(mask_path, class_id=0):
    # 1. Carica la maschera
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""

    # 2. Binarizzazione
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. Trova i contorni
    # RETR_EXTERNAL prende solo il contorno esterno (evita buchi interni)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(mask.shape)
    height, width = mask.shape[0], mask.shape[1]
    yolo_lines = []

    for contour in contours:
        # Opzionale: filtrare contorni troppo piccoli (rumore)
        if cv2.contourArea(contour) < 10:
            continue
        #Using Douglas-Peucker approximation to lower the number of segments in a contour
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        polygon = []
        
        if len(approx) < 3:
            if len(contour) >= 3:
                points_to_use = contour
            else:
                return None
        else:
            points_to_use = approx

        polygon = []
        for point in points_to_use:
            x, y = point[0]
            # Normalizzazione (assicurati che width e height siano float per precisione)
            polygon.append(round(x / width, 6))
            polygon.append(round(y / height, 6))
        
        # Genera la riga solo se abbiamo dati validi
        if polygon:
            line = f"{class_id} " + " ".join(map(str, polygon))
            yolo_lines.append(line)
        
    return "\n".join(yolo_lines)


def create_yolo_labels(dataset_root):
    """
    Generate .txt files for YOLO starting from LINEMOD's gt.yml.
    """
    # Instead of creating a separate “labels” folder, we put the txt files 
    # in the same folder as the images (“rgb”). YOLO will definitely find them.

    all_folders = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
    id_map = {
                1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 
                9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
            }
    #Use the paths in each line of train_split to get image bb gt, and mask
    for folder_id in all_folders:
        #Get to train and val split files
        train_split = os.path.join(dataset_root, "data", folder_id, "train.txt")
        val_split = os.path.join(dataset_root, "data", folder_id, "test.txt")

        #Turn into list of image paths #####.png
        
        train_split = [line.strip()+".png" for line in open(train_split, "r").readlines()]
        val_split = [line.strip()+".png" for line in open(val_split, "r").readlines()]

        
        for line in train_split:
            mask_path = os.path.join(dataset_root, "data", folder_id, "mask", line)
            if not os.path.exists(mask_path):
                print(f"{mask_path} does not exist, skkipping it")
                continue

            mapped_id = id_map[int(folder_id)]
            polygon = mask_to_yolo_polygons(mask_path, mapped_id)
            if polygon:
                txt_path = os.path.join(dataset_root, "data", folder_id, "rgb", line.replace(".png", ".txt"))
                if os.path.exists(txt_path):
                    continue
                else:
                    with open(txt_path, 'w') as f_out:
                        f_out.write(polygon)
        print(f"Training masks created for object {folder_id}")        

    
        for line in val_split:
            mask_path = os.path.join(dataset_root, "data", folder_id, "mask", line)
            if not os.path.exists(mask_path):
                print(f"{mask_path} does not exist, skkipping it")
                continue

            mapped_id = id_map[int(folder_id)]
            polygon = mask_to_yolo_polygons(mask_path, mapped_id)
            if polygon:
                txt_path = os.path.join(dataset_root, "data", folder_id, "rgb", line.replace(".png", ".txt"))
                if os.path.exists(txt_path):
                    continue
                else:
                    with open(txt_path, 'w') as f_out:
                        f_out.write(polygon)
        print(f"Validation masks created for object {folder_id}")
    

def create_yolo_config_all(dataset_root):
    """
    Strategia STRATIFICATA:
    1. Itera su ogni cartella.
    2. Divide 15/85 le immagini DI QUELLA CARTELLA.
    3. Accumula nelle liste globali Train e Val.
    4. Mischia le liste globali alla fine.
    
    Garantisce che ogni oggetto sia rappresentato equamente sia in Train che in Val.
    """
    all_folders = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
    
    # Liste globali (i "sacchi")
    global_train_list = []
    global_val_list = []

    print(f"[INFO] Inizio splitting stratificato su {len(all_folders)} cartelle...")

    for folder_id in all_folders:
        images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
        train_split = os.path.join(dataset_root, 'data', folder_id, 'train.txt')
        val_split = os.path.join(dataset_root, 'data', folder_id, 'test.txt')
        # Trova immagini png
        with open(train_split, 'r') as f:
            train_imgs = [line.strip()+".png" for line in f.readlines()]
        with open(val_split, 'r') as f:
            val_imgs = [line.strip()+".png" for line in f.readlines()]

        imgs = glob.glob(os.path.join(images_dir, "*.png"))
        
        # Filtra solo quelle che hanno la label .txt generata (sicurezza)
        valid_train_imgs = [os.path.join(images_dir, img) for img in train_imgs if os.path.exists(os.path.join(images_dir, img)) and os.path.exists(os.path.join(images_dir, img.replace('.png', '.txt')))]
        valid_val_imgs = [os.path.join(images_dir, img) for img in val_imgs if os.path.exists(os.path.join(images_dir, img)) and os.path.exists(os.path.join(images_dir, img.replace('.png', '.txt')))]
        
        if not valid_train_imgs:
            print(f"Warning: No valid train images in {folder_id}")
            continue
        
        if not valid_val_imgs:
            print(f"Warning: No valid val images in {folder_id}")
            continue
        global_train_list.extend(valid_train_imgs)
        global_val_list.extend(valid_val_imgs)


    train_path = os.path.join(dataset_root, 'standard_seg_train_split.txt')
    val_path = os.path.join(dataset_root, 'standard_seg_val_split.txt')

    with open(train_path, 'w') as f:
        for item in global_train_list:
            f.write(f"{item}\n")
    with open(val_path, 'w') as f:
        for item in global_val_list:
            f.write(f"{item}\n")


    config = {
        'path': dataset_root, 
        'train': train_path,
        'val': val_path,
        'names': {
            0: 'Ape', 1: 'Benchvise', 2: 'Cam', 3: 'Can', 4: 'Cat', 
            5: 'Driller', 6: 'Duck', 7: 'Eggbox', 8: 'Glue', 
            9: 'Holepuncher', 10: 'Iron', 11: 'Lamp', 12: 'Phone'
        }
    }
    
    config_path = os.path.join(dataset_root, 'linemod_yolo_config_standard.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"[INFO] Configurazione salvata in: {config_path}")

    return config_path


if __name__=="__main__":
    dataset_root = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed"



    