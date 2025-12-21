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


def create_yolo_labels(dataset_root, train_split, val_split):
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
    with open(train_split, "r") as file:
        for line in file:
            line = line.strip()
            img_num_format, obj_id = line.split("/")[-1], line.split("/")[-3]
            img_num = int(img_num_format.split(".")[0])
            img_num =f"{img_num:04d}"
            mask_path = os.path.join(dataset_root, "data", obj_id, "mask", img_num_format)
            if not os.path.exists(mask_path):
                print(f"{mask_path} does not exist, skkipping it")
                continue
            
            mapped_id = id_map[int(obj_id)]
            polygon = mask_to_yolo_polygons(mask_path, mapped_id)
            if polygon:
                txt_path = os.path.join(dataset_root, "data", obj_id, "rgb", img_num_format.replace(".png", ".txt"))
                if os.path.exists(txt_path):
                    print(f"txt file already exists, skipping it")
                else:
                    with open(txt_path, 'w') as f_out:
                        f_out.write(polygon)
    print(f"Training masks created")        

    with open(val_split, "r") as file:
        for line in file:
            line = line.strip()
            img_num_format, obj_id = line.split("/")[-1], line.split("/")[-3]
            img_num = int(img_num_format.split(".")[0])
            img_num =f"{img_num:04d}"
            mask_path = os.path.join(dataset_root, "data", obj_id, "mask", img_num_format)
            if not os.path.exists(mask_path):
                print(f"{mask_path} does not exist, skkipping it")
                continue
            mapped_id = id_map[int(obj_id)]
            polygon = mask_to_yolo_polygons(mask_path, mapped_id)
            if polygon:
                txt_path = os.path.join(dataset_root, "data", obj_id, "rgb", img_num_format.replace(".png", ".txt"))
                if os.path.exists(txt_path):
                    print(f"txt file already exists, skipping it")
                else:
                    with open(txt_path, 'w') as f_out:
                        f_out.write(polygon)
    print("Validation masks created")
    

def create_yolo_config_all(dataset_root, train_split, val_split):
    """
    Strategia STRATIFICATA:
    1. Itera su ogni cartella.
    2. Divide 80/20 le immagini DI QUELLA CARTELLA.
    3. Accumula nelle liste globali Train e Val.
    4. Mischia le liste globali alla fine.
    
    Garantisce che ogni oggetto sia rappresentato equamente sia in Train che in Val.
    """
    
    def is_valid_image(dataset_root, line):
        img_path = line.split("/")[-4:]
        path= os.path.join(dataset_root, img_path[0], img_path[1], img_path[2], img_path[3].split(".")[0]+".txt")
        img = os.path.join(dataset_root, img_path[0], img_path[1], img_path[2], img_path[3])
        return path, img

    valid_train = []
    valid_val = []
    with open(train_split, "r") as file:
        for line in file:
            path, img = is_valid_image(dataset_root, line)
            #print(f"checking this path: {path}")
            if os.path.exists(path):
                valid_train.append(img)
        
    print(len(valid_train))

    with open(val_split, "r") as file:
         for line in file:
            path, img = is_valid_image(dataset_root, line)
            if os.path.exists(path):
                valid_val.append(img)
            else:
                print(f"Path {path} not existing")
        
    print(len(valid_val))


    valid_train_paths = os.path.join(dataset_root, "valid_train_paths.txt")
    with open(valid_train_paths, "w") as file:
        for line in valid_train:
            file.write(line)


    valid_val_paths = os.path.join(dataset_root, "valid_val_paths.txt")
    with open(valid_val_paths, "w")as file:
        for line in valid_val: 
            file.write(line)


    config = {
        'path': dataset_root, 
        'train': valid_train_paths,
        'val': valid_val_paths,
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


if __name__=="__main__":
    dataset_root = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed"
    train_split = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\data\\autosplit_train_ALL.txt"
    val_split = "C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\data\\autosplit_val_ALL.txt"

    