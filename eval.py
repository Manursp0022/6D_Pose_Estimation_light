import argparse
import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def show_image(img, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help="Path del dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path del file .pt allenato")
    parser.add_argument('--num_images', type=int, default=5, help="Quante immagini testare")
    
    args, unknown = parser.parse_known_args()

    print(f"--- Caricamento Modello da: {args.model_path} ---")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Non trovo il modello in: {args.model_path}")

    model = YOLO(args.model_path)

    val_txt_path = os.path.join(args.dataset_root, 'autosplit_val.txt')
    if not os.path.exists(val_txt_path):
        raise FileNotFoundError("Non trovo autosplit_val.txt. Hai fatto il training?")
        
    with open(val_txt_path, 'r') as f:
        val_images = f.read().splitlines()
    
    print(f"Trovate {len(val_images)} immagini di validazione.")

    selected_imgs = random.sample(val_images, args.num_images)

    print(f"--- Visualizzazione Risultati ({args.num_images} random) ---")
    
    for img_path in selected_imgs:
        results = model.predict(img_path, conf=0.5) 
        result_plot = results[0].plot()

        show_image(result_plot, title=f"Predizione su: {os.path.basename(img_path)}")

if __name__ == "__main__":
    evaluate()