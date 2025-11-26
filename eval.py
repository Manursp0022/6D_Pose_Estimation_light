import argparse
import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Se sei su Colab, matplotlib è il modo migliore per vedere le immagini
def show_image(img, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=5)
    
    args, unknown = parser.parse_known_args()

    print(f"--- load model from: {args.model_path} ---")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Can't find the model here: {args.model_path}")

    model = YOLO(args.model_path)

    # retrieve the validation images from the autosplit file.
    val_txt_path = os.path.join(args.dataset_root, 'autosplit_val.txt')
    if not os.path.exists(val_txt_path):
        raise FileNotFoundError("I can't find autosplit_val.txt. Did you do the training?")
        
    with open(val_txt_path, 'r') as f:
        val_images = f.read().splitlines()
    
    print(f"find {len(val_images)} validation images.")

    selected_imgs = random.sample(val_images, args.num_images)

    print(f"--- View Results ({args.num_images} random) ---")
    
    for img_path in selected_imgs:
        # conf=0.5 means: only show if you are 50% certain
        results = model.predict(img_path, conf=0.5) 
        
        # YOLO has a built-in plot() function that draws boxes
        # Returns a numpy array (BGR image)
        result_plot = results[0].plot()
        
        # show on Colab
        show_image(result_plot, title=f"Predizione su: {os.path.basename(img_path)}")

if __name__ == "__main__":
    evaluate()