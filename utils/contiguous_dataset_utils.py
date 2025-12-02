import os
import yaml
import logging
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageChops

#Colab download function
def download_dataset():
    try:
        import gdown
    except ImportError:
        print("gdown not found, installing...")
        !pip install gdown -qq
        import gdown

    # Google Drive file ID for Linemod_preprocessed.zip
    # PLEASE REPLACE 'YOUR_FILE_ID_HERE' with the actual file ID from your Google Drive.
    # Ensure the file is shared with 'Anyone with the link' permission for gdown to work.
    file_id = '1gO4VKeQzPXWBtC1u-Oy6o6TxSuB3JU_I'

    # Destination directory for the downloaded file
    output_dir = '/content/6D_Pose_Estimation_light/dataset/'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading Linemod_preprocessed.zip from Google Drive file ID: {file_id} into '{output_dir}'...")

    # Download the specific file
    !gdown --id {file_id} -O {output_dir}Linemod_preprocessed.zip

    print("Download complete.")

    print(f"Contents of the downloaded folder in '{output_dir}':")

    !unzip /content/6D_Pose_Estimation_light/dataset/DenseFusion/Linemod_preprocessed.zip -d /content/6D_Pose_Estimation_light/dataset/

def rename_rgb_items(base_dataset_path, output_directory_path):
    global_counter = 0
    print("Function 'rename_rgb_items' started.")

    # Handle output directory: clear if not empty, then ensure it exists
    if os.path.exists(output_directory_path):
        if len(os.listdir(output_directory_path)) > 0:
            print(f"Output directory '{output_directory_path}' is not empty. Clearing contents...")
            shutil.rmtree(output_directory_path)
            os.makedirs(output_directory_path)
            print(f"Output directory '{output_directory_path}' cleared and recreated.")
        else:
            print(f"Output directory '{output_directory_path}' already exists and is empty. Proceeding.")
    else:
        os.makedirs(output_directory_path)
        print(f"Output directory '{output_directory_path}' created.")

    print(f"Output directory '{output_directory_path}' prepared for copying.")

    # Iterate through object IDs from 1 to 15 to match verification and gt processing
    for obj_id in range(1, 16):
        obj_id_str = f'{obj_id:02d}' # Zero-pad to two digits

        # Skip object '02', '03', and '07'
        if obj_id_str in ['02', '03', '07'] :
            print(f"Skipping object ID: {obj_id_str}")
            continue

        # Construct path to the 'rgb' folder for the current object
        rgb_folder_path = os.path.join(base_dataset_path, f'{obj_id_str}', 'rgb')

        if os.path.exists(rgb_folder_path) and os.path.isdir(rgb_folder_path):
            print(f"Processing rgb folder: {rgb_folder_path}")
            # List and sort contents of the 'rgb' folder
            files = sorted([f for f in os.listdir(rgb_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            for old_filename in files:
                # Preserve original file extension
                file_extension = os.path.splitext(old_filename)[1]
                new_filename = f'{global_counter:05d}{file_extension}' # Zero-pad counter to 5 digits

                old_filepath = os.path.join(rgb_folder_path, old_filename)
                new_filepath = os.path.join(output_directory_path, new_filename)

                try:
                    shutil.copy2(old_filepath, new_filepath) # Use copy2 to preserve metadata
                    logging.debug(f"Copied '{old_filename}' to '{new_filename}' in {output_directory_path}")
                    global_counter += 1
                except OSError as e:
                    logging.error(f"Error copying file {old_filepath} to {new_filepath}: {e}")
                    print(f"Error copying file {old_filepath} to {new_filepath}: {e}")
                    return
        else:
            print(f"RGB folder not found or not a directory for object ID {obj_id_str}: {rgb_folder_path}")
    print("Function 'rename_rgb_items' done.")

def verify_image_consistency(base_dataset_path, processed_images_path):
    global_counter = 0
    all_consistent = True
    print("Function 'verify_image_consistency' called. Initializing counter and consistency flag.")

    # Image extensions to consider
    image_extensions = ('.png', '.jpg', '.jpeg')

    # Iterate through object IDs from 1 to 15
    for obj_id in range(1, 16): 
        obj_id_str = f'{obj_id:02d}'

        # Skip object '02', '03', and '07' as requested
        if obj_id_str in ['02', '03', '07']:
            print(f"Skipping object ID: {obj_id_str}")
            continue

        # Construct path to the original 'rgb' folder for the current object
        original_rgb_folder_path = os.path.join(base_dataset_path, obj_id_str, 'rgb')

        if os.path.exists(original_rgb_folder_path) and os.path.isdir(original_rgb_folder_path):
            print(f"Processing original rgb folder: {original_rgb_folder_path}")

            # List and sort contents of the 'rgb' folder
            original_files = sorted([f for f in os.listdir(original_rgb_folder_path) if f.lower().endswith(image_extensions)])

            for original_filename in original_files:
                # Construct FULL path for the original file (Necessario per aprirlo con PIL)
                original_full_path = os.path.join(original_rgb_folder_path, original_filename)

                # Preserve original file extension
                file_extension = os.path.splitext(original_filename)[1]
                
                # Construct the expected new filename in the processed directory
                expected_new_filename = f'{global_counter:05d}{file_extension}' 
                expected_new_filepath = os.path.join(processed_images_path, expected_new_filename)

                # 1. VERIFICA ESISTENZA FILE
                if not os.path.exists(expected_new_filepath):
                    logging.error(f"Consistency check failed: Processed file '{expected_new_filepath}' not found for original '{original_filename}'.")
                    print(f"MISSING FILE: '{expected_new_filepath}' not found.")
                    all_consistent = False
                
                # 2. VERIFICA METADATI (Se il file esiste)
                else:
                    try:
                        with Image.open(original_full_path) as img_orig, Image.open(expected_new_filepath) as img_proc:
                            
                            # A. Verifica Dimensioni (Width, Height)
                            if img_orig.size != img_proc.size:
                                logging.error(f"Size mismatch: Original {original_filename} is {img_orig.size}, Processed {expected_new_filename} is {img_proc.size}")
                                print(f"METADATA ERROR (Size): {original_filename} {img_orig.size} != {expected_new_filename} {img_proc.size}")
                                all_consistent = False

                            # B. Verifica Modalità Colore (es. RGB, L, CMYK)
                            # Nota: Se il processamento converte png in jpg, la modalità potrebbe cambiare da RGBA a RGB.
                            # Se ti aspetti che siano identiche, lascia questo controllo.
                            elif img_orig.mode != img_proc.mode:
                                logging.warning(f"Mode mismatch: Original {img_orig.mode}, Processed {img_proc.mode}")
                                print(f"METADATA WARNING (Mode): {original_filename} ({img_orig.mode}) != {expected_new_filename} ({img_proc.mode})")
                                # Se il cambio di modalità è un errore critico, decommenta la riga sotto:
                                # all_consistent = False 

                            # C. Verifica EXIF (Base)
                            # Attenzione: Molti processi di salvataggio rimuovono gli EXIF.
                            # Qui controlliamo solo se i dizionari info coincidono (es. DPI).
                            else:
                                # Esempio: Controllo DPI (dots per inch)
                                info_orig = img_orig.info.get('dpi')
                                info_proc = img_proc.info.get('dpi')
                                if info_orig and info_proc and info_orig != info_proc:
                                    logging.warning(f"DPI mismatch: Original {info_orig}, Processed {info_proc}")

                                logging.debug(f"Verified file and metadata: {expected_new_filepath}")

                    except Exception as e:
                        logging.error(f"Error reading image metadata for {original_filename}: {str(e)}")
                        print(f"ERROR reading metadata: {str(e)}")
                        all_consistent = False

                global_counter += 1
        else:
            print(f"Original RGB folder not found or not a directory for object ID {obj_id_str}: {original_rgb_folder_path}")
            all_consistent = False 

    logging.info(f"Consistency check completed. Total files expected/processed: {global_counter}")
    
    if all_consistent:
        print("All image files AND metadata are consistent.")
    else:
        print("Inconsistencies found between original and processed image files.")

    return all_consistent

def count_image_files(directory_path, count_within_obj_rgb=False):
    image_count = 0
    image_extensions = ('.png', '.jpg', '.jpeg') # Tuple for efficient checking

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return 0

    if count_within_obj_rgb:
        # Iterate through object IDs and count files within their 'rgb' subfolders
        for obj_id in range(1, 15):
            obj_id_str = f'{obj_id:02d}' # Zero-pad to two digits

            # Skip object IDs '02', '03', and '07'
            if obj_id_str in ['02', '03', '07']:
                continue

            # Construct path to the 'rgb' folder for the current object
            rgb_folder_path = os.path.join(directory_path, obj_id_str, 'rgb')

            if os.path.exists(rgb_folder_path) and os.path.isdir(rgb_folder_path):
                for file in os.listdir(rgb_folder_path):
                    if file.lower().endswith(image_extensions):
                        image_count += 1
    else:
        # Original recursive counting logic
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_count += 1
    return image_count

def rot_matrix_to_quaternion(rot_matrix):
    """
    Converts a 3x3 rotation matrix to a quaternion (w, x, y, z).
    Args:
        rot_matrix (np.array): A 3x3 rotation matrix.
    Returns:
        np.array: A 4-element numpy array representing the quaternion (w, x, y, z).
    """
    rot_matrix = np.array(rot_matrix)
    r = R.from_matrix(rot_matrix)
    quat_xyzw = r.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz

def process_gt_files(dataset_base_path, output_gt_path):
    global_img_counter = 0
    all_gt_data = {}

    # Object IDs to process (1 to 15, skipping 02, 03, 07)
    obj_ids_to_process = [f'{i:02d}' for i in range(1, 16) if f'{i:02d}' not in ['02', '03', '07']]

    print(f"Starting processing of gt.yml files from: {dataset_base_path}")

    for obj_id_str in obj_ids_to_process:
        gt_filepath = os.path.join(dataset_base_path, obj_id_str, 'gt.yml')
        rgb_folder_path = os.path.join(dataset_base_path, obj_id_str, 'rgb')

        if not os.path.exists(gt_filepath):
            print(f"Warning: gt.yml not found for object {obj_id_str}. Skipping.")
            continue

        num_rgb_images = count_image_files(rgb_folder_path)

        try:
            with open(gt_filepath, 'r') as f:
                gt_data = yaml.safe_load(f)

            if gt_data is None:
                print(f"Warning: gt.yml for object {obj_id_str} is empty or invalid. Skipping.")
                continue

            # Ensure gt_data is a dictionary, sometimes it might be a list of dicts for a single frame
            if isinstance(gt_data, list):
                # If the entire file is a list of entries, convert it to a dict indexed by frame_id
                temp_gt_data = {}
                for entry in gt_data:
                    if isinstance(entry, dict) and 0 in entry: # Assuming frame 0 if not explicitly indexed
                        temp_gt_data[0] = entry[0]
                        break # Only handle the first if it's a simple list for a single frame
                gt_data = temp_gt_data

            if not isinstance(gt_data, dict):
                print(f"Warning: Unexpected format in gt.yml for object {obj_id_str}. Expected dictionary. Skipping.")
                continue

            # Sort frame IDs to ensure consistent processing order
            sorted_frame_ids = sorted(gt_data.keys())

            for frame_idx in range(num_rgb_images):
                # Check if the frame_idx exists in gt_data, otherwise skip
                if frame_idx not in gt_data:
                    # This can happen if gt.yml has fewer entries than actual rgb images,
                    # or if numbering is not consecutive. Adjust as per actual data structure.
                    continue

                frame_data_list = gt_data[frame_idx]

                # Ensure frame_data_list is a list, as gt.yml structure can vary (list of objects per frame)
                if not isinstance(frame_data_list, list):
                    frame_data_list = [frame_data_list]

                processed_frame_objects = []
                for obj_entry in frame_data_list:
                    if 'cam_R_m2c' in obj_entry and 'cam_t_m2c' in obj_entry and 'obj_bb' in obj_entry and 'obj_id' in obj_entry:
                        rotation_matrix = np.array(obj_entry['cam_R_m2c']).reshape((3, 3))
                        quaternion = rot_matrix_to_quaternion(rotation_matrix)

                        processed_obj_entry = {
                            'cam_R_m2c': obj_entry['cam_R_m2c'],
                            'cam_t_m2c': obj_entry['cam_t_m2c'],
                            'obj_bb': obj_entry['obj_bb'],
                            'quaternion': [round(float(q), 8) for q in quaternion],
                            'obj_id': obj_entry['obj_id']
                        }
                        processed_frame_objects.append(processed_obj_entry)

                if processed_frame_objects:
                    all_gt_data[global_img_counter] = processed_frame_objects
                    global_img_counter += 1

        except Exception as e:
            print(f"Error processing gt.yml for object {obj_id_str}: {e}")

    # Write the consolidated data to the output YAML file
    if all_gt_data:
        with open(output_gt_path, 'w') as f:
            # Manually write to get the desired format (frame_id: - obj_data)
            for frame_id in sorted(all_gt_data.keys()):
                f.write(f"{frame_id}:\n")
                for obj in all_gt_data[frame_id]:
                    f.write(f"- cam_R_m2c: {obj['cam_R_m2c']}\n")
                    f.write(f"  cam_t_m2c: {obj['cam_t_m2c']}\n")
                    f.write(f"  obj_bb: {obj['obj_bb']}\n")
                    f.write(f"  quaternion: {obj['quaternion']}\n")
                    f.write(f"  obj_id: {obj['obj_id']}\n")
        print(f"Consolidated gt data successfully written to: {output_gt_path}")
    else:
        print("No data to consolidate. output_gt.yml not created.")

    print(f"Total processed frames: {global_img_counter}")

    """# Example usage:
    dataset_base_path = '/content/6D_Pose_Estimation_light/dataset/Linemod_preprocessed/data'
    output_gt_path = '/content/6D_Pose_Estimation_light/dataset/Linemod_preprocessed/consolidated_gt.yml'

    process_gt_files(dataset_base_path, output_gt_path)"""

def delete_rgb_folders(base_dataset_path):
    
    print(f"Starting deletion of rgb folders within: {base_dataset_path}")

    # Object IDs to process (1 to 15, skipping 02, 03, 07)
    obj_ids_to_process = [f'{i:02d}' for i in range(1, 16) if f'{i:02d}' not in ['02', '03', '07']]

    for obj_id_str in obj_ids_to_process:
        rgb_folder_path = os.path.join(base_dataset_path, obj_id_str, 'rgb')
        if os.path.exists(rgb_folder_path) and os.path.isdir(rgb_folder_path):
            try:
                shutil.rmtree(rgb_folder_path)
                print(f"Successfully deleted: {rgb_folder_path}")
            except OSError as e:
                print(f"Error deleting {rgb_folder_path}: {e}")
        else:
            print(f"RGB folder not found for object {obj_id_str} at {rgb_folder_path}. Skipping.")

    print("Finished deleting rgb folders.")