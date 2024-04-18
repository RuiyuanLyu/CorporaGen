import os
import shutil
import logging
from utils.utils_read import load_json
from tqdm import tqdm

# To avoid accidental overwrite, we use a safe mode that only write logs.
# If you want to actually overwrite files, set SAFE_MODE_ON to False.
# We recommend to use a safe mode and check the logs to make sure everything is fine, before setting SAFE_MODE_ON to False.
SAFE_MODE_ON = True

def get_dataset_prefix(scene_id):
    if 'mp3d' in scene_id:
        dataset_prefix = 'matterport3d'
    elif '3rscan' in scene_id:
        dataset_prefix = '3rscan'
    elif 'scene' in scene_id:
        dataset_prefix = 'scannet'
    return dataset_prefix
    
def copy_tree(src, dst):
    if os.path.exists(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if not SAFE_MODE_ON:
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)  # 使用 copy2 来保留元数据
        logging.info(f'Copied "{src}" to "{dst}"')
    else:
        logging.warning(f'Source directory "{src}" does not exist.')
        # print(f'Source directory "{src}" does not exist.')

def contains_prefix(string, prefixes):
    return any(string.startswith(prefix) for prefix in prefixes)

from functools import wraps
def mmengine_track_func(func):
    @wraps(func)
    def wrapped_func(args):
        result = func(*args)
        return result
    return wrapped_func

@mmengine_track_func
def copy_corpora_data(scene_id, model_name):
    dataset_prefix = get_dataset_prefix(scene_id)
    corpora_string = f'corpora_object_{model_name}_crop'
    src = os.path.join(ROOT_DIR, dataset_prefix, scene_id, corpora_string)
    dst = os.path.join(TARGET_DIR, scene_id, corpora_string)
    copy_tree(src, dst)

@mmengine_track_func
def copy_corpora_data_content_aware(scene_id, model_name):
    corpora_string = f'corpora_object_{model_name}_crop'
    src_dir = os.path.join(ROOT_DIR, scene_id, corpora_string)
    dst_dir = os.path.join(TARGET_DIR, scene_id, corpora_string)
    if not os.path.exists(src_dir):
        logging.warning(f'Source directory "{src_dir}" does not exist.')
        return
    for item in os.listdir(src_dir):
        if not item.endswith('.json'):
            continue
        src_file = os.path.join(src_dir, item)
        dst_file = os.path.join(dst_dir, item)
        if check_need_to_skip(dst_file):
            logging.info(f'Skiping "{dst_file}", since it exists and is good enough.')
            continue
        # if not SAFE_MODE_ON:
        #     shutil.copy2(src_file, dst_file)  # 使用 copy2 来保留元数据
        logging.info(f'Copied "{src_file}" to "{dst_file}"')

import json
def check_need_to_skip(img_dict_path):
    """
        If existing dict is good enough, return True.
    """
    if not os.path.exists(img_dict_path):
        return False
    with open(img_dict_path) as f:
        data = json.load(f)
    original_desc = data["original_description"]
    object_type = os.path.basename(img_dict_path)
    object_type = object_type.split("_")[1]
    if object_type in original_desc:
        return True
    return False


mapping_mp3d = load_json('scene_mappings/mp3d_mapping.json') 
mapping_mp3d = {v: k for k, v in mapping_mp3d.items()}

def back_map_view_id(view_id, mode, house_hash=None):
    """
        Args:
            view id: (some hash value like b185432bf33645aca813ac2a961b4140_i2_5) or frame-000137.color
            mode: must in 'mp3d' or '3rscan'
        Returns:
            mapped view id (the numbered view id in anno file). Example: 0165_2_5 (mp3d) or 000137 (3rscan)
    """
    assert mode in ['mp3d', '3rscan'], f"unsupported mode {mode}"
    if mode == 'mp3d':
        assert house_hash is not None
        camera_hash = view_id.split('_')[0]
        angle_id = view_id[-3:]
        mapping_for_cur_scene = load_json(f'scene_mappings/mp3d_rename/{house_hash}.json')
        camera_id = mapping_for_cur_scene.get(camera_hash, camera_hash)
        return f"{camera_id}_{angle_id}"
    elif mode == '3rscan':
        return view_id.replace('frame-', '').replace('.color', '')

def rename_pictures_and_jsons(scene_id):
    dataset = get_dataset_prefix(scene_id)
    scene_dir = os.path.join(ROOT_DIR, scene_id)
    sub_dirs = os.listdir(scene_dir)
    for sub_dir in sub_dirs:
        if not sub_dir.startswith('corpora_object') and not sub_dir in ['cropped_objects', 'painted_objects']:
            continue
        directory = os.path.join(scene_dir, sub_dir)
        for file_name in os.listdir(directory):
            if not (file_name.endswith('.json') or file_name.endswith('.jpg')):
                continue
            if dataset == 'matterport3d':
                # "004_doorframe_44661972414d44fabc0799f237e4d7f0_i1_3.json" -> "004_doorframe_123_1_3.json"
                view_id = ('_').join(file_name.split('_')[2:]).split('.')[0]
                house_hash = mapping_mp3d[scene_id.split('_region')[0]]
                new_view_id = back_map_view_id(view_id, 'mp3d', house_hash)
                new_file_name = file_name.replace(view_id, new_view_id)
            elif dataset == '3rscan':
                # 003_mirror_frame-000018.color.json -> 003_mirror_000018.json
                new_file_name = file_name.replace('frame-', '').replace('.color', '')
            src = os.path.join(directory, file_name)
            dst = os.path.join(directory, new_file_name)
            shutil.move(src, dst)
            logging.info(f'Renamed "{src}" to "{dst}"')

if __name__ == "__main__":
    log_file_name = f"{__file__[:-3]}.log"
    print(f"log will be written to {log_file_name}")
    # clear the log file
    with open(log_file_name, 'w'): 
        pass
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(message)s')
    ROOT_DIR = 'mp3d_cogvlm'
    # ROOT_DIR = 'data'
    TARGET_DIR = 'data'
    # if not os.path.exisdts(TARGET_DIR):
    #     os.makedirs(TARGET_DIR)
    #################################################################################
    ## rename the pictures and jsons for the given scenes
    # scene_ids = ['3rscan0000', '3rscan0040', '3rscan0063', '3rscan0078', '3rscan0102', '3rscan0131', '3rscan0151', '3rscan0182', '3rscan0210', '3rscan0261', '3rscan0284', '3rscan0312', '3rscan0339', '3rscan0368', '3rscan0389', '3rscan0409', '3rscan0501', '3rscan0530', '3rscan0544', '3rscan0575', '3rscan0599', '3rscan0602', '3rscan0637', '3rscan0672', '3rscan0698', '3rscan0746', '3rscan0777', '3rscan0803', '3rscan0854', '3rscan0886', '3rscan0921', '3rscan0943', '3rscan0970', '3rscan0999', '3rscan1016', '3rscan1049', '3rscan1077', '3rscan1098', '3rscan1127', '3rscan1169', '3rscan1192', '3rscan1231', '3rscan1260', '3rscan1298', '3rscan1326', '3rscan1359']
    # import time
    # import cv2
    # for scene_id in tqdm(scene_ids):
    #     # rename_pictures_and_jsons(scene_id)
    #     # new task: rotate the painted images by 90 degrees counterclockwise
    #     image_dir = os.path.join(ROOT_DIR, scene_id, 'painted_objects')
    #     for file_name in os.listdir(image_dir):
    #         if not file_name.endswith('.jpg'):
    #             continue
    #         src = os.path.join(image_dir, file_name)
    #         img = cv2.imread(src)
    #         img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         cv2.imwrite(src, img)
    #         logging.info(f'Rotated "{src}" by 90 degrees counterclockwise')
    #     time.sleep(0.1)
    # scene_ids = ['3rscan0000', '3rscan0040', '3rscan0063', '3rscan0078', '3rscan0102', '3rscan0131', '3rscan0151', '3rscan0182', '3rscan0210', '3rscan0261', '3rscan0284', '3rscan0312', '3rscan0339', '3rscan0368', '3rscan0389', '3rscan0409', '3rscan0501', '3rscan0530', '3rscan0544', '3rscan0575', '3rscan0599', '3rscan0602', '3rscan0637', '3rscan0672', '3rscan0698', '3rscan0746', '3rscan0777', '3rscan0803', '3rscan0854', '3rscan0886', '3rscan0921', '3rscan0943', '3rscan0970', '3rscan0999', '3rscan1016', '3rscan1049', '3rscan1077', '3rscan1098', '3rscan1127', '3rscan1169', '3rscan1192', '3rscan1231', '3rscan1260', '3rscan1298', '3rscan1326', '3rscan1359']
    # import time
    # import cv2
    # for scene_id in tqdm(scene_ids):
    #     # rename_pictures_and_jsons(scene_id)
    #     # new task: rotate the painted images by 90 degrees counterclockwise
    #     image_dir = os.path.join(ROOT_DIR, scene_id, 'painted_objects')
    #     for file_name in os.listdir(image_dir):
    #         if not file_name.endswith('.jpg'):
    #             continue
    #         src = os.path.join(image_dir, file_name)
    #         img = cv2.imread(src)
    #         img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         cv2.imwrite(src, img)
    #         logging.info(f'Rotated "{src}" by 90 degrees counterclockwise')
    #     time.sleep(0.1)
    #################################################################################
    ## rename the pictures and jsons for the given scenes with specific prefixes
    ## only applicatble for matterport3d
    # import pandas as pd
    # import time
    # scene_id_prefixes = ['1mp3d_0051', '1mp3d_0080', '1mp3d_0020', '1mp3d_0068', '1mp3d_0056', '1mp3d_0034', '1mp3d_0089', '1mp3d_0037', '1mp3d_0055', '1mp3d_0015', '1mp3d_0053', '1mp3d_0028', '1mp3d_0063', '1mp3d_0031', '1mp3d_0086', '1mp3d_0054', '1mp3d_0021', '1mp3d_0049']
    # prefix_series = pd.Series(scene_id_prefixes)
    # scene_ids = os.listdir(ROOT_DIR)
    # scene_ids = [scene for scene in scene_ids if contains_prefix(scene, prefix_series)]
    # for scene_id in tqdm(scene_ids):
    #     rename_pictures_and_jsons(scene_id)
    #     time.sleep(0.1)
    #################################################################################
    # copy the selected corpora data for the given scenes 
    # scene_ids_to_ignore = set(["1mp3d_0001_region8", "1mp3d_0002_region23", "3rscan0138", "3rscan0036", "scene0026_00", "scene0094_00", "scene0147_00"])
    scene_ids = os.listdir(ROOT_DIR)
    scene_ids = [scene for scene in scene_ids if contains_prefix(scene, ['1mp3d', '3rscan', 'scene'])]
    # scene_ids_mp3d = os.listdir(os.path.join(ROOT_DIR, "matterport3d"))
    # scene_ids_scannet = os.listdir(os.path.join(ROOT_DIR, "scannet"))
    # scene_ids_3rscan = os.listdir(os.path.join(ROOT_DIR, "3rscan"))
    # scene_ids = scene_ids_mp3d + scene_ids_scannet + scene_ids_3rscan
    scene_ids.sort()
    model_names = ['cogvlm', 'InternVL-Chat-V1-2-Plus'][:1]
    tasks = []
    for scene_id in scene_ids:
        # if scene_id in scene_ids_to_ignore:
        #     continue
        for model_name in model_names:
            tasks.append((scene_id, model_name))
    import mmengine
    mmengine.track_parallel_progress(copy_corpora_data_content_aware, tasks, nproc=1)
    #################################################################################
    ## copy all data for the given scenes 
    # scene_ids = ['3rscan0000', '3rscan0040', '3rscan0063', '3rscan0078', '3rscan0102', '3rscan0131', '3rscan0151', '3rscan0182', '3rscan0210', '3rscan0261', '3rscan0284', '3rscan0312', '3rscan0339', '3rscan0368', '3rscan0389', '3rscan0409', '3rscan0501', '3rscan0530', '3rscan0544', '3rscan0575', '3rscan0599', '3rscan0602', '3rscan0637', '3rscan0672', '3rscan0698', '3rscan0746', '3rscan0777', '3rscan0803', '3rscan0854', '3rscan0886', '3rscan0921', '3rscan0943', '3rscan0970', '3rscan0999', '3rscan1016', '3rscan1049', '3rscan1077', '3rscan1098', '3rscan1127', '3rscan1169', '3rscan1192', '3rscan1231', '3rscan1260', '3rscan1298', '3rscan1326', '3rscan1359']
    # for scene_id in scene_ids:
    #     dataset_prefix = get_dataset_prefix(scene_id)
    #     src = os.path.join(ROOT_DIR, dataset_prefix, scene_id)
    #     dst = os.path.join(TARGET_DIR, scene_id)
    #     copy_tree(src, dst)
    #################################################################################
    ## copy all data for scenes with specfic prefixes
    # import pandas as pd
    # scene_id_prefixes = ['1mp3d_0051', '1mp3d_0080', '1mp3d_0020', '1mp3d_0068', '1mp3d_0056', '1mp3d_0034', '1mp3d_0089', '1mp3d_0037', '1mp3d_0055', '1mp3d_0015', '1mp3d_0053', '1mp3d_0028', '1mp3d_0063', '1mp3d_0031', '1mp3d_0086', '1mp3d_0054', '1mp3d_0021', '1mp3d_0049']
    # prefix_series = pd.Series(scene_id_prefixes)
    # scene_ids = os.listdir(os.path.join(ROOT_DIR, "matterport3d"))
    # scene_ids = [scene for scene in scene_ids if contains_prefix(scene, prefix_series)]
    # for scene_id in tqdm(scene_ids):
    #     dataset_prefix = get_dataset_prefix(scene_id)
    #     src = os.path.join(ROOT_DIR, dataset_prefix, scene_id)
    #     dst = os.path.join(TARGET_DIR, scene_id)
    #     copy_tree(src, dst)