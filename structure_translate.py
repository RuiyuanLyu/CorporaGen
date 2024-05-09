import numpy as np
import logging
import os
from utils.decorators import mmengine_track_func
from strict_translate import strict_list_translate, is_chinese


@mmengine_track_func
def anno_translation(region_info_file,src_lang,tgt_lang,need_translate_index = [1,3,4,5]):
    save_file = region_info_file.replace('.npy','_trans.npy')
    region_info = np.load(region_info_file,allow_pickle=True)
    if not region_info.any():
        return region_info
    if os.path.exists(save_file):
        # load from save_file if exists. 
        region_info = np.load(save_file, allow_pickle=True)
        # The translation might be imperfect. So we need to load from region_info_file if the save_file is empty.
        if not region_info.any():
            region_info = np.load(region_info_file, allow_pickle=True)
    assert region_info.any(), f"{region_info_file} should not be empty"
    total_list = []
    for _index in need_translate_index:
        total_list += [region_info[_index][k] for k in region_info[_index].keys()]
    # If some are already in Chinese, we don't need to translate them, which is implemented in strict_list_translate.
    output_list, num_trys = strict_list_translate(total_list, src_lang, tgt_lang, max_try=3)
    if num_trys > 0:
        logging.warning(f"Translated with {num_trys} trys for {region_info_file}")
    else: 
        logging.warning(f"{region_info_file} is already in Chinese")
        return region_info
    if num_trys >= 3:
        # Do not save partial results if the length do not match.
        if len(output_list) != len(total_list):
            np.save(save_file, None)
            return None
    start = 0
    for _index in need_translate_index:
        for key in region_info[_index].keys():
            region_info[_index][key] = output_list[start]
            start+=1
    if not all([is_chinese(text) for text in output_list]):
        logging.warning(f"Warning: NOT all translated texts are chinese for {region_info_file}")
    np.save(save_file,region_info)
    return region_info

@mmengine_track_func
def check_translation(region_info_file,src_lang,tgt_lang,need_translate_index = [1,3,4,5]):
    # check the length of "total list" is the same
    translation_file = region_info_file.replace('.npy','_trans.npy')
    region_info = np.load(region_info_file,allow_pickle=True)
    if not region_info.any():
        logging.warning(f"{region_info_file} is empty, Nothing to translate")
        return False
    if not os.path.exists(translation_file):
        logging.warning(f"{translation_file} does not exist")
        return False
    trans_info = np.load(translation_file,allow_pickle=True)
    if not trans_info.any():
        logging.warning(f"{translation_file} is empty")
        return False
    total_list = []
    for _index in need_translate_index:
        total_list += [region_info[_index][k] for k in region_info[_index].keys()]
    total_list2 = []
    for _index in need_translate_index:
        total_list2 += [trans_info[_index][k] for k in trans_info[_index].keys()]
    ret = len(total_list) == len(total_list2)
    if not ret:
        logging.warning(f"The length of total_list is not the same in {region_info_file} and {translation_file}")
        return False
    ret = all([is_chinese(x) for x in total_list2])
    if not ret:
        logging.warning(f"Not all texts in {translation_file} are in Chinese")
    return ret


if __name__ == "__main__":
    DATA_ROOT = 'data'
    REGION_VIEW_DIR_NAME = 'region_views'
    log_file_name = f"{__file__[:-3]}.log"
    print(f"log will be written to {log_file_name}")
    # Set level to warning to avoid writing HTTP requests to log
    logging.basicConfig(filename=log_file_name, level=logging.WARNING, format='%(asctime)s - %(message)s')

    # scene_id = 'scene0000_00'
    # data_dir = f'data/{scene_id}/region_views/4_toliet region'
    # region_info_file = os.path.join(data_dir,'struction.npy')
    # region_info = np.load(region_info_file,allow_pickle=True)
    # print(region_info)
    # out = anno_translation(region_info_file,"English","Chinese")
    # print(out)
    # np.save(data_dir+'/struction_trans.npy',out)
    import os
    scene_ids = os.listdir(DATA_ROOT)
    scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith('scene') or scene_id.startswith('1mp3d') or scene_id.startswith('3rscan')]
    scene_ids = sorted(scene_ids)
    tasks = []
    for scene_id in scene_ids:
        dir_to_check = os.path.join(DATA_ROOT, scene_id, REGION_VIEW_DIR_NAME)
        if not os.path.exists(dir_to_check):
            continue
        region_names = os.listdir(dir_to_check)
        region_names = [region_name for region_name in region_names if region_name.endswith('region')]
        region_names = sorted(region_names)
        data_dirs = [os.path.join(dir_to_check, region_name) for region_name in region_names]
        for data_dir in data_dirs:
            region_info_file = os.path.join(data_dir,'struction.npy')
            if not os.path.exists(region_info_file):
                continue
            tasks.append((region_info_file,"English","Chinese"))

    # anno_translation(tasks[0])
    import mmengine
    # mmengine.utils.track_parallel_progress(anno_translation, tasks, nproc=8)
    mmengine.utils.track_parallel_progress(check_translation, tasks, nproc=8)

