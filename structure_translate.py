import numpy as np
import logging
import os
from object_text_anno import mmengine_track_func
from strict_translate import strict_list_translate
import re

def is_chinese(text):
    # 正则表达式匹配中文字符
    pattern = re.compile(r'[\u4e00-\u9fff]')
    if pattern.search(text):
        return True
    else:
        return False

@mmengine_track_func
def anno_translation(region_info_file,src_lang,tgt_lang,need_translate_index = [1,3,4,5]):
    save_file = region_info_file.replace('.npy','_trans.npy')
    region_info = np.load(region_info_file,allow_pickle=True)
    if not region_info.any():
        return region_info
    if os.path.exists(save_file):
        return np.load(save_file,allow_pickle=True)
    total_list = []
    for _index in need_translate_index:
        total_list += [region_info[_index][k] for k in region_info[_index].keys()]
    # import pdb; pdb.set_trace()
    output_list, num_trys = strict_list_translate(total_list, src_lang, tgt_lang)
    # logging.warning(f"Translated with {num_trys} trys, prompt_tokens: {token_usage['prompt_tokens']}, completion_tokens: {token_usage['completion_tokens']}")
    logging.warning(f"Translated with {num_trys} trys for {region_info_file}")
    if num_trys >= 10:
        logging.warning(f"Failed to translate {region_info_file}")
        return region_info
    start = 0
    for _index in need_translate_index:
        for key in region_info[_index].keys():
            region_info[_index][key] = output_list[start]
            start+=1
    all_chinese = [is_chinese(text) for text in output_list]
    if not all(all_chinese):
        logging.warning(f"Warning: NOT all translated texts are chinese for {region_info_file}")
    np.save(save_file,region_info)
    return region_info

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

    anno_translation(tasks[0])
    import mmengine
    mmengine.utils.track_parallel_progress(anno_translation, tasks, nproc=8)

