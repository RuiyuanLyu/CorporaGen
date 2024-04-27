from strict_translate import strict_list_translate
import numpy as np

from object_text_anno import mmengine_track_func

@mmengine_track_func
def anno_translation(region_info_file,src_lang,tgt_lang,need_translate_index = [1,3,4,5]):
    region_info = np.load(region_info_file,allow_pickle=True)
    total_list = []
    for _index in need_translate_index:
        total_list+= [region_info[_index][k] for k in region_info[_index].keys()]

    output_list,try_ = strict_list_translate(total_list,src_lang,tgt_lang)
    print(try_)

    start = 0
    for _index in need_translate_index:
        for k in range(len(region_info[_index].keys())):
            _key = list(region_info[_index].keys())[k]
            region_info[_index][_key] = output_list[k+start]
        start+=len(region_info[_index].keys())
    np.save(region_info_file.replace('.npy','_trans.npy'),region_info)
    return region_info

if __name__ == "__main__":
    DATA_ROOT = 'data'
    REGION_VIEW_DIR_NAME = 'region_views'
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
    scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith('scene') or scene_id.startswith('1mp3d')]
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

    import mmengine
    mmengine.utils.track_parallel_progress(anno_translation, tasks, nproc=8)

