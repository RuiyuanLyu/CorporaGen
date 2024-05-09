import os
import json
from tqdm import tqdm
from utils.openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups
from utils.utils_read import load_json
import numpy as np
import logging

RELATION_list = ['in', 'on', 'under', 'closed by', 'next to', 'hanging/lying on', 'above']
DESCRIBE_ASPECT_list = ['location and function description', 'space layout and dimensions',
                        'doors, windows, walls, floors and ceilings', 'soft fitting elements and decorative details',
                        'lighting design', 'color matching and style theme', 'special features']


def annotate_region_image(image_paths, region_type, object_ids, object_types):
    """
        Uses GPT-4 to annotate an object in an image.
        Args:
            image_paths: A list of paths to images corresponding to the single region.
            region_type: A string indicating the type of the area, e.g. "bedroom".
            object_ids: A list of ints indicating the ids of the objects in the image.
            object_types: A list of strings indicating the types of the objects in the image.
        Returns:
            A list of strings, each string is a part of the annotation.
    """
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements,and you are particularly familiar with how each piece of furniture relates to everyday human activity. The expected reader is a high-school student with average knowledge of furniture design."
    region_type = region_type.replace("other ", "")
    user_message4 = f"I will share photos of a room's {region_type} and use 3D boxes to highlight important items within it. In this region, the items that must be described include "
    for (object_id, object_type) in zip(object_ids, object_types):
        user_message4 += f"<{object_type}_{object_id}>, "
    user_message4 = user_message4[:-2] + ",don't leave out any of them."
    user_message4 += f" I want you to list the location relationships between these objects, where location relationships are between two objects, chosen from {RELATION_list}. "
    user_message4 += "I want you to provide a JSON file that describes the location relationship of these important items. (a dict in the form of '(A,B) : their location relationship', such as: ('<pillow_2>','<bed_1>'). Here 'on' means that the first item is 'on' the second), for example: { ('<pillow_2>','<bed_1>'):'on', ('<chair_3>','<table_4>'):'next to', ...}. Please only give me the result as a JSON file."
    user_message5 = "Which objects are lying/hanging on the wall? Which objects are standing on the floor? Just pick them out and give me a JSON file(a dict), like {'wall':['<picture_1>','<picture_2>'],'floor':['<table_3>','<chair_4>']}"
    user_message6 = 'What makes the item special in this region? Why did you notice the item? You can think about it in these terms: its special postion (like "the chair in the middle of several chairs"), its special role in everyday life (like "I will sit on this chair while eating"). Just give me the result as a JSON file (a dict, the key is the item, the value is a sentence describe its particularity, such as {<backpack_1>:"the only one on the floor, the owner may carry it when going to school. "}).'
    user_message7 = f"What's more, based on these layouts, could you share information about the region, these should be included: "
    user_message9 = 'Further, I want you to think about are these objects related to each other in the use of functions in everyday activities? Please list the these relationships, in the format of a JSON file (a dict, the key is a tuple of two items, the value is a sentence describe their relationship, such as {(<towel_0>,<sink_1>):"the <towel_0> can be used to clean the <sink_1>",(<vase_2>,<plant_3>):"the <plant_3> is normally placed in the <vase_2> to express its beauty"})'

    for _index in range(len(DESCRIBE_ASPECT_list)):
        user_message7 += f'{_index}.{DESCRIBE_ASPECT_list[_index]}, '
    user_message7 = user_message7[:-2] + f" Just give me a JSON file (a dict with keys:{DESCRIBE_ASPECT_list}, "
    user_message7 += "with corresponding values are strings), like {'lighting design':...,'special features':...,...}.Just give me the result as a JSON file."
    user_message8 = "Which set of these items together belong to a larger class or perform some function together in the region (a set must contain at least two items)? I want you to write this into a JSON file (a dict, the key is a list of items, the value is a sentence. The sentence describes a larger class they belongs to, their role, and their function in the region, such as {[<picture_1>,<vase_2>]:'Belong to the same class decoration, together contribute to the aesthetic feeling'})"

    source_groups = [
        [user_message4, *image_paths],
        [user_message5],
        [user_message9],
        [user_message8],
        [user_message6],
        [user_message7]
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)  # high detail
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    conversation, token_usage = mimic_chat_budget(content_groups, system_prompt=system_prompt, num_turns_expensive=6,
                                                  report_token_usage=True, max_token_length=2500, json_mode=False)
    logging.warning(
        f"Prompt tokens: {token_usage['prompt_tokens']}, Completion tokens: {token_usage['completion_tokens']}")
    annotation = []
    for message in conversation:
        if message["role"] == "assistant":
            annotation.append(message["content"])
    return annotation

def check_annotation(annotation, object_ids, object_types):
    """
        Checks if the annotation contains the object ids and types correctly.
        Args:
            annotation: A list of descriptions.
            object_ids: A list of ints, the ids of the objects in the region.
            object_types: A list of strings, the types of the objects in the region.
        Returns:
            is_valid: A boolean indicating if the annotation is valid.
            error_message: A string containing the error message if the annotation is not valid.
    """
    description = " ".join(annotation)
    num_objects = len(object_ids)
    num_detected_objects = 0
    for object_id, object_type in zip(object_ids, object_types):
        if f"<{object_id}>" not in description:
            print(f"Error: {object_type} <{object_id}> not found in annotation.")
        else:
            num_detected_objects += 1
    print(f"Detected {num_detected_objects} out of {num_objects} objects in the annotation.")
    return num_detected_objects == num_objects

def get_visible_objects_dict(dataset_tar, scene_id_tar):
    """
        Reads the annotation files and returns a dictionary {object_id: visible_view_ids} for each object in the scene.
        Args:
            dataset_tar: A string indicating the dataset.
            scene_id_tar: A string indicating the scene id.
        Returns:
            visible_view_object_dict: A dictionary {view_id: visible_object_ids}
            object_ids: A numpy.ndarray of ints indicating the ids of the objects in the scene.
            object_types: A list of strings indicating the types of the objects in the scene.
    """
    pickle_file_val = 'embodiedscan_infos_val_full.pkl'
    pickle_file_train = 'embodiedscan_infos_train_full.pkl'
    from utils.utils_read import read_annotation_pickles
    anno_dict = read_annotation_pickles([pickle_file_val, pickle_file_train])
    keys = sorted(list(anno_dict.keys()))
    for key_index in range(len(keys)):
        key = keys[key_index]
        anno = anno_dict[key]
        image_paths = anno['image_paths']
        dataset, _, scene_id, _ = image_paths[0].split('.')[0].split('/')
        if dataset != dataset_tar or scene_id != scene_id_tar:
            continue
        object_ids = anno['object_ids']
        object_types = anno['object_types']
        visible_view_object_dict = anno['visible_view_object_dict']
        return visible_view_object_dict, object_ids, object_types
    
def prepare_visible_objects(visible_view_object_dict, view_ids, object_ids, object_types):
    """
        Prepares the visible objects for the region annotation.
        Args:
            visible_view_object_dict: A dictionary {view_id: visible_object_ids}
            view_ids: A list of strings, the views used to annotate the region.
            object_ids: A numpy.ndarray of ints indicating the ids of the objects in the scene.
            object_types: A list of strings indicating the types of the objects in the scene.
        Returns:
            visible_object_ids: A list of ints indicating the ids of the visible objects in the region.
            visible_object_types: A list of strings indicating the types of the visible objects in the region.
    """
    visible_object_ids_pre = []
    for view_id in view_ids:
        visible_object_ids_pre += visible_view_object_dict[view_id].tolist()
    visible_object_ids_pre = sorted(list(set(visible_object_ids_pre)))
    visible_object_ids, visible_object_types = [], []
    from utils.utils_read import EXCLUDED_OBJECTS
    for object_id in visible_object_ids_pre:
        object_index = np.where(object_ids == object_id)[0][0]
        object_type = object_types[object_index]
        if object_type in EXCLUDED_OBJECTS:
            continue
        visible_object_ids.append(object_id)
        visible_object_types.append(object_type)
    return visible_object_ids, visible_object_types


def check_forget_id(text, object_ids, object_types):
    from strict_translate import strict_check
    _, item_dict = strict_check(text)
    forget_id = []
    for (object_id, object_type) in zip(object_ids, object_types):
        if f'<{object_type}_{object_id}>' not in item_dict.keys():
            forget_id.append(object_id)
    return forget_id


def check_and_filter_json_files(raw_annos, object_ids, object_types, d_list=DESCRIBE_ASPECT_list,
                                relation_list=RELATION_list):
    try:
        raw_dicts = []
        filter_dicts = []
        raw_annos = list(raw_annos)
        raw_annos[3] = raw_annos[3].replace('[', '(').replace(']', ')')
        for raw_anno in raw_annos:
            a = raw_anno.index('{')
            b = raw_anno.index('}')
            raw_dicts.append(eval(raw_anno[a:b + 1]))
        object_names = []
        for (object_id, object_type) in zip(object_ids, object_types):
            object_names.append(f'<{object_type}_{object_id}>')
    
        # object relation struction
        filter_dict = {}
        for object_pair in raw_dicts[0].keys():
            if (object_pair[0] in object_names) and (object_pair[1] in object_names) and (
                    raw_dicts[0][object_pair] in relation_list):
                filter_dict[object_pair] = raw_dicts[0][object_pair]
        filter_dicts.append(filter_dict)
        filter_dict = {}
        for object_pair in raw_dicts[2].keys():
            if (object_pair[0] in object_names) and (object_pair[1] in object_names):
                filter_dict[object_pair] = raw_dicts[2][object_pair]
        filter_dicts.append(filter_dict)
    
        # object relation struction addition
        filter_dict = {}
    
        wall_objects = raw_dicts[1].get('wall', [])
        floor_objects = raw_dicts[1].get('floor', [])
        filter_dict['wall'] = []
        filter_dict['floor'] = []
        for object_name in wall_objects:
            if object_name in object_names:
                filter_dict['wall'].append(object_name)
        for object_name in floor_objects:
            if object_name in object_names:
                filter_dict['floor'].append(object_name)
    
        filter_dicts.append(filter_dict)
    
        # object function struction
        filter_dict = {}
        for object_name in object_names:
            if object_name in raw_dicts[4].keys() and isinstance(raw_dicts[4][object_name], str):
                filter_dict[object_name] = raw_dicts[4][object_name]
            else:
                filter_dict[object_name] = ''
        filter_dicts.append(filter_dict)
    
        # objects group struction
        filter_dict = {}
        for object_list in raw_dicts[3].keys():
            skip_out = False
            for object_name in object_list:
                if object_name not in object_names:
                    skip_out = True
            if skip_out:
                continue
            filter_dict[object_list] = raw_dicts[3][object_list]
        filter_dicts.append(filter_dict)
    
        # region struction
        filter_dict = {}
        for d_name in d_list:
            if d_name in raw_dicts[5].keys() and isinstance(raw_dicts[5][d_name], str):
                filter_dict[d_name] = raw_dicts[5][d_name]
            else:
                filter_dict[d_name] = ''
        filter_dicts.append(filter_dict)
    
        return filter_dicts
    except:
        return None


from utils.decorators import mmengine_track_func

@mmengine_track_func
def annotate_region(scene_id, region_name, max_additional_tries=4):
    region_type = region_name.split('_')[1]
    # scene_info = all_scene_info[scene_id]
    object_data = annotation_data[scene_id]
    bboxes = object_data['bboxes']
    object_ids = object_data['object_ids']
    object_types = object_data['object_types']
    region_dir = os.path.join('data', scene_id, REGION_VIEW_DIR_NAME, region_name)
    if os.path.exists(os.path.join(region_dir, 'struction.npy')):
        # already annotated
        return

    image_paths = [os.path.join(region_dir, img_name) for img_name in os.listdir(region_dir) if
                   img_name[-4:] == '.jpg']
    # get visible object
    visible_object_ids = np.load(os.path.join(region_dir, 'object_filter.npy'))
    visible_object_types = [object_types[list(object_ids).index(idx)] for idx in visible_object_ids]
    #    prefilter_annotation_path = os.path.join(region_dir, 'struction_prefilter.npy')
    #    if os.path.exists(prefilter_annotation_path):
    #        annotations = np.load(prefilter_annotation_path, allow_pickle=True)
    #    else:
    annotations = annotate_region_image(image_paths, region_type, visible_object_ids, visible_object_types)
    #        np.save(os.path.join(region_dir, 'struction_prefilter.npy'), annotations)

    filter_annos = check_and_filter_json_files(annotations, visible_object_ids, visible_object_types)
    try_ = 0
    while filter_annos==None and try_<max_additional_tries:
        # print(annotations)
        annotations = annotate_region_image(image_paths, region_type, visible_object_ids, visible_object_types)
        filter_annos = check_and_filter_json_files(annotations, visible_object_ids, visible_object_types)
        try_+=1
    # print(f'try {try_} times')
    logging.warning(f"region {region_name} of scene {scene_id} annotated with {try_+1} times.")
        
    # print(filter_annos)
    np.save(os.path.join(region_dir,'struction.npy'), filter_annos)
    if filter_annos==None:
        print(f"region {region_name} of scene {scene_id} annotated failed.")
        logging.warning(f"region {region_name} of scene {scene_id} annotated failed but try {try_} times.")
    else:
        logging.warning(f"region {region_name} of scene {scene_id} annotated successfully.")

if __name__ == "__main__":
    log_file_name = f"{__file__[:-3]}.log"
    print(f"log will be written to {log_file_name}")
    # Set level to warning to avoid writing HTTP requests to log
    logging.basicConfig(filename=log_file_name, level=logging.WARNING, format='%(asctime)s - %(message)s')
    DATA_ROOT = 'data'
    REGION_VIEW_DIR_NAME = 'region_views'
    # choose the scene
    # scene_id = 'scene0000_00'
    # choose the region
    # region_name = '4_toliet region'

    all_scene_info = np.load('all_render_param.npy', allow_pickle=True).item()
    from utils.utils_read import read_annotation_pickles

    annotation_data = read_annotation_pickles(["embodiedscan_infos_train_full.pkl", "embodiedscan_infos_val_full.pkl",
                                               "3rscan_infos_test_MDJH_aligned_full_10_visible.pkl",
                                               "matterport3d_infos_test_full_10_visible.pkl"])
    scene_ids = os.listdir(DATA_ROOT)
    scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith('scene') or scene_id.startswith('1mp3d')]
    scene_ids = sorted(scene_ids)
    # scene_ids = scene_ids[:2]
    tasks = []
    for scene_id in scene_ids:
        dir_to_check = os.path.join(DATA_ROOT, scene_id, REGION_VIEW_DIR_NAME)
        if not os.path.exists(dir_to_check):
            continue
        region_names = os.listdir(dir_to_check)
        region_names = [region_name for region_name in region_names if region_name.endswith('region')]
        region_names = sorted(region_names)
        tasks += [(scene_id, region_name) for region_name in region_names]
    import mmengine
    mmengine.utils.track_parallel_progress(annotate_region, tasks, nproc=2)








