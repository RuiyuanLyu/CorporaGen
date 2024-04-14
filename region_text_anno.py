import os
import json
from tqdm import tqdm
from openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups
from utils_read import load_json
from strict_translate import strict_translate
import numpy as np


def annotate_region_image(image_paths, region_type, object_ids, object_types):
    """
        Uses GPT-4 to annotate an object in an image.
        Args:
            image_paths: A list of paths to images cooresponding to the single region.
            region_type: A string indicating the type of the area, e.g. "bedroom".
            object_ids: A list of ints indicating the ids of the objects in the image.
            object_types: A list of strings indicating the types of the objects in the image.
        Returns:
            A list of strings, each string is a part of the annotation.
    """    
    num_words = (150 + 10*len(object_ids))/5
    user_message1 = "I will share photos of a room's {} and use 3D boxes to highlight important items within it. Your task is to provide a description of the primary items in the area as well as their respective positions(And please describe their relative position, if any).What's more,noting their roles in the region(what contribution do they make to the region?). Please aim for a roughly {}-word description, and when referring to a object, enclose its name and id in an angle bracket, like <pillow_17>.Please not miss any items!".format(region_type,num_words)
    user_message1 += "In this region, the items that must be described include "
    for (object_id, object_type) in zip(object_ids, object_types):
        user_message1 += f"<{object_type}_{object_id}>, "
    user_message1 = user_message1[:-2] + ",don't leave out any of them! Please focus on describing them in order of their significance rather than the order I mentioned them."
    user_message2 = 'Further, I would like you to give a comprehensive description of the relationships between these objects:Which of these items together provides what function (not single but together,e.g. book and pen provide the function of writing).'
    user_message3 = "Based on these layouts, could you share information about the region,these should be included:its location and function description, its space layout and dimensions, its Doors and Windows, walls, floors and ceilings,its soft fitting elements and decorative details,its lighting design,its color matching and style theme and its special features."
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements. The expected reader is a high-school student with average knowledge of furniture design."

    source_groups = [
        [user_message1, *image_paths],
        [user_message2, *image_paths],
        [user_message3, *image_paths]

    ]
    content_groups = get_content_groups_from_source_groups(source_groups) #high detail
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    conversation = mimic_chat_budget(content_groups, system_prompt=system_prompt, num_turns_expensive=3)
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
    from utils_read import read_annotation_pickles
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
    from utils_read import EXCLUDED_OBJECTS
    for object_id in visible_object_ids_pre:
        object_index = np.where(object_ids == object_id)[0][0]
        object_type = object_types[object_index]
        if object_type in EXCLUDED_OBJECTS:
            continue
        visible_object_ids.append(object_id)
        visible_object_types.append(object_type)
    return visible_object_ids, visible_object_types

def check_forget_id(text,object_ids,object_types):
    from strict_translate import strict_check
    _, item_dict = strict_check(text)
    forget_id = []
    for (object_id, object_type) in zip(object_ids, object_types):
        if f'<{object_type}_{object_id}>' not in item_dict.keys():
            forget_id.append(object_id)
    return forget_id




if __name__ == "__main__":

    from region_matching import get_data,process_data

    # choose the scene
    scene_id = 'scene0000_00'
    region_view_dir_name = 'region_view_test'
    # choose the region
    region_name = '4_toliet region'
    region_type = region_name.split('_')[1]

    all_scene_info = np.load('all_render_param.npy', allow_pickle=True).item()
    scene_info = all_scene_info[scene_id]
    from utils_read import read_annotation_pickles

    annotation_data = read_annotation_pickles(["embodiedscan_infos_train_full.pkl", "embodiedscan_infos_val_full.pkl",
                                               "3rscan_infos_test_MDJH_aligned_full_10_visible.pkl",
                                               "matterport3d_infos_test_full_10_visible.pkl"])

    object_data = annotation_data[scene_id]
    bboxes = object_data['bboxes']
    object_ids = object_data['object_ids']
    object_types = object_data['object_types']

    image_paths = [f'data/{scene_id}/{region_view_dir_name}/{region_name}/'+img_name for img_name in  os.listdir(f'data/{scene_id}/{region_view_dir_name}/{region_name}') if img_name[-4:]=='.jpg']
    # get visible object
    visible_object_ids = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_name}/object_filter.npy')
    visible_object_types = [object_types[list(object_ids).index(idx)] for idx in visible_object_ids ]
    annotations_out = annotate_region_image(image_paths, region_type, visible_object_ids, visible_object_types)
    annotations = annotations_out

    code_check_info = dict()
    code_check_info['translation_success'] = []



    # Save English and Chinese output


    annotations_tran = []
    for index_ in range(3):

        out_of_it,success = strict_translate(annotations[index_],src_lang="English", tgt_lang="Chinese",show_out=False)
        print(out_of_it,success)
        code_check_info['translation_success'].append(success)
        annotations_tran.append(out_of_it)

    English_json = json.dumps({'object': annotations[0], 'group': annotations[1], 'region': annotations[2]})
    Chinese_json = json.dumps({'object': annotations_tran[0], 'group': annotations_tran[1], 'region': annotations_tran[2]})

    with open(f'data/{scene_id}/{region_view_dir_name}/{region_name}/English.json', 'w') as f:
        f.write(English_json)
    with open(f'data/{scene_id}/{region_view_dir_name}/{region_name}/Chinese.json', 'w', encoding='utf-8') as f:
        f.write(Chinese_json)

    code_check_info['forget_id'] = check_forget_id(annotations[0],object_ids=visible_object_ids,object_types=visible_object_types)
    print(code_check_info)
    np.save(f'data/{scene_id}/{region_view_dir_name}/{region_name}/region_text_anno_info.npy',code_check_info)




