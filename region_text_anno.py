import os
import json
from tqdm import tqdm
from openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups
from utils_read import load_json
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
    num_words = 150 + 10*len(object_ids)
    user_message1 = "I will share photos of a room's {} and use 3D boxes to highlight important items within it. Your task is to provide a description of the primary furniture and decorations found in the area, as well as their respective positions. Please aim for a roughly {}-word description, and when referring to objects, enclose their names and ids in angle brackets, like <piano> <01>.".format(region_type, num_words)
    user_message1 += "In this region, the items that need to be described include "
    for (object_id, object_type) in zip(object_ids, object_types):
        user_message1 += f"<{object_type}> <{object_id}>, "
    user_message1 = user_message1[:-2] + ". Please focus on describing them in order of their significance rather than the order I mentioned them."
    user_message2 = "Based on these layouts, could you share information about how crowded it is, how well it's organized, the lighting, and (optional) whether there are any elements that tell a story?"
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements. You are talking with a high-school student with average knowledge of furniture design."
    source_groups = [
        [user_message1, *image_paths],
        [user_message2]
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    conversation = mimic_chat_budget(content_groups, system_prompt=system_prompt)
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
    pickle_file_val = './example_data/embodiedscan_infos_val_full.pkl'
    pickle_file_train = './example_data/embodiedscan_infos_train_full.pkl'
    from utils_read import read_annotation_pickle
    anno_dict1 = read_annotation_pickle(pickle_file_val)
    anno_dict2 = read_annotation_pickle(pickle_file_train)
    anno_dict = {**anno_dict1, **anno_dict2}
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

if __name__ == "__main__":
    # image_path = "./example_data/anno_lang/painted_images/068_chair_00232.jpg"
    # annotation = annotate_object_image(image_path)
    # for i, line in enumerate(annotation):
    #     print(f"Line {i+1}:")
    #     print(line)
    view_ids = ["01750", "01860", "04600", "04970"]
    region_name = "02_kitchen"
    image_paths = [f"./example_data/anno_lang/regions/{region_name}/{view_id}_annotated.jpg" for view_id in view_ids]
    region_type = "kitchen region"
    visible_view_object_dict, object_ids, object_types = get_visible_objects_dict("scannet", "scene0000_00")
    output_path = f"./example_data/anno_lang/regions/{region_name}/annotation.json"
    visible_object_ids, visible_object_types = prepare_visible_objects(visible_view_object_dict, view_ids, object_ids, object_types)
    if 1==1:
        my_object_ids = [4, 43, 44, 41, 183, 52, 184, 181, 182, 46, 180, 34, 53, 179, 3, 31, 32, 30, 29, 155, 54, 185, 186, 1, 154]
        my_object_ids = sorted(list(set(my_object_ids)))
        my_object_types = []
        for object_id in my_object_ids:
            object_index = np.where(object_ids == object_id)[0][0]
            object_type = object_types[object_index]
            my_object_types.append(object_type)
        visible_object_ids = my_object_ids
        visible_object_types = my_object_types
    annotations = annotate_region_image(image_paths, region_type, visible_object_ids, visible_object_types)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)
    annotations = load_json(output_path)
    print(annotations)
    check_annotation(annotations, visible_object_ids, visible_object_types)
