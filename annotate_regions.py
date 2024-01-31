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

def check_annotation_path(json_dir):
    valid_dict = {}
    num_invalid = 0
    for file_name in os.listdir(json_dir):
        if not file_name.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file_name), "r") as f:
            object_id, object_type, image_id = file_name.split(".")[0].split("_")
            data = json.load(f) 
            is_valid, error_message = check_annotation(data)
            if not is_valid:
                num_invalid += 1
                print(f"Invalid annotation for object {object_id}: {error_message}")
            valid_dict[object_id] = is_valid
    print(f"Checked {len(valid_dict)} annotations. {num_invalid} annotations are invalid.")
    return valid_dict

def check_annotation(annotation):
    """
        Checks if the annotation is valid.
        Args:
            annotation: A list of descriptions. Annotation[0] should be the long version of the description, and Annotation[1] should be the short version.
        Returns:
            is_valid: A boolean indicating if the annotation is valid.
            error_message: A string containing the error message if the annotation is not valid.
    """
    if not isinstance(annotation, list):
        return False, "Type error. Annotation should be a *list* of two strings."
    if len(annotation)!= 2:
        return False, "Number of elements error. Annotation should be a list of *two* strings."
    if not isinstance(annotation[0], str) or not isinstance(annotation[1], str):
        return False, "Type error. Annotation should be a list of two *strings*."
    long_description = annotation[0].lower()
    short_description = annotation[1].lower()
    if "sorry" in long_description or "misunderstand" in long_description:
        return False, "The model may not describe objects accurately or the object we want."
    if len(short_description) > len(long_description):
        return False, "Length error. The shorthand version is longer. Hallucinations are not allowed."
    if len(short_description) > 1200:
        return False, "Length error. The shorthand version is too long."
    return True, ""

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
    visible_object_ids_pre = list(set(visible_object_ids_pre))
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
    view_ids = ["00860", "00970"]
    image_paths = ["./{}_annotated.jpg".format(view_id) for view_id in view_ids]
    region_type = "sleeping region"
    visible_view_object_dict, object_ids, object_types = get_visible_objects_dict("scannet", "scene0000_00")
    # output_path = "./example_data/anno_lang/corpora_region"
    visible_object_ids, visible_object_types = prepare_visible_objects(visible_view_object_dict, view_ids, object_ids, object_types)
    annotations = annotate_region_image(image_paths, region_type, visible_object_ids, visible_object_types)
    with open("annotation.json", "w") as f:
        json.dump(annotations, f, indent=4)
    print(annotations)
