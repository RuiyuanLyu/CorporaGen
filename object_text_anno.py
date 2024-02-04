import os
import json
from tqdm import tqdm
from openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups
from utils_read import load_json


def annotate_objects(image_dir, output_dir, skip_existing=True):
    """
        Uses GPT-4 to annotate objects in a directory of images.
        Returns:
            A dictionary of object annotations, where the keys are object ids and the values are lists of object descriptions.
    """
    annotations = {}
    file_names = []
    object_ids = []
    for file_name in os.listdir(image_dir):
        if not file_name.endswith(".jpg"):
            continue
        object_id, object_type, image_id = file_name.split("_") # example file name: 068_chair_00232.jpg
        # the line is used to prevent unwanted files from being annotated
        file_names.append(file_name)
    pbar = tqdm(range(len(file_names)))
    for i in pbar:
        file_name = file_names[i]
        object_id, object_type, image_id = file_name.split(".")[0].split("_")
        image_path = os.path.join(image_dir, file_name)
        json_path = os.path.join(output_dir, f"{object_id}_{object_type}_{image_id}.json")
        pbar.set_description(f"Annotating object {object_id}")
        if skip_existing and os.path.exists(json_path):
            annotation = load_json(json_path)
            is_valid, error_message = check_annotation(annotation)
            if is_valid:
                annotations[object_id] = annotation
                print(f"Skipping existing annotation for object {object_id}")
                continue
        annotation = annotate_object_image(image_path)
        annotations[object_id] = annotation
        with open(json_path, "w") as f:
            json.dump(annotation, f, indent=4)
    return annotations


def annotate_object_image(image_path):
    """
        Uses GPT-4 to annotate an object in an image.
        Returns:
            The discription of the object in the image.
    """
    image_path = image_path.replace("\\", "/")
    image_name = image_path.split("/")[-1]
    object_id, object_type, image_id = image_name.split(".")[0].split("_")
    
    user_message1 = "Please describe the {} in the highlighted box, mainly including the following aspects: appearance (shape, color), material, size (e.g., larger or smaller compared to similar items), condition (e.g., whether a door is open or closed), placement (e.g.,vertical/leaning/slanting/stacked), functionality (compared to similar items), and design features (e.g., whether a chair has armrests/backrest)".format(object_type) 
    user_message2 = "Please omit the plain and ordinary parts of the description, only retaining the unique characteristics of the objects; rewrite and recombine the retained descriptions to make the language flow naturally, without being too rigid. Make the descriptions about 150 words."
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements."
    source_groups = [
        [user_message1, image_path],
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


if __name__ == "__main__":
    # image_path = "./example_data/anno_lang/painted_images/068_chair_00232.jpg"
    # annotation = annotate_object_image(image_path)
    # for i, line in enumerate(annotation):
    #     print(f"Line {i+1}:")
    #     print(line)
    image_dir = "./example_data/anno_lang/painted_images"
    output_dir = "./example_data/anno_lang/corpora_object"
    annotations = annotate_objects(image_dir, output_dir, skip_existing=True)
    check_annotation_path(output_dir)
