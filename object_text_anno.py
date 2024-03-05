import os
import json
from tqdm import tqdm
from openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups
from utils_read import load_json


def annotate_objects(image_dir, output_dir, skip_existing=True, force_invalid=True, max_additional_attempts=0):
    """
        Uses GPT-4 to annotate objects in a directory of images.
        Args:
            image_dir: A string of the path to the directory of images.
            output_dir: A string of the path to the directory to save the annotations.
            skip_existing: True if existing annotations should be skipped.
            force_invalid: True if existing invalid annotations should be forced to be re-annotated.
            max_additional_attempts: An integer of the maximum number of additional attempts to generate a valid annotation.
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
            is_valid, error_message = check_annotation_validity(annotation)
            if is_valid or not force_invalid:
                annotations[object_id] = annotation
                print(f"Skipping existing annotation for object {object_id}")
                continue
        annotation = annotate_object_image(image_path, max_additional_attempts)
        annotations[object_id] = annotation
        with open(json_path, "w") as f:
            json.dump(annotation, f, indent=4)
    return annotations


def annotate_object_image(image_path, max_additional_attempts=0):
    """
        Uses GPT-4 to annotate an object in an image.
        Args:
            image_path: A string of the path to the image.
            max_additional_attempts: An integer of the maximum number of additional attempts to generate a valid annotation.
        Returns:
            A dict of plain text descriptions.
            "original_description": A string of the original description.
            "simplified_description": A string of the simplified description.
            "translated_extension": A string of the translated original description. (optional)
            "translated_description": A string of the translated simplified description. (optional)
    """
    image_path = image_path.replace("\\", "/")
    image_name = image_path.split("/")[-1]
    object_id, object_type, image_id = image_name.split(".")[0].split("_")
    
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements. You are visiting some ordinary rooms that conform to the daily life of an average person. You neither overly boast about the items in the room nor harshly criticize them. Instead, you use your professional expertise to truthfully point out their various aspects. The expected reader is a high-school student with average knowledge of furniture design."
    user_message1 = "Please describe the {} in the highlighted box, mainly including the following aspects: appearance (shape, color), material, size (e.g., larger or smaller compared to similar items), condition (e.g., whether a door is open or closed), placement (e.g.,vertical/leaning/slanting/stacked), functionality (compared to similar items), and design features (e.g., whether a chair has armrests/backrest). Please aim for a roughly 300-word description".format(object_type) 
    user_message2 = "Please omit the plain and ordinary parts of the description, only retaining the unique characteristics of the objects; rewrite and recombine the retained descriptions to make the language flow naturally, without being too rigid. Make the descriptions about 150 words."
    source_groups = [
        [user_message1, image_path],
        [user_message2]
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    conversation = mimic_chat_budget(content_groups, system_prompt=system_prompt, max_additional_attempts=max_additional_attempts)
    raw_annotation = []
    for message in conversation:
        if message["role"] == "assistant":
            raw_annotation.append(message["content"])
    annotation = {"original_description": raw_annotation[0], "simplified_description": raw_annotation[1]}
    return annotation

def translate_to_chinese(text):
    """
        Translates text to Chinese using the OpenAI API.
        Args:
            text: A string of text to be translated.
        Returns:
            A string of the translated text.
    """
    user_message = text
    system_prompt = "You are an excellent translator, who does more than rigidly translating English into Chinese. Your choice of words and phrases are natural and fluent. The expressions are easy to understand. The expected reader is a middle-school student."
    source_groups = [
        [user_message],
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    conversation = mimic_chat(content_groups, model="gpt-3.5-turbo-0125", system_prompt=system_prompt)
    for message in conversation:
        if message["role"] == "assistant":
            return message["content"]

def check_annotation_validity_path(json_dir):
    valid_dict = {}
    num_invalid = 0
    for file_name in os.listdir(json_dir):
        if not file_name.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file_name), "r", encoding="utf-8") as f:
            object_id, object_type, image_id = file_name.split(".")[0].split("_")
            data = json.load(f) 
            is_valid, error_message = check_annotation_validity(data)
            if not is_valid:
                num_invalid += 1
                print(f"Invalid annotation for object {object_id}: {error_message}")
            valid_dict[object_id] = is_valid
    print(f"Checked {len(valid_dict)} annotations. {num_invalid} annotations are invalid.")
    return valid_dict

def check_annotation_validity(annotation):
    """
        Checks if the annotation is valid.
        Args:
            A dictionary containing the keys:
                 "original_description", "simplified_description", "translated_description", "modified_description", and "accuracy_dict".
        Returns:
            is_valid: A boolean indicating if the annotation is valid.
            error_message: A string containing the error message if the annotation is not valid.
    """
    assert isinstance(annotation, dict)
    long_description = annotation["original_description"].lower()
    short_description = annotation["simplified_description"].lower()
    if "sorry" in long_description or "misunderst" in long_description:
        return False, "The model may not describe objects accurately or the object we want."
    if len(short_description) > len(long_description):
        return False, "Length error. The shorthand version is longer. Hallucinations are not allowed."
    return True, ""

def map_choice_to_bool(choice):
    if choice == "是":
        return True
    elif choice == "否":
        return False
    elif choice == "该物体没有这一属性":
        return True
    elif choice == "该物体具有这一属性，描述遗漏了":
        return False
    elif choice == "不要选这个选项":
        return None # special value if the user is lazy.
    else:
        raise ValueError(f"Invalid choice: {choice}")    
def map_choice_to_text(choice):
    if choice == "是":
        return "True"
    elif choice == "否":
        return "False"
    elif choice == "该物体没有这一属性":
        return "Inrelevant"
    elif choice == "该物体具有这一属性，描述遗漏了":
        return "Missing"
    elif choice == "不要选这个选项":
        return None # special value if the user is lazy.
    else:
        raise ValueError(f"Invalid choice: {choice}")
def map_text_to_bool(text):
    if isinstance(text, bool):
        return text
    if text == "True":
        return True
    elif text == "False":
        return False
    elif text == "Inrelevant":
        return True
    elif text == "Missing":
        return False
    else:
        raise ValueError(f"Invalid text: {text}")


def check_annotation_quality(annotations, display_stats=True):
    """
        Reads the accuracy_dict from the annotations and calculates the quality for a list of annotations. 
        Args:
            a list of dictionaries containing the keys:
                 "original_description", "simplified_description", "translated_description", "modified_description", and "accuracy_dict".
            The accuracy_dict is a dictionary containing the keys: "meta", "category", "appearance", "material", "size", "state", "position", "placement", "special_function", "other_features". The values are in ["True", "False", "Inrelevant", "Missing"]
        Returns:
            A dict containing the keys:
                "meta", "category", "appearance", "material", "size", "state", "position", "placement", "special_function", "other_features"
    """
    quality_dict = {"meta":0, "category":0, "appearance":0, "material":0, "size":0, "state":0, "position":0, "placement":0, "special_function":0, "other_features":0}
    for annotation in annotations:
        assert isinstance(annotation, dict)
        if "accuracy_dict" not in annotation:
            continue
        accuracy_dict = annotation["accuracy_dict"]
        for key in quality_dict:
            if key in accuracy_dict:
                quality_dict[key] += map_text_to_bool(accuracy_dict[key])
    for key in quality_dict:
        quality_dict[key] /= len(annotations)
    for key in quality_dict:
        if key == "meta":
            continue
        quality_dict[key] /= quality_dict["meta"]

    if display_stats:
        sum = 0
        for key in quality_dict:
            if key == "meta":
                continue
            print("{}: {:.2f}%".format(key, quality_dict[key]*100))
            sum += quality_dict[key] 
        print("Overall quality meta: {:.2f}%".format(quality_dict["meta"]*100))
        print("Overall quality other: {:.2f}%".format(sum/(len(quality_dict)-1)*100))
        print("Overall quality meta * other: {:.2f}%".format((quality_dict["meta"] * sum/(len(quality_dict)-1)) * 100))
    return quality_dict

if __name__ == "__main__":
    image_dir = "./example_data/anno_lang/painted_images"
    output_dir = "./example_data/anno_lang/corpora_object"
    # annotations = annotate_objects(image_dir, output_dir, skip_existing=True, force_invalid=True, max_additional_attempts=3)
    check_annotation_validity_path(output_dir)
    # for file_name in tqdm(os.listdir(output_dir)):
    #     if not file_name.endswith(".json"):
    #         continue
    #     annotation = load_json(os.path.join(output_dir, file_name))
    #     is_valid, error_message = check_annotation_validity(annotation)
    #     if not is_valid:
    #         continue
    #     if "translated_description" in annotation:
    #         continue
    #     # print("Short description: ", annotation["simplified_description"])
    #     # print("Original description: ", annotation["original_description"])
    #     translated_description = translate_to_chinese(annotation["simplified_description"])
    #     # print(f"Translated description: {translated_description}")
    #     annotation["translated_description"] = translated_description
    #     with open(os.path.join(output_dir, file_name), "w") as f:
    #         json.dump(annotation, f, indent=4)

    annotations = []
    for file_name in os.listdir(output_dir):
        if not file_name.endswith(".json"):
            continue
        annotation = load_json(os.path.join(output_dir, file_name))
        annotations.append(annotation)
    quality_dict = check_annotation_quality(annotations)
    # print(quality_dict)