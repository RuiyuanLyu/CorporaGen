import os
import json
from tqdm import tqdm
from utils.openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups, num_tokens_from_string, get_full_response
from utils.utils_read import load_json
from utils.decorators import mmengine_track_func

def annotate_objects_by_directory(image_dir, output_dir, skip_existing=True, force_invalid=True, max_additional_attempts=0, pick_ids=None, with_highlight=True, high_detail=False):
    """
        Uses GPT-4 to annotate objects in a directory of images.
        Args:
            image_dir: A string of the path to the directory of images.
            output_dir: A string of the path to the directory to save the annotations.
            skip_existing: True if existing annotations should be skipped.
            force_invalid: True if existing invalid annotations should be forced to be re-annotated.
            max_additional_attempts: An integer of the maximum number of additional attempts to generate a valid annotation.
            pick_ids: A list of object ids to be annotated. Set to None to annotate all objects in the directory.
            with_highlight: True if the image is painted with a highlighted box. (as opposed to the object is cropped from the image)
        Returns:
            A dictionary of object annotations, where the keys are object ids and the values are lists of object descriptions.
    """
    file_names = []
    if pick_ids is not None:
        pick_ids = set(pick_ids)
    for file_name in os.listdir(image_dir):
        if not file_name.endswith(".jpg"):
            continue
        object_id = file_name.split("_")[0] # example file name: 068_chair_00232.jpg
        # the line is used to prevent unwanted files from being annotated
        if pick_ids is not None and int(object_id) not in pick_ids:
            continue
        file_names.append(file_name)
    inputs = [(file_name, image_dir, output_dir, skip_existing, force_invalid, max_additional_attempts, with_highlight, high_detail) for file_name in file_names]
    import mmengine
    results = mmengine.track_parallel_progress(annotate_object, inputs, nproc=8)
    return results

def annotate_object_parallel(inputs):
    return annotate_object(*inputs)

def annotate_object(file_name, image_dir, output_dir, skip_existing, force_invalid, max_additional_attempts, with_highlight=True, high_detail=False):
    object_id, object_type, image_id = file_name.split(".")[0].split("_")
    image_path = os.path.join(image_dir, file_name)
    json_path = os.path.join(output_dir, f"{object_id}_{object_type}_{image_id}.json")
    if skip_existing and os.path.exists(json_path):
        annotation = load_json(json_path)
        is_valid, error_message = check_annotation_validity(annotation)
        if is_valid or not force_invalid:
            print(f"Skipping existing annotation for object {object_id}")
            return annotation
    annotation = annotate_object_by_image(image_path, max_additional_attempts, with_highlight=with_highlight, high_detail=high_detail)
    with open(json_path, "w") as f:
        json.dump(annotation, f, indent=4)
    return annotation

def annotate_object_by_image(image_path, max_additional_attempts=0, with_highlight=True, high_detail=False):
    """
        Uses GPT-4 to annotate an object in an image.
        Args:
            image_path: A string of the path to the image.
            max_additional_attempts: An integer of the maximum number of additional attempts to generate a valid annotation.
            with_highlight: True if the image is painted with a highlighted box. (as opposed to the object is cropped from the image)
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
    
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements. You are visiting some ordinary rooms that conform to the daily life of an average person. The expected reader is a high-school student with average knowledge of furniture design."
    if with_highlight:
        user_message1 = "Please describe the {} in the highlighted box,".format(object_type)
    else:
        user_message1 = "Please describe the {} in the middle of the image,".format(object_type)
    user_message1 += " mainly including the following aspects: appearance (shape, color), material, size (e.g., larger or smaller compared to similar items), condition (e.g., whether a door is open or closed), placement (e.g.,vertical/leaning/slanting/stacked), functionality (compared to similar items), and design features (e.g., whether a chair has armrests/backrest). Please aim for a roughly 300-word description,"
    if with_highlight:
        user_message1 += " and do not mention the highlight box in the description."
    else:
        user_message1 += " and do not use expressions like 'the middle of the image' in the description."
    user_message2 = "Please omit the plain and ordinary parts of the description, only retaining the unique characteristics of the objects; rewrite and recombine the retained descriptions to make the language flow naturally, without being too rigid. Make the description about 150 words."
    source_groups = [
        [user_message1, image_path],
        [user_message2]
    ]
    content_groups = get_content_groups_from_source_groups(source_groups, high_detail=high_detail)
    # conversation = mimic_chat(content_groups, model="gpt-4-vision-preview", system_prompt=system_prompt)
    conversation = mimic_chat_budget(content_groups, system_prompt=system_prompt, max_additional_attempts=max_additional_attempts)
    raw_annotation = []
    for message in conversation:
        if message["role"] == "assistant":
            raw_annotation.append(message["content"])
    if len(raw_annotation) == 0:
        # In case the GPT-model refusing to generate any output. (triggered by safety mechanism)
        return {"original_description": "", "simplified_description": ""}
    if len(raw_annotation) == 1:
        # In case the GPT-model only generates one output. (triggered by safety mechanism)
        return {"original_description": raw_annotation[0], "simplified_description": ""}
    
    annotation = {"original_description": raw_annotation[0], "simplified_description": raw_annotation[1]}
    return annotation

def translate(text, src_lang="English", tgt_lang="Chinese", object_type_hint=None):
    """
        Translates text using the OpenAI API.
        Args:
            text: A string of text to be translated.
            src_lang: A string of the source language code.
            tgt_lang: A string of the target language code.
        Returns:
            A string of the translated text.
    """
    user_message = text
    src_lang = src_lang.capitalize()
    tgt_lang = tgt_lang.capitalize()
    system_prompt = f"You are an excellent translator, who does more than rigidly translating {src_lang} into {tgt_lang}. Your choice of words and phrases is natural and fluent. The expressions are easy to understand. The expected reader is a middle-school student."
    if object_type_hint is not None:
        system_prompt += f" The user will provide a description of a {object_type_hint}."
    source_groups = [
        [user_message],
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    conversation = mimic_chat(content_groups, model="gpt-3.5-turbo-0125", system_prompt=system_prompt)
    for message in conversation:
        if message["role"] == "assistant":
            return message["content"]

def summarize(text):
    system_prompt = "You are an expert interior designer, who is very sensitive at room furnitures and their placements. You are visiting some ordinary rooms that conform to the daily life of an average person. The expected reader is a high-school student with average knowledge of furniture design."
    user_message1 = "Please describe the object in the middle of the image, mainly including the following aspects: appearance (shape, color), material, size (e.g., larger or smaller compared to similar items), condition (e.g., whether a door is open or closed), placement (e.g.,vertical/leaning/slanting/stacked), functionality (compared to similar items), and design features (e.g., whether a chair has armrests/backrest). Please aim for a roughly 300-word description."
    user_message2 = "Please omit the plain and ordinary parts of the description, only retaining the unique characteristics of the objects; rewrite and recombine the retained descriptions to make the language flow naturally, without being too rigid. Make the description about 150 words."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message1},
        {"role": "assistant", "content": text},
        {"role": "user", "content": user_message2}
    ]

    return get_full_response(messages, model="gpt-3.5-turbo-0125")

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
    short_description = annotation.get("simplified_description", "").lower()
    accuracy_dict = annotation.get("accuracy_dict", None)
    # if accuracy_dict:
    #     meta = map_text_to_bool(accuracy_dict.get("meta", "False"))
    #     if not meta:
    #         return False, "The model is not describing the object we want."
    if len(long_description) < 100:
        return False, "The description is too short."
    if "sorry" in long_description or "misunderst" in long_description:
        return False, "The model may not describe objects accurately or the object we want."
    if len(short_description) > len(long_description):
        return False, "Length error. The shorthand version is longer. Hallucinations are not allowed."
    return True, ""

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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

choice_text_mapping = {
    "是": "True",
    "否": "False",
    "该物体没有这一属性": "Inrelevant",
    "该物体具有这一属性，描述遗漏了": "Missing",
    "不要选这个选项": None # special value if the user is lazy.
}

def map_CNchoice_to_ENtext(choice):
    if choice in choice_text_mapping:
        return choice_text_mapping[choice]
    else:
        raise ValueError(f"Invalid choice: {choice}")
def map_ENtext_to_CNchoice(text):
    if isinstance(text, bool):
        return "是" if text else "否"
    for key in choice_text_mapping:
        if choice_text_mapping[key] == text:
            return key
    raise ValueError(f"Invalid text: {text}")

text_bool_mapping = {
    "True": True,
    "False": False,
    "Inrelevant": True,
    "Missing": False
}

def map_text_to_bool(text):
    if isinstance(text, bool):
        return text
    if text in text_bool_mapping:
        return text_bool_mapping[text]
    else:
        raise ValueError(f"Invalid text: {text}")

@DeprecationWarning
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
    quality_dict = {"meta":0, "visual_info_sufficient":0, "category":0, "appearance":0, "material":0, "size":0, "state":0, "position":0, "placement":0, "special_function":0, "other_features":0}
    for annotation in annotations:
        assert isinstance(annotation, dict)
        if "accuracy_dict" not in annotation:
            continue
        accuracy_dict = annotation["accuracy_dict"]
        for key in quality_dict:
            if key in accuracy_dict:
                quality_dict[key] += map_text_to_bool(accuracy_dict[key]) and accuracy_dict.get("visual_info_sufficient", 1)
            elif key == "visual_info_sufficient":
                quality_dict[key] += 1
    for key in quality_dict:
        quality_dict[key] /= len(annotations)
    for key in quality_dict:
        if key == "meta":
            continue
        if key == "visual_info_sufficient":
            continue
        quality_dict[key] /= quality_dict["meta"]

    if display_stats:
        sum = 0
        for key in quality_dict:
            print("{}: {:.2f}%".format(key, quality_dict[key]*100))
            if key in ["meta", "visual_info_sufficient"]:
                continue
            sum += quality_dict[key] 
        print("Overall quality meta: {:.2f}%".format(quality_dict["meta"]*100))
        print("Overall quality other: {:.2f}%".format(sum/(len(quality_dict)-2)*100))
        print("Overall quality meta*other: {:.2f}%".format((quality_dict["meta"] * sum/(len(quality_dict)-2)) * 100))
    return quality_dict

@mmengine_track_func
def translate_annotation_from_file(json_path, src_lang="English", tgt_lang="Chinese", force_translate=False):
    """
        Translates the "original_description" field of an annotation from a file.
        Returns:
            A dictionary of the translated annotation.
    """
    if not json_path.endswith(".json"):
        return
    annotation = load_json(json_path)
    if not annotation:
        return
    is_valid, error_message = check_annotation_validity(annotation)
    if not is_valid:
        return
    if "translated_description" in annotation and len(annotation["translated_description"]) > 1 and not force_translate:
        return annotation
    annotation_to_translate = annotation["original_description"]
    if "simplified_description" in annotation:
        annotation_to_translate = annotation["simplified_description"]
    if len(annotation_to_translate) <= 1:
        return
    try:
        translated_description = translate(annotation_to_translate, src_lang=src_lang, tgt_lang=tgt_lang)
    except Exception as e:
        print(f"Error translating {json_path}: {e}")
        return
    annotation["translated_description"] = translated_description
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=4)
    return annotation

@mmengine_track_func
def summarize_annotation_from_file(json_path, force_summarize=False):
    """
        Summarizes the "original_description" field of an annotation from a file.
        Returns:
            A dictionary of the summarized annotation.
    """
    if not json_path.endswith(".json"):
        return
    # check if the file is empty
    with open(json_path, "r", encoding="utf-8") as f:
        if f.read().strip() == "":
            return
    annotation = load_json(json_path)
    is_valid, error_message = check_annotation_validity(annotation)
    # if not is_valid:
    #     return
    text = annotation.get("original_description", "")
    if not text:
        return
    if num_tokens_from_string(text) < 200:
        return 
    if "simplified_description" in annotation and not force_summarize:
        return annotation
    summary = summarize(text)
    annotation["simplified_description"] = summary
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=4)
    return annotation

with open("expressions_to_remove.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    EXPRESSION_TO_REMOVE = [line.strip() for line in lines]

def remove_specific_expressions(text):
    if not isinstance(text, str):
        return text
    for expression in EXPRESSION_TO_REMOVE:
        text = text.replace(expression, "")
    text = text.replace("  ", " ")
    text = text.strip()
    return text

def remove_specific_expressions_from_description(json_path):
    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Failed to load {json_path}")
            return None
    if isinstance(data, dict):
        if data.get("modified_description", ""):
            return None
        description = data.get("original_description", "")
        if description:
            data["original_description"] = remove_specific_expressions(description)
        description = data.get("simplified_description", "")
        if description:
            data["simplified_description"] = remove_specific_expressions(description)
    elif isinstance(data, list):
        data = [remove_specific_expressions(d) for d in data]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


DATA_ROOT = "./data"
if __name__ == "__main__":
    scene_ids = ["1mp3d_0001_region8", "1mp3d_0002_region23", "3rscan0138", "3rscan0036", "scene0026_00", "scene0094_00", "scene0147_00"]
    # SCENE_ID = "scene0000_00"
    # image_dir = os.path.join(DATA_ROOT, SCENE_ID, "painted_objects")
    # output_dir = os.path.join(DATA_ROOT, SCENE_ID, "corpora_object_XComposer2_crop/user_test")
    # os.makedirs(output_dir, exist_ok=True)
    # output_dir = os.path.join(DATA_ROOT, SCENE_ID, "corpora_object_gpt4v_paint")
    # output_dir = os.path.join(DATA_ROOT, SCENE_ID, "corpora_object_gpt4v_crop")
    # output_dir = os.path.join(DATA_ROOT, SCENE_ID, "corpora_object_cogvlm_crop")
    # os.makedirs(output_dir, exist_ok=True)

    ###################################################################
    ## Annotation usage here.
    # my_ids = [3, 5, 8, 13, 15, 18, 19, 30, 54, 59, 60, 66, 68, 147, 150, 159, 168, 172, 178, 180]
    # for scene_id in scene_ids:
    #     image_dir = os.path.join(DATA_ROOT, scene_id, "repainted_objects")
    #     output_dir = os.path.join(DATA_ROOT, scene_id, "corpora_object_gpt4v_paint_highdetail")
    #     os.makedirs(output_dir, exist_ok=True)
    #     annotations = annotate_objects_by_directory(image_dir, output_dir, skip_existing=False, force_invalid=False, max_additional_attempts=1, pick_ids = None, with_highlight=True, high_detail=True)
    # ATTENTION: MODIFY with_highlight 
    # check_annotation_validity_path(output_dir)

    ###################################################################
    ## Remove specific expressions usage here.
    ## NOTE: use this before summarization.
    # tasks = []
    # for root, dirs, files in os.walk(DATA_ROOT):
    #     for file in files:
    #         if file.endswith(".json"):
    #             tasks.append(os.path.join(root, file))

    # import mmengine
    # mmengine.track_parallel_progress(remove_specific_expressions_from_description, tasks, nproc=8)

    ##################################################################
    ## Summarization usage here.
    # my_ids = [3, 5, 8, 13, 15, 18, 19, 30, 54, 59, 60, 66, 68, 147, 150, 159, 168, 172, 178, 180]
    # my_ids = set(my_ids)
    # file_names = [file for file in os.listdir(output_dir) if file.endswith(".json") and int(file.split("_")[0]) in my_ids]
    # scene_ids = os.listdir(DATA_ROOT)
    # scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith("1mp3d")]
    # inputs = []
    corpora_strings = ["corpora_object_InternVL-Chat-V1-2-Plus_crop", "corpora_object_cogvlm_crop"][:]
    # for scene_id in scene_ids:
    #     for corpora_string in corpora_strings:
    #         output_dir = os.path.join(DATA_ROOT, scene_id, corpora_string)
    #         if not os.path.exists(output_dir):
    #             continue
    #         file_names = [file for file in os.listdir(output_dir) if file.endswith(".json")]
    #         json_paths = [os.path.join(output_dir, file_name) for file_name in file_names]
    #         # False means force_summarize=False
    #         inputs.extend([(json_path, False) for json_path in json_paths])
    # import mmengine
    # results = mmengine.track_parallel_progress(summarize_annotation_from_file, inputs, nproc=8)

    ##################################################################
    ## Translate usage here.
    # my_ids = [3, 5, 8, 13, 15, 18, 19, 30, 54, 59, 60, 66, 68, 147, 150, 159, 168, 172, 178, 180]
    # my_ids = set(my_ids)
    # output_dir = os.path.join(DATA_ROOT, "scene0000_00", "corpora_object_InternVL-Chat-V1-2-Plus_crop")
    # file_names = [file for file in os.listdir(output_dir) if file.endswith(".json") and int(file.split("_")[0]) in my_ids]
    scene_ids = os.listdir(DATA_ROOT)
    scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith("1mp3d")]
    corpora_strings = ["corpora_object_InternVL-Chat-V1-2-Plus_crop", "corpora_object_cogvlm_crop"][:1]
    inputs = []
    for scene_id in scene_ids:
        for corpora_string in corpora_strings:
            output_dir = os.path.join(DATA_ROOT, scene_id, corpora_string)
            if not os.path.exists(output_dir):
                continue
            file_names = [file for file in os.listdir(output_dir) if file.endswith(".json")]
            json_paths = [os.path.join(output_dir, file_name) for file_name in file_names]
            # False means force_translate=False
            inputs.extend([(json_path, "English", "Chinese", False) for json_path in json_paths])
    import mmengine
    results = mmengine.track_parallel_progress(translate_annotation_from_file, inputs, nproc=8)
    # for input_tuple in inputs:
    #     translate_annotation_from_file(input_tuple)
