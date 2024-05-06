import os
import json
from utils.utils_read import read_annotation_pickle
from object_text_anno import translate
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups
import mmengine
DATA_ROOT = "data/"
SPLIT_DATA_ROOT = "splitted_infos/"
example_dict = {
    "scan_id": "scannet/scene0072_01",
    "target_id": 20,
    "distractor_ids": [
        8,
        28,
        29
    ],
    "text": "find the pillow that is on top of the shelf",
    "target": "pillow",
    "anchors": [
        "shelf"
    ],
    "anchor_ids": [
        22
    ],
    "tokens_positive": [
        [
            9,
            15
        ]
    ]
}
REFERING_TYPES = [
    "fine grained category",
    "coarse grained category",
    "color",
    "texture",
    "material",
    "weight",
    "size",
    "shape",
    "placement",
    "state",
    "function",
    "other features"
]

def check_object_text_anno_is_valid(json_data):
    accuracy_dict = json_data.get("accuracy_dict", {})
    if not (accuracy_dict.get("meta", "") and accuracy_dict.get("visual_info_sufficient", "")):
        return False
    return True
    
# def generate_object_type_reference(scene_id):
def back_translate_text_anno_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    object_type = os.path.basename(json_path).split("_")[1]
    if json_data.get("modified_description_en", ""):
        return
    if not check_object_text_anno_is_valid(json_data):
        # The description is not accurate enough. Skip.
        return 
    modified_description = json_data.get("modified_description", "")
    if modified_description:
        back_translated_description = translate(modified_description, "zh", "en", object_type_hint=object_type)
        json_data["modified_description_en"] = back_translated_description
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def extract_attribute_from_text(text):
    system_prompt = "You are given a text description of an object. Your task is to identify the attributes of the object. The 12 attributes are: fine grained category, coarse grained category, color, texture, material, weight, size, shape, placement (e.g. upright, piled up), state (e.g. open, locked, empty), function, and other features. Missing attributes can be left blank. Please reply in json, in the following format: {'function': 'blocks the light from entering the room', 'state': 'closed'}."
    content_groups = get_content_groups_from_source_groups([text])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0, report_token_usage=True, json_mode=True)
    response = messages[-1]
    assert response["role"] == "assistant"
    response_dict = json.loads(response["content"])
    response_dict = {k.lower().replace(" ", "_"): v for k, v in response_dict.items()}
    return response_dict

def extract_attribute_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    text = json_data.get("modified_description_en", "")
    if not text:
        return
    if json_data.get("attributes", ""):
        return
    response_dict = extract_attribute_from_text(text)
    # import pdb; pdb.set_trace()
    json_data["attributes"] = response_dict
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def generate_reference_from_attributes(attribute_dict):
    attributes = {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in attribute_dict.items()}
    sentence_list = []
    # Define templates for different parts of the sentence
    from gen.vg.object_templates import OBJECT_TEMPLATES
    import random
    templates = random.sample(OBJECT_TEMPLATES, 3)
    sentence_list = [template.format(**attributes) for template in templates]
    return sentence_list

def generate_reference_dict_from_sentence(sentence, target_object_type):
    """
        Generate a reference dict from a sentence and a target object type.
        NOTE: The scan id and target id are not set in this function.
    """
    reference_dict = {
        "scan_id": None,
        "target_id": None,
        "distractor_ids": [],
        "text": sentence,
        "target": target_object_type,
        "anchors": [],
        "anchor_ids": [],
        "tokens_positive": find_matches(sentence, target_object_type)
    }
    return reference_dict

def find_matches(sentence, s):
    """
    Find all occurrences of string s in sentence.
    Returns a list of lists [[start, end], [start, end],...] of the matches.
    """
    matches = []
    start = 0
    while True:
        start = sentence.find(s, start)
        if start == -1:  
            break
        end = start + len(s)
        matches.append([start, end])
        start += 1
    return matches


if __name__ == '__main__':
    scene_id = "0_demo"
    scene_dir = os.path.join(DATA_ROOT, scene_id)
    object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
    object_text_anno_paths = os.listdir(object_text_annos_dir)
    object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if p.endswith(".json")]
    # mmengine.utils.track_parallel_progress(back_translate_text_anno_json, object_text_anno_paths, nproc=8)  
    # mmengine.utils.track_parallel_progress(extract_attribute_from_json, object_text_anno_paths, nproc=8)      
    #########################
    # Test code
    with open(object_text_anno_paths[0], "r", encoding="utf-8") as f:
        json_data = json.load(f)
    attribute_dict = json_data.get("attributes", {})
    sentences = generate_reference_from_attributes(attribute_dict)
    import pdb; pdb.set_trace()
    #########################


    scene_id = "scene0000_00"
    annotation_path = os.path.join(SPLIT_DATA_ROOT, f"{scene_id}.pkl")
    anno_embodiedscan = read_annotation_pickle(annotation_path, show_progress=False)[scene_id]

    
