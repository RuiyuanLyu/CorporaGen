import os
import json
from utils.utils_read import read_annotation_pickle
from object_text_anno import translate
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups

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
Common_Descripition = dict()

Common_Descripition['coarse_grained_category'] =[
    'Furniture',
    'Sleep/Rest appliance',
    'Decorations',
    'Store and organize supplies',
    'Study and office supplies',
    'Cleaning supplies',
    'Kitchenware and dining utensil',
    'Sanitary ware products',
    'Lighting fixtures',
    'Entertainment and leisure equipment',
    'Electronic products',
    'Personal effects'
]
Common_Descripition['color'] = [
    'Red',
    'Orange',
    'Yellow',
    'Green',
    'Blue',
    'Purple',
    'Black',
    'White',
    'Gray',
    'Brown',
    'Pink',
'Silver','Gold',
'Transparent','More than one color'

]


Common_Descripition['material'] =[
    'Wood',
    'Metal',
    'Plastic',
    'Glass',
    'Fabric/Feather',
    'Leather',
    'Ceramic',
    'Concrete',
    'Paper',
    'Stone'
]
Common_Descripition['shape'] = [
    'Rectangular',
    'Circular',
    'Cylindrical',
    'Spherical',
    'Triangular',
    'Cuboid',
    'Irregular',
    'Conical'
]
Common_Descripition['weight'] = [
    'Heavy',
    'Medium',
    'Light'
]
Common_Descripition['size'] = [
    'Large',
    'Medium',
    'Small'
]
Common_Descripition['placement'] = [
    'Standing upright', 'Piled up', 'Leaning', 'Lying flat', 'Hanging'
]

def check_object_text_anno_is_valid(json_data):
    accuracy_dict = json_data.get("accuracy_dict", {})
    if not (accuracy_dict.get("meta", "") and accuracy_dict.get("visual_info_sufficient", "")):
        return False
    return True
def check_attribute(text):
    if text is None or text=='' or text=="N/A":
        return False
    return True
    
# def generate_object_type_reference(scene_id):
def back_translate_text_anno_json(json_path):
    """
        Back-translate the modified_description field in the json file to English, and store it in the original json file.
        Returns None.
    """
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


def extract_only_one_attribute_from_text(text):
    system_prompt = "You are given a text description of an object. Your task is to identify the coarse grained category of the object. The coarse grained category means the larger class this object belongs to in our daily residential life (for example, electronics, Furniture, Sleep/Rest appliance, Decorations, Store and organize supplies, Study and office supplies, Cleaning supplies, Kitchenware and dining utensil, Sanitary ware products, Lighting fixtures, Entertainment and leisure equipment, Personal effects ...). Just give me the result."

    content_groups = get_content_groups_from_source_groups([text])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0, report_token_usage=True)
    response = messages[-1]
    assert response["role"] == "assistant"
    return response["content"]


def extract_attribute_from_text(text):
    system_prompt = "You are given a text description of an object. Your task is to identify the attributes of the object. The 11 attributes are: fine grained category, color, texture, material, weight, size, shape, placement (e.g. standing upright, piled up, leaning, lying flat, hanging), state (e.g. open, closed, locked, empty), function, and other features. Missing attributes can be left blank. Please reply in json, in the following format: {'function': 'blocks the light from entering the room', 'state': 'closed'}."
    content_groups = get_content_groups_from_source_groups([text])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0, report_token_usage=True, json_mode=True)
    response = messages[-1]
    assert response["role"] == "assistant"
    response_dict = json.loads(response["content"])
    response_dict = {k.lower().replace(" ", "_"): v for k, v in response_dict.items()}
    coarse_grained_category = extract_only_one_attribute_from_text(text)
    response_dict['coarse_grained_category'] = coarse_grained_category
    return response_dict

def find_closet_word(word,attr):
    if attr=='color':
        addition_text = "What's more,if the text means more than one color, return 'More than one color'. "
    else:
        addition_text = ''
    system_prompt = f"I will give you one text describing the {attr} of an item, please find the text with the closest meaning from the list {Common_Descripition[attr]}, if existing,just return the text,else return ''.{addition_text}I only need the text chosen from the list as the result!"

    content_groups = get_content_groups_from_source_groups([word])
    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0,
                                              report_token_usage=True)
    response = messages[-1]
    assert response["role"] == "assistant"

    result = ''
    for word in Common_Descripition[attr]:
        if word in response["content"]:
            result=word

    return result

def extract_bracket_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if not json_data.get("modified_description_en", ""):

        return


    if json_data.get("common_attribute", ""):

        return
    text = json_data['modified_description_en']
    t = ''
    for key_ in Common_Descripition.keys():
        t+=f'{key_}, '


    system_prompt = f"You are given a text description of an object. Your task is to identify the attributes of the object. The {len(Common_Descripition)} attributes are: {t[:-2]}.Give me the result as a dict(a JSON file). The keys of the dict are: {t[:-2]}.The values of them must be selected in a specified category,don't directly use the word in the text! "
    for key_ in Common_Descripition.keys():
        system_prompt += f"The value of the key '{key_}' must be chosen from {Common_Descripition[key_]} if the attribute ‘{key_}’ is not missing in the text."
    system_prompt = system_prompt[:-2]+".If one attribute is missing, Just leave its value a blank .This is an JSON file example:{'color':'Brown','material':'Wood','shape':'Rectangular','weight':'Heavy','size':'Large'}."


    content_groups = get_content_groups_from_source_groups([text])

    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0, report_token_usage=True, json_mode=True)
    response = messages[-1]
    assert response["role"] == "assistant"
    response_dict = json.loads(response["content"])
    response_dict = {k.lower().replace(" ", "_"): v for k, v in response_dict.items()}

    json_data["common_attribute"] = response_dict
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def new_extract_bracket_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if not json_data.get("modified_description_en", ""):

        return
    # if json_data.get("common_attribute", ""):
    #
    #     return

    attr_dict = json_data['attributes']

    response_dict = dict()

    for key_ in attr_dict.keys():
        if key_ in Common_Descripition.keys() and check_attribute(attr_dict[key_]):

            response_dict[key_]=find_closet_word(attr_dict[key_],key_)

    json_data["common_attribute"] = response_dict
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)



def extract_bracket_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if not json_data.get("modified_description_en", ""):

        return


    if json_data.get("common_attribute", ""):

        return
    text = json_data['modified_description_en']
    t = ''
    for key_ in Common_Descripition.keys():
        t+=f'{key_}, '


    system_prompt = f"You are given a text description of an object. Your task is to identify the attributes of the object. The {len(Common_Descripition)} attributes are: {t[:-2]}.Give me the result as a dict(a JSON file). The keys of the dict are: {t[:-2]}.The values of them must be selected in a specified category,don't directly use the word in the text! "
    for key_ in Common_Descripition.keys():
        system_prompt += f"The value of the key '{key_}' must be chosen from {Common_Descripition[key_]} if the attribute ‘{key_}’ is not missing in the text."
    system_prompt = system_prompt[:-2]+".If one attribute is missing, Just leave its value a blank .This is an JSON file example:{'color':'Brown','material':'Wood','shape':'Rectangular','weight':'Heavy','size':'Large'}."




    content_groups = get_content_groups_from_source_groups([text])

    messages, token_usage = mimic_chat_budget(content_groups, system_prompt, num_turns_expensive=0, report_token_usage=True, json_mode=True)
    response = messages[-1]
    assert response["role"] == "assistant"
    response_dict = json.loads(response["content"])
    response_dict = {k.lower().replace(" ", "_"): v for k, v in response_dict.items()}

    json_data["common_attribute"] = response_dict
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


def extract_attribute_from_json(json_path):
    """
        Extract attributes from a json file and store them in the json file.
        Returns None.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    text = json_data.get("modified_description_en", "")
    if not text:
        return
    if not json_data.get("attributes", ""):
        response_dict = extract_attribute_from_text(text)
    else:
        response_dict = json_data.get("attributes", "")
    response_dict = extract_attribute_from_text(text)


    # 加入物体类型

    object_type = json_path.split("user_shujutang_czc")[-1].split('_')[1]
    if object_type!= "object":
        response_dict["type"] = json_path.split("user_shujutang_czc")[-1].split('_')[1]
    # import pdb; pdb.set_trace()
    json_data["attributes"] = response_dict
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    scene_id = "1mp3d_0000_region10"
    scene_dir = os.path.join(DATA_ROOT, scene_id)
    object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
    object_text_anno_paths = os.listdir(object_text_annos_dir)
    object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if p.endswith(".json")]
    from tqdm import tqdm


    for json_path in tqdm(object_text_anno_paths):
        back_translate_text_anno_json(json_path)
        extract_attribute_from_json(json_path)
        new_extract_bracket_from_json(json_path)



    
