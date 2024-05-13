import os
import json
import numpy as np
import random
from utils.utils_read import read_annotation_pickle
from object_text_anno import translate
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups
from gen.vg.main import Common_Descripition

EXCLUDED_OBJECTS = ["wall", "ceiling", "floor", "pillar", "column", "roof", "ridge", "arch", "beam"]

ANCHOR_TYPE = ['bed','table']
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



def generate_reference_from_attributes(attribute_dict, scan_id=None, target_id=None):
    attributes = {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in attribute_dict.items()}
    sentence_list = []
    # Define templates for different parts of the sentence
    from gen.vg.object_templates import OBJECT_TEMPLATES
    import random
    templates = random.sample(OBJECT_TEMPLATES, 3)
    print(attributes)
    sentence_list = [template.format(**attributes) for template in templates]
    target_object_type = attributes.get("fine_grained_category", "")
    reference_list = [generate_reference_dict_from_sentence(sentence, target_object_type, scan_id, target_id) for
                      sentence in sentence_list]

    return reference_list


def generate_reference_dict_from_sentence(sentence, target_object_type, scan_id=None, target_id=None):
    """
        Generate a reference dict from a sentence and a target object type.
        NOTE: The scan id and target id are not set in this function if they are not provided.
    """
    reference_dict = {
        "scan_id": scan_id,
        "target_id": target_id,
        "distractor_ids": [],
        "text": sentence,
        "target": target_object_type,
        "anchors": [],
        "anchor_ids": [],
        "tokens_positive": find_matches(sentence, target_object_type)
    }
    return reference_dict


def find_matches(sentence, target):
    """
    Find all occurrences of string s in sentence.
    Returns a list of lists [[start, end], [start, end],...] of the matches.
    """
    matches = []

    if isinstance(target,str):
        s_list = [target]
    else:
        s_list = target
    for s in s_list:
        start = 0
        while True:
            start = sentence.find(s, start)
            if start == -1:
                break
            end = start + len(s)
            matches.append([start, end])
            start += 1
    return matches


def generate_single_reference(json_path, scan_id, region_anno_dict=None,region_id=None):
    """
        pipeline to generate reference from a single json file.
        Args:
            json_path: path to the json file.
            scan_id: the scan id of the scene. e.g. "scannet/scene0072_01"
        Returns a list of reference dicts.
    """

    base_name = os.path.basename(json_path)
    object_id = int(base_name.split("_")[0])
    object_type = base_name.split("_")[1]
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    attribute_dict = json_data.get("attributes", {})
    reference_list = generate_reference_from_attributes(attribute_dict, scan_id=scan_id, target_id=object_id)
    # for a full scene, the reference list will also be a list of reference dicts, as it is for each object in the scene.
    # for a single object, the reference list may also contain multiple reference dicts.

    # 增加了通过o-r关系去refer物体: 基于区域文本中的物体特殊性标注
    if region_anno_dict!=None:
        object_special_text = region_anno_dict[3][f'<{object_type}_{object_id}>']
        if object_special_text!='':
            object_special_text = object_special_text.replace(f'<{object_type}_{object_id}>','X')
            vg_text = f'In the {region_id.split("_")[1]} of this room, '+object_special_text+'Please find the X.'
            categorize_list = [
                [(object_id,object_type)]]
            tokens_list = ['X']
            reference_list.append(generate_multi_reference(vg_text,categorize_list,tokens_list,scan_id))



    return reference_list


def generate_multi_reference(text,categorize_list,tokens_list,scan_id,anchor_list=[],distractor_ids=[]):
    '''
        Combine some categorize dicts to refer multi items
        Args:
            text

            categorize_list: a list of categorize dict, for each dict, the key is a describing text, the value is a list of (object_id,oject_type)
    '''

    from functools import reduce
    object_set_list = [set(categorize_) for categorize_ in categorize_list]
    in_object_list = list(reduce(lambda x, y: x & y, object_set_list))


    reference_dict = {
        "scan_id": scan_id,
        "target_id": [t[0] for t in in_object_list],
        "distractor_ids": [],
        "text": text,
        "target": [t[1] for t in in_object_list],
        "anchors": [t[0] for t in anchor_list],
        "anchor_ids": [t[1] for t in anchor_list],
        "tokens_positive": find_matches(text,tokens_list)
    }

    return reference_dict

def get_object_attribute_dict(scene_id):
    '''
        return a dict describing the attributes of an object.
        such as:
            {
                object_id1,object_type1:
                {
                    attribute:{}
                    common_attribute:{}
                }
            }
    '''
    object_attribute_dict = dict()
    scene_dir = os.path.join(DATA_ROOT, scene_id)
    object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
    object_text_anno_paths = os.listdir(object_text_annos_dir)
    object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if
                              p.endswith(".json")]
    for json_path in object_text_anno_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        base_name = os.path.basename(json_path)
        object_id = int(base_name.split("_")[0])
        object_type = base_name.split("_")[1]
        attribute_dict = json_data.get("attributes", {})
        common_attribute_dict = json_data.get("common_attribute",{})
        if len(attribute_dict)==0:
            continue
        object_attribute_dict[(object_id,object_type)] = dict()
        object_attribute_dict[(object_id,object_type)]["attributes"] = attribute_dict
        object_attribute_dict[(object_id,object_type)]["common_attribute"] = common_attribute_dict

    return object_attribute_dict

def generate_common_attribute_categorize(scan_id):
    '''
        Use some common description to categorize items in a room
        Args:

            scan_id: the scan id of the scene. e.g. "scannet/scene0072_01"

    '''

    # example of Common_Descripition_object_set:
    # {
    #     'color':
    #       {
    #           'White':[(id1,type1),(id2,type2),...]
    #              .....
    #       }
    #      .......
    # }
    Common_Descripition_object_set = dict()
    for key_ in Common_Descripition:
        Common_Descripition_object_set[key_]=dict()
        for value_ in Common_Descripition[key_]:
            Common_Descripition_object_set[key_][value_] = []


    base_path = f'data/{scan_id}/corpora_object/user_shujutang_czc'
    json_paths = [base_path+'/'+path for path in os.listdir(base_path)]
    for json_path in json_paths:
        base_name = os.path.basename(json_path)
        object_id = base_name.split("_")[0]
        object_type = base_name.split("_")[1]
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            object_Common_Descripition_dict = json_data.get("common_attribute", {})
            if object_Common_Descripition_dict=={}:
                continue

            for key_ in Common_Descripition:
                value_ = object_Common_Descripition_dict.get(key_,'')
                if value_ in Common_Descripition_object_set[key_].keys():
                    Common_Descripition_object_set[key_][value_].append((object_id,object_type))
                else:
                    for attr_describe in Common_Descripition_object_set[key_].keys():
                        if attr_describe in value_:
                            Common_Descripition_object_set[key_][value_].append((object_id, object_type))
                            break



    return Common_Descripition_object_set

def get_relation_from_base(anchor_id,anchor_type,annotation_data,search_range=None):
    object_ids = annotation_data['object_ids']
    object_types = annotation_data['object_types']
    object_bboxes = annotation_data['bboxes']
    _index = list(object_ids).index(anchor_id)
    anchor_bbox = object_bboxes[_index]
    anchor_volumn = anchor_bbox[3]*anchor_bbox[4]*anchor_bbox[5]
    if search_range==None:
        search_range = [object_ids[_index] for _index in range(len(object_ids)) if object_types[_index] not in EXCLUDED_OBJECTS]


    #1. 比它高/低的物体
    higher_list = []
    lower_list = []

    for _index in range(len(object_bboxes)):

        object_bbox = object_bboxes[_index]
        object_id = object_ids[_index]
        object_type = object_types[_index]

        if object_id not in search_range or object_id==anchor_id:
            continue


        # 如何定义比它高?(什么阈值合理)
        if object_bbox[2]>anchor_bbox[2]+anchor_bbox[5]/2:
            higher_list.append((object_id,object_type))
        # 如何定义比它低?(什么阈值合理)
        if object_bbox[2] < anchor_bbox[2] - anchor_bbox[5] / 2:
            lower_list.append((object_id, object_type))

    #2. 比它大/小的物体
    larger_list = []
    smaller_list = []

    for _index in range(len(object_bboxes)):
        object_bbox = object_bboxes[_index]
        object_id = object_ids[_index]
        object_type = object_types[_index]
        object_volumn = object_bbox[3]*object_bbox[4]*object_bbox[5]

        if object_id not in search_range or object_id==anchor_id:
            continue

        threshold = 0.2

        # 如何定义比它大?(什么阈值合理)

        if object_volumn > (1+threshold)*anchor_volumn:
            larger_list.append((object_id, object_type))
        # 如何定义比它小?(什么阈值合理)
        if object_volumn < (1-threshold)*anchor_volumn:
            smaller_list.append((object_id, object_type))

    #3. 离它最近/最远的物体
    farest_object = None
    closet_object = None
    min_distance = np.inf
    max_distance = 0

    for _index in range(len(object_bboxes)):
        object_bbox = object_bboxes[_index]
        object_id = object_ids[_index]
        object_type = object_types[_index]
        if object_id not in search_range or object_id==anchor_id:
            continue


        distance = np.sqrt((object_bbox[0]-anchor_bbox[0])**2+(object_bbox[1]-anchor_bbox[1])**2+(object_bbox[2]-anchor_bbox[2])**2)


        if distance>max_distance:
            farest_object = (object_id,object_type)
            max_distance = distance
        if distance <min_distance:
            closet_object = (object_id, object_type)
            min_distance = distance
    anchor_dict = dict()
    anchor_dict['higher'] = higher_list
    anchor_dict['lower'] = lower_list
    anchor_dict['larger'] = larger_list
    anchor_dict['smaller'] = smaller_list

    return anchor_dict

def generate_function_relation_reference(region_anno_dict,region_id,scene_id):
    ##todo 合并同类项

    reference_list = []

    function_relations = region_anno_dict[1]
    anchor_function_relations = dict()

    large_class_dict = region_anno_dict[4]

    for object_tuple in function_relations.keys():

        if (ANCHOR_TYPE==None or object_tuple[0].split('_')[0][1:] in ANCHOR_TYPE):
            if object_tuple[0] not in anchor_function_relations.keys():
                anchor_function_relations[object_tuple[0]] = [(object_tuple[1],function_relations[object_tuple])]
            else:
                anchor_function_relations[object_tuple[0]].append((object_tuple[1], function_relations[object_tuple]))
        if (ANCHOR_TYPE == None or object_tuple[1].split('_')[0][1:] in ANCHOR_TYPE):
            if object_tuple[1] not in anchor_function_relations.keys():
                anchor_function_relations[object_tuple[1]] = [(object_tuple[0],function_relations[object_tuple])]
            else:
                anchor_function_relations[object_tuple[1]].append((object_tuple[0], function_relations[object_tuple]))

    for object_list in large_class_dict.keys():
        text = f'This is a descripition of {len(object_list)} items in the {region_id.split("_")[1]}:{large_class_dict[object_list]}Please find them. '
        categorize_list = [[(object_name.split('_')[0][1:],object_name.split('_')[1][:-1]) for object_name in object_list]]
        tokens_list = [f'{len(object_list)} items in the {region_id.split("_")[1]}']
        reference_list.append(generate_multi_reference(text, categorize_list, tokens_list, scene_id))



    for anchor_name in anchor_function_relations:
        anchor_type,anchor_id = anchor_name.split('_')[0][1:],anchor_name.split('_')[1][:-1]
        for target_name,text in anchor_function_relations[anchor_name]:
            target_type,target_id = target_name.split('_')[0][1:], target_name.split('_')[1][:-1]
            text = text.replace(target_name,'X')
            text = text.replace(anchor_name,anchor_type)
            text+='Please find the X.'
            categorize_list=[[(target_id,target_type)]]
            tokens_list = [anchor_type,'X']
            reference_list.append(generate_multi_reference(text, categorize_list, tokens_list, scene_id))

    return reference_list



def generate_space_relation_reference(sr3d_dict,annotation_data,scene_id):
    example_input = [
        {
            "scan_id": "scene0119_00",
            "target_id": 5,
            "distractor_ids": [
                4
            ],
            "utterance": "select [the cabinet 5] that is supporting [the sink 3]",
            "stimulus_id": "scene0119_00-cabinet-2-5-4",
            "coarse_reference_type": "support",
            "reference_type": "supporting",
            "instance_type": "cabinet",
            "anchors_types": [
                "sink"
            ],
            "anchor_ids": [
                3
            ]
        }]
    reference_list = []
    raw_space_list = sr3d_dict[scene_id]
    scene_data = annotation_data[scene_id]

    def process_text(text):

        _index = 0
        output_list = []
        while '[' in text[_index:] and ']' in text[_index:]:
            a = text[_index:].index('[')
            b = text[_index:].index(']')
            output_list.append(text[_index + a:_index + b + 1])
            if '[' in text[_index + a+1:_index + b ] or ']' in text[_index + a+1:_index + b ]:
                return ''
            _index = _index + b + 1

        for object_name in output_list:

            text = text.replace(object_name,'the '+object_name.split(' ')[1])
        return text
    for raw_space_item in raw_space_list:
        vg_text = process_text(raw_space_item["utterance"])
        if len(vg_text)==0:
            continue
        reference_dict = {
            "scan_id": scene_id,
            "target_id": [raw_space_item["target_id"]],
            "distractor_ids": [],
            "text": vg_text,
            "target": [raw_space_item["instance_type"]],
            "anchors": raw_space_item["anchors_types"],
            "anchor_ids": raw_space_item["anchor_ids"],
            "tokens_positive": find_matches(vg_text, [raw_space_item["instance_type"]]+raw_space_item["anchors_types"])
        }


        reference_list.append(reference_dict)
    return reference_list

def choose_items(annotation_data,sample_num = 3):

    # 在这里选取一个合适的物体开展基于基础标注的相对位置关系

    # 现在暂时先使用:在search_range中随机sample
    object_ids = annotation_data['object_ids']
    object_types = annotation_data['object_types']
    possible_index = []
    for index_ in range(len(object_types)):
        if ANCHOR_TYPE!=None and object_types[index_] not in ANCHOR_TYPE:
            continue
        possible_index.append(index_)
    choose_index = random.sample(possible_index,min(3,len(possible_index)))

    return [(object_ids[_index],object_types[_index]) for _index in choose_index]


def generate_base_space_reference(annotation_data,object_attribute_dict,scene_id):
    reference_list = []

    anchor_list = choose_items(annotation_data)

    for anchor_id,anchor_type in anchor_list:
        anchor_dict = get_relation_from_base(anchor_id,anchor_type,annotation_data)
        attribute_dict = object_attribute_dict[(anchor_id,anchor_type)]['attributes']
        # todo: add the text to better refer anchor
        describe_text = 'There is a '
        if "color" in attribute_dict.keys():
            describe_text+=f"{attribute_dict['color']} color,"
        if "texture" in attribute_dict.keys():
            describe_text += f"{attribute_dict['texture']} texture,"
        if "material" in attribute_dict.keys():
            describe_text += f"{attribute_dict['material']} material,"
        describe_text = describe_text[:-1]+' '+anchor_type
        if "shape" in attribute_dict.keys():
            describe_text += f" with a {attribute_dict['shape']} shape"


        for compare_word in anchor_dict.keys():



            vg_text = f'. Find all the items {compare_word} than the {anchor_type} in the room. '

            categorize_list = [anchor_dict[compare_word]]
            tokens_list = [f'the items {compare_word}',anchor_type]
            anchor_list = [(anchor_id,anchor_type)]



            reference_list.append(generate_multi_reference(describe_text+vg_text, categorize_list, tokens_list, scene_id,anchor_list=anchor_list))



    return reference_list

def generate_Common_attribute_reference(Common_Descripition,scan_id):
    reference_list = [ ]

    # 单种归类的

    for attribute_name in Common_Descripition.keys():
        for key_ in Common_Descripition[attribute_name].keys():
            if len(Common_Descripition[attribute_name][key_])>0:
                text = f'Find all the items with {key_} {attribute_name} in the room'
                categorize_list = [
                    Common_Descripition[attribute_name][key_],
                ]
                token_list = [f'items with {key_} {attribute_name}']
                reference_list.append(generate_multi_reference(text, categorize_list, token_list, scan_id))

    # 多种归类的

    color_key_list = []
    material_key_list = []
    shape_key_list = []
    for key_ in Common_Descripition['color'].keys():
        if len(Common_Descripition['color'][key_])>0:
            color_key_list.append(key_)
    for key_ in Common_Descripition['shape'].keys():
        if len(Common_Descripition['shape'][key_])>0:
            shape_key_list.append(key_)
    for key_ in Common_Descripition['material'].keys():
        if len(Common_Descripition['material'][key_])>0:
            material_key_list.append(key_)

    import itertools
    color_shape_pair_list = list(itertools.product(color_key_list,shape_key_list))
    for color_key,shape_key in color_shape_pair_list:
        combine_list = list(set(Common_Descripition['color'][color_key]) & set(Common_Descripition['shape'][shape_key]))
        if len(combine_list)>0:

            text = f'Find all the {color_key} and {shape_key} items in the room. '
            categorize_list = [
                Common_Descripition['color'][color_key],
                Common_Descripition['shape'][shape_key]
            ]
            token_list = [f'{color_key} and {shape_key} items']
            reference_list.append(generate_multi_reference(text,categorize_list,token_list,scan_id))
    color_material_pair_list = list(itertools.product(color_key_list, material_key_list))
    for color_key, material_key in color_material_pair_list:
        combine_list = list(set(Common_Descripition['color'][color_key]) & set(Common_Descripition['material'][material_key]))
        if len(combine_list) > 0:
            text = f'Find all the {color_key} and {material_key} items in the room'
            categorize_list = [
                Common_Descripition['color'][color_key],
                Common_Descripition['material'][material_key]
            ]
            token_list = [f'{color_key} and {material_key} items']
            reference_list.append(generate_multi_reference(text, categorize_list, token_list, scan_id))


    return reference_list



if __name__ == '__main__':



    scene_id = "scene0000_00"
    scene_dir = os.path.join(DATA_ROOT, scene_id)
    object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
    object_text_anno_paths = os.listdir(object_text_annos_dir)
    object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if
                              p.endswith(".json")]
    region_text_annos_dir = os.path.join(scene_dir, "region_views")
    region_text_anno_dict = dict()
    annotation_data = read_annotation_pickle('splitted_infos/scene0000_00.pkl')
    for region_id in os.listdir(region_text_annos_dir):
        if region_id[-4:]=='.png':
            continue
        region_text_anno_dict[region_id] = np.load(os.path.join(region_text_annos_dir,region_id,'struction.npy'),allow_pickle=True)

    object_attribute_dict = get_object_attribute_dict(scene_id)
    # with open("all_sr3d_relations.json", 'r', encoding='UTF-8') as f:
    #     sr3d_dict = json.load(f)
    sr3d_dict = {scene_id:np.load('aaaaaa.npy',allow_pickle=True)}


    #example 1: 基于位置功能联系/大类标注(from 区域文本标注)



    ex_function_dict ={('<stool_29>', '<table_3>'): 'The <stool_29> is used for sitting at the <table_3>.', ('<stool_30>', '<table_3>'): 'The <stool_30> is used for sitting at the <table_3>.', ('<stool_31>', '<table_3>'): 'The <stool_31> is used for sitting at the <table_3>.', ('<stool_32>', '<table_3>'): 'The <stool_32> is used for sitting at the <table_3>.', ('<cup_154>', '<table_3>'): 'The <cup_154> is placed on <table_3> to hold a beverage which can be enjoyed while sitting on the stools.', ('<object_155>', '<table_3>'): 'The <object_155> is placed on <table_3> for functional or decorative purposes.'}
    ex_special_dict = {'<table_3>': 'The <table_3> is the centerpiece of the dining area, serving as the main surface for meals and gatherings.'}
    ex_large_class_dict = {('<table_3>', '<stool_29>', '<stool_30>', '<stool_31>', '<stool_32>'): 'Belong to the class of dining furniture, together they form a dining set used for eating and socializing.', ('<cup_154>', '<table_3>'): 'Belong to the class of tableware.'}
    ex_region_anno_dict = [{},ex_function_dict,{},ex_special_dict,ex_large_class_dict]

    ex_output1 = generate_space_relation_reference(sr3d_dict, annotation_data,scene_id)


    ex_output2 = generate_function_relation_reference(ex_region_anno_dict, '1_dinning region', scene_id)
    #print(ex_output2)

    #example 2: 基于初级标注数据bboxes

    ex_output3 = generate_base_space_reference(annotation_data[scene_id],object_attribute_dict,scene_id)

    #example 3: 基于文本产生的大类(from 物体文本标注)

    Common_Descripition_object_set = generate_common_attribute_categorize(scene_id)

    ex_output4 = generate_Common_attribute_reference(Common_Descripition_object_set,scene_id)

    #print(show_dict)

    #example 4: 基于物体文本标注

    ex_output5 = generate_single_reference(object_text_anno_paths[0],scene_id,ex_region_anno_dict,'1_dinning region')

    ex_output = ex_output1+ex_output2+ex_output3+ex_output4+ex_output5
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    with open("VG.json", "w") as f:

        json.dump(ex_output, f,cls=NpEncoder)














