import os
import json
import numpy as np
import random
from utils.utils_read import read_annotation_pickle
from object_text_anno import translate
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups
from gen.extract import Common_Descripition

EXCLUDED_OBJECTS = ["wall", "ceiling", "floor", "pillar", "column", "roof", "ridge", "arch", "beam"]

ANCHOR_TYPE = None
DATA_ROOT = "data/"
SPLIT_DATA_ROOT = "splitted_infos/"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
All_VG_SUB_CLASS=[
    'VG_Relation_Space','VG_Relation_Function0','VG_Relation_Function1','VG_Relation_Base',
    'VG_Attribute_classify1','VG_Attribute_classify2','VG_Attribute_Single'
]

REFERING_TYPES = [
    "fine_grained_category",
    "coarse_grained_category",
    "color",
    "texture",
    "material",
    "weight",
    "size",
    "shape",
    "placement",
    "state",
    "function",
    "other_features"
]

def check_attribute(text):
    '''
        check to ensure the output of GPT is valid.
    '''
    if text is None or text=='' or text=="N/A" or not isinstance(text,str):
        return False
    return True
def filter_repeat_utterance(sr3d_list):
    '''
        for repeat space relation, only choose one of them randomly.
    '''
    from copy import deepcopy
    store_flag = None
    store_list = []
    output_list = []
    for l in sr3d_list:
        if store_flag is not None:
            if store_flag['target_id'] == l['target_id'] and store_flag['distractor_ids'] == l['distractor_ids'] and \
                    store_flag['anchor_ids'] == l['anchor_ids']:
                store_list.append(l)
            else:
                output_list.append(random.sample(store_list, 1)[0])
                store_flag = deepcopy(l)
        else:
            store_flag = deepcopy(l)
    return output_list

def filter_excluded_type(annotation):
    fix_anno = {}
    fix_anno['object_ids'] = []
    fix_anno['object_types'] = []
    fix_anno['bboxes'] = []
    for _index in range(len(annotation['object_ids'])):
        if annotation['object_types'][_index] not in EXCLUDED_OBJECTS:
            fix_anno['object_ids'].append(annotation['object_ids'][_index])
            fix_anno['object_types'].append(annotation['object_types'][_index])
            fix_anno['bboxes'].append(annotation['bboxes'][_index])
    return fix_anno

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

def generate_reference_from_attributes(object_type,attribute_dict, scan_id=None, target_id=None):
    '''
        generate a list of reference dict of a single object from attributes.
    '''
    attributes = {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in attribute_dict.items()}
    sentence_list = []
    # Define templates for different parts of the sentence
    from gen.vg.object_templates import OBJECT_TEMPLATES
    import random
    template_choice = []
    for sentence in OBJECT_TEMPLATES:
        flag=True
        for attribute_name in REFERING_TYPES:
            if attribute_name in sentence and (attribute_name not in attributes.keys() or not check_attribute(attributes[attribute_name])):
                flag=False
        if flag:
            template_choice.append(sentence)


    templates = random.sample(template_choice, min(3,len(template_choice)))
    sentence_list = [template.format(**attributes) for template in templates]

    reference_list = []
    for sentence in sentence_list:
        reference_dict = {
            "sub_class": 'VG_Attribute_Single',
            "scan_id": scan_id,
            "target_id": target_id,
            "distractor_ids": [],
            "text": sentence,
            "target": object_type,
            "anchors": [],
            "anchor_ids": [],
            "tokens_positive": find_matches(sentence, object_type)
        }
        reference_list.append(reference_dict)

    return reference_list




def generate_base_attribute_reference(json_path, scan_id, region_anno_dict=None,region_id=None):
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
    reference_list = generate_reference_from_attributes(object_type,attribute_dict, scan_id=scan_id, target_id=object_id)

    # 增加了通过o-r关系去refer物体: 基于区域文本中的物体特殊性标注
    if region_anno_dict is not None:
        object_special_text = region_anno_dict[3][f'<{object_type}_{object_id}>']
        if object_special_text!='':
            object_special_text = object_special_text.replace(f'<{object_type}_{object_id}>','X')
            vg_text = f'In the {region_id.split("_")[1]} of this room, '+object_special_text+'Please find the X.'
            categorize_list = [
                [(object_id,object_type)]]
            tokens_list = ['X']
            #print(generate_multi_reference(vg_text,categorize_list,tokens_list,scan_id,sub_class='VG_Attribute_Single'))
            reference_list.append(generate_multi_reference(vg_text,categorize_list,tokens_list,scan_id,sub_class='VG_Attribute_Single'))



    return reference_list


def generate_multi_reference(text,categorize_list,tokens_list,scan_id,anchor_list=[],distractor_ids=[],sub_class=''):
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
        "sub_class": sub_class,
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
        if object_type in EXCLUDED_OBJECTS:
            continue
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


    # 涉及大类的生成

    for object_list in large_class_dict.keys():
        text = f'This is a description of {len(object_list)} items in the {region_id.split("_")[1]}:{large_class_dict[object_list]}Please find them. '
        categorize_list = [[(int(object_name.split('_')[1][:-1]),object_name.split('_')[0][1:]) for object_name in object_list]]
        tokens_list = [f'{len(object_list)} items in the {region_id.split("_")[1]}']
        reference_list.append(generate_multi_reference(text, categorize_list, tokens_list, scene_id,sub_class="VG_Relation_Function0"))


    # 涉及功能联系的生成

    for anchor_name in anchor_function_relations:
        anchor_type,anchor_id = anchor_name.split('_')[0][1:],int(anchor_name.split('_')[1][:-1])

        anchor_function_text_dict = {}
        for target_name, text in anchor_function_relations[anchor_name]:
            target_type, target_id = target_name.split('_')[0][1:], int(target_name.split('_')[1][:-1])
            text = text.replace(target_name, 'X')
            text = text.replace(anchor_name, anchor_type)
            text = text[:-1]+ f' in the {region_id.split("_")[1]}. Please find the X.'

            if text not in anchor_function_text_dict:
                anchor_function_text_dict[text] = [(target_id, target_type)]
            else:
                anchor_function_text_dict[text].append((target_id, target_type))
        for text in anchor_function_text_dict.keys():
            categorize_list = [anchor_function_text_dict[text]]
            tokens_list = [anchor_type, 'X']
            reference_list.append(generate_multi_reference(text, categorize_list, tokens_list, scene_id,sub_class="VG_Relation_Function1"))

    return reference_list



def generate_space_relation_reference(sr3d_dict,annotation_data,scene_id):

    reference_list = []
    raw_space_list = filter_repeat_utterance(sr3d_dict[scene_id])

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

        all_types = raw_space_item["anchors_types"]+[raw_space_item["instance_type"]]
        skip_flag = False
        for o_type in all_types:
            if o_type in EXCLUDED_OBJECTS:
                skip_flag = True
        if skip_flag:
            continue

        reference_dict = {
            "sub_class": "VG_Relation_Space",
            "scan_id": scene_id,
            "target_id": [raw_space_item["target_id"]],
            "distractor_ids": raw_space_item["distractor_ids"],
            "text": vg_text,
            "target": [raw_space_item["instance_type"]],
            "anchors": raw_space_item["anchors_types"],
            "anchor_ids": raw_space_item["anchor_ids"],
            "tokens_positive": find_matches(vg_text, [raw_space_item["instance_type"]]+raw_space_item["anchors_types"])
        }
        reference_list.append(reference_dict)
    return reference_list

def choose_items(annotation_data,object_attribute_dict,sample_num = 10):

    # 在这里选取一个合适的物体开展

    # 现在暂时先使用:在search_range中随机sample
    object_ids = annotation_data['object_ids']
    object_types = annotation_data['object_types']
    possible_index = []
    for index_ in range(len(object_types)):
        if ANCHOR_TYPE != None and object_types[index_] not in ANCHOR_TYPE:
            continue
        if (object_ids[index_],object_types[index_]) not in object_attribute_dict.keys():
            continue
        possible_index.append(index_)
    choose_index = random.sample(possible_index,min(sample_num,len(possible_index)))

    return [(object_ids[_index],object_types[_index]) for _index in choose_index]


def generate_base_relation_reference(annotation_data,Common_Descripition_object_set,object_attribute_dict,scene_id):
    reference_list = []

    annotation_data = annotation_data[scene_id]

    anchor_list = choose_items(annotation_data,object_attribute_dict)

    for anchor_id,anchor_type in anchor_list:
        anchor_dict = get_relation_from_base(anchor_id,anchor_type,annotation_data)
        attribute_dict = object_attribute_dict[(anchor_id,anchor_type)]['attributes']
        # todo: add the text to better refer anchor

        describe_text = 'There is a '
        if "color" in attribute_dict.keys() and check_attribute(attribute_dict['color']):
            describe_text+=f"{attribute_dict['color']} color,"
        if "texture" in attribute_dict.keys() and check_attribute(attribute_dict["texture"]):
            describe_text += f"{attribute_dict['texture']} texture,"
        if "material" in attribute_dict.keys() and check_attribute(attribute_dict["material"]):
            describe_text += f"{attribute_dict['material']} material,"
        describe_text = describe_text[:-1]+' '+anchor_type
        if "shape" in attribute_dict.keys() and check_attribute(attribute_dict["shape"]):
            describe_text += f" with a {attribute_dict['shape']} shape"

        common_attribute_dict = object_attribute_dict[(anchor_id, anchor_type)]['common_attribute']

        for key_ in common_attribute_dict.keys():
            if common_attribute_dict[key_]!='' and key_!='size':
                object_id_types_list = Common_Descripition_object_set[key_][common_attribute_dict[key_]]
                vg_text = f'. Find all the items are the same as this {anchor_type} in {key_} in the room. '
                categorize_list = [object_id_types_list]
                tokens_list = [f'the items', anchor_type]
                anchor_list = [(anchor_id, anchor_type)]
                reference_list.append(
                    generate_multi_reference(describe_text + vg_text, categorize_list, tokens_list, scene_id,
                                             anchor_list=anchor_list, sub_class='VG_Relation_Base'))

        for compare_word in anchor_dict.keys():

            vg_text = f'. Find all the items {compare_word} than the {anchor_type} in the room. '

            categorize_list = [anchor_dict[compare_word]]
            tokens_list = [f'the items {compare_word}',anchor_type]
            anchor_list = [(anchor_id,anchor_type)]
            reference_list.append(generate_multi_reference(describe_text+vg_text, categorize_list, tokens_list, scene_id,anchor_list=anchor_list,sub_class='VG_Relation_Base'))



    return reference_list

def generate_common_attribute_reference(annotation_data,Common_Descripition,scan_id):
    reference_list = [ ]

    object_ids = annotation_data[scan_id]['object_ids']
    object_types = annotation_data[scan_id]['object_types']


    # 单种归类的

    # commmon基础类别组
    for attribute_name in Common_Descripition.keys():
        for key_ in Common_Descripition[attribute_name].keys():
            if len(Common_Descripition[attribute_name][key_])>0:
                if attribute_name== "size":
                    text = f'Find all the items which are {key_} size for objects of the same kind in the room.'
                else:
                    text = f'Find all the items with {key_} {attribute_name} in the room.'
                categorize_list = [
                    Common_Descripition[attribute_name][key_],
                ]
                token_list = [f'items with {key_} {attribute_name}']
                reference_list.append(generate_multi_reference(text, categorize_list, token_list, scan_id,sub_class='VG_Attribute_classify1'))
    # type的查询
    object_exist_dict = {}
    type_require_num = 5

    for index_ in range(len(object_ids)):
        if object_types[index_]== 'object':
            continue
        if object_types[index_] in object_exist_dict.keys():
            object_exist_dict[object_types[index_]].append(object_ids[index_])
        else:
            object_exist_dict[object_types[index_]] = [object_ids[index_]]

    object_type_ids_list = []
    for key_ in object_exist_dict.keys():
        object_type_ids_list.append((key_, object_exist_dict[key_]))
    object_type_ids = random.sample(object_type_ids_list, min(len(object_type_ids_list), type_require_num))

    for object_type,_ids in object_type_ids:
        text = f'Find all the {object_type}s in the room.'
        categorize_list = [
           [ (_id,object_type) for _id in _ids]
        ]
        token_list = [f'{object_type}s']
        reference_list.append(
            generate_multi_reference(text, categorize_list, token_list, scan_id, sub_class='VG_Attribute_classify1'))



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
            reference_list.append(generate_multi_reference(text,categorize_list,token_list,scan_id,sub_class='VG_Attribute_classify2'))
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
            reference_list.append(generate_multi_reference(text, categorize_list, token_list, scan_id,sub_class='VG_Attribute_classify2'))


    return reference_list



def get_attribute_reference(annotation_data, Common_Descripition_object_set,object_text_anno_paths, region_anno_dict,scene_id):
    '''
        所有有关instance本身的板块属性
    '''
    reference_list = generate_common_attribute_reference(annotation_data, Common_Descripition_object_set,scene_id)
    for object_text_anno_path in object_text_anno_paths:
        base_name = os.path.basename(object_text_anno_path)
        object_id = int(base_name.split("_")[0])
        object_type = base_name.split("_")[1]
        region_id = None
        for _id in region_anno_dict:
            if object_id in region_anno_dict[_id]["objects"]:
                region_id = _id
        if region_id is not None:
            reference_list += generate_base_attribute_reference(object_text_anno_path, scene_id,
                                                                     region_anno_dict[region_id]["annotation"], region_id)
        else:
            reference_list += generate_base_attribute_reference(object_text_anno_path, scene_id,
                                                               None, '')
    return reference_list

def get_relation_reference(annotation_data, Common_Descripition_object_set,sr3d_dict,object_attribute_dict, region_anno_dict,scene_id):
    '''
        所有有关instance与其他联系的板块属性
    '''
    reference_list = generate_space_relation_reference(sr3d_dict, annotation_data, scene_id)

    for _id in region_anno_dict:

        reference_list += generate_function_relation_reference(region_anno_dict[_id]["annotation"], _id,
                                                                       scene_id)

    reference_list += generate_base_relation_reference(annotation_data, Common_Descripition_object_set,
                                                               object_attribute_dict, scene_id)
    return reference_list

if __name__ == '__main__':

    scene_id = "1mp3d_0000_region10"
    scene_dir = os.path.join(DATA_ROOT, scene_id)
    object_text_annos_dir = os.path.join(scene_dir, "corpora_object", "user_shujutang_czc")
    object_text_anno_paths = os.listdir(object_text_annos_dir)
    object_text_anno_paths = [os.path.join(object_text_annos_dir, p) for p in object_text_anno_paths if
                              p.endswith(".json")]
    region_text_annos_dir = os.path.join(scene_dir, "region_views")
    region_anno_dict = dict()
    annotation_data = read_annotation_pickle(f'splitted_infos/{scene_id}.pkl')

    annotation_data[scene_id] = filter_excluded_type(annotation_data[scene_id])
    for region_id in os.listdir(region_text_annos_dir):
        if region_id[-4:]=='.png':
            continue
        if os.path.exists(os.path.join(region_text_annos_dir,region_id,'struction_shujutang_czc.npy')):

            region_anno_dict[region_id] = {"annotation":np.load(os.path.join(region_text_annos_dir,region_id,'struction_shujutang_czc.npy'),allow_pickle=True),
                                                "objects":np.load(os.path.join(region_text_annos_dir,region_id,'object_filter.npy'))}

    object_attribute_dict = get_object_attribute_dict(scene_id)
    with open("all_sr3d_relations.json", 'r', encoding='UTF-8') as f:
        sr3d_dict = json.load(f)

    # sr3d_dict = {scene_id:np.load('aaaaaa.npy',allow_pickle=True)}
    # print(sr3d_dict)

    Common_Descripition_object_set = generate_common_attribute_categorize(scene_id)



    # 基于instance自身属性的
    attribute_reference = get_attribute_reference(annotation_data, Common_Descripition_object_set, object_text_anno_paths, region_anno_dict,
                            scene_id)


    # 基于instance与其它联系的
    relation_reference = get_relation_reference(annotation_data, Common_Descripition_object_set, sr3d_dict, object_attribute_dict,
                           region_anno_dict, scene_id)

    result_dict = attribute_reference+relation_reference
    count_dict = dict()
    for name_ in All_VG_SUB_CLASS:
        count_dict[name_] = 0

    for dict_ in result_dict:
        dict_['ID'] = dict_['sub_class'] + '_' + str(count_dict[dict_['sub_class']])
        count_dict[dict_['sub_class']] += 1

    with open(f"{scene_id}_VG.json", "w") as f:

        json.dump(result_dict, f, cls=NpEncoder)














