import os
import json
import numpy as np
import random
from utils.utils_read import read_annotation_pickle
from object_text_anno import translate
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups
from gen.vg.generate import generate_common_attribute_categorize

attribute_list_total = ["fine grained category",
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
                                "other features"] # other features
QA_generate_num_for_one_scene ={
    'Presence and quantity':40
}

EXCLUDED_OBJECTS = ["wall", "ceiling", "floor", "pillar", "column", "roof", "ridge", "arch", "beam"]

ANCHOR_TYPE = ['bed','table']
DATA_ROOT = "data/"
SPLIT_DATA_ROOT = "splitted_infos/"
example_dict = {"answers": ["dark brown", "brown"], "object_ids": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "object_names": ["chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair", "chair"], "question": "What color is the chair in the kitchen?", "question_id": "val-scene0011-0", "scene_id": "scene0011_00"}
def generate_qa_dict(question,answers,object_list,scan_id,input_bboxes_id = None, output_bboxes_id = None):
    qa_dict = {
        "answers":answers,
        "object_ids": [t[0] for t in object_list],
        "object_names": [t[1] for t in object_list],
        "question":question,
        "scan_id":scan_id,
        "input_bboxes_id":input_bboxes_id,
        "output_bboxes_id":output_bboxes_id
    }

    return qa_dict


def generate_single_item_QA(json_path,scene_id):
    pass

def generate_attribute_QA(scene_id,object_attribute_dict):
    annotation_data = read_annotation_pickle(f'splitted_infos/{scene_id}.pkl')[scene_id]
    object_ids = annotation_data['object_ids']
    object_types = annotation_data['object_types']

    QA_list = []


    # 首先是存在和数量，所有可能的物体类别现在先用color_map里的


    with open("color_map.txt", "r") as f:
        txt_content = f.read()
    item_colors = {}
    for line in txt_content.strip().split("\n"):
        item, color = line.split("[")
        item = item.strip()
        color = color.strip("] ").split(",")
        color = tuple(int(c) for c in color)
        item_colors[item] = color
    all_types = list(item_colors.keys())



    object_exist_dict = {}

    for index_ in range(len(object_ids)):
        if object_types[index_] in object_exist_dict.keys():
            object_exist_dict[object_types[index_]].append(object_ids[index_])
        else:
            object_exist_dict[object_types[index_]]=[object_ids[index_]]

    object_type_ids_list = []
    for key_ in object_exist_dict.keys():
        object_type_ids_list.append((key_,object_exist_dict[key_]))
    object_type_ids = random.sample(object_type_ids_list,min(len(object_type_ids_list),QA_generate_num_for_one_scene['Presence and quantity']//2))

    pos_types = list(object_exist_dict)
    neg_types = [type_ for type_ in all_types if type_ not in pos_types]
    pos_num = min(len(pos_types),QA_generate_num_for_one_scene['Presence and quantity']//4)
    neg_num = QA_generate_num_for_one_scene['Presence and quantity']//4-pos_num
    pos_choice = random.sample(pos_types,pos_num)
    neg_choice = random.sample(neg_types,neg_num)

    # 存在性(pos+neg)
    for object_type in pos_choice:
        question = f'Is there a {object_type} in the room?'
        answers = ['Yes.','Yes, there is.']
        object_list = [(object_id, object_type) for object_id in object_exist_dict[object_type]]
        scan_id = scene_id
        output_bboxes_id = [object_id for object_id in object_exist_dict[object_type]]
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id,None,output_bboxes_id))
    for object_type in neg_choice:
        question = f'Is there a {object_type} in the room?'
        answers = ['No.',"No, there isn't."]
        object_list = []
        scan_id = scene_id
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id))

    # 数量
    for object_type,object_ids in object_type_ids:
        question = f'How many {object_type}s are there in the room?'
        answers = [f'{len(object_ids)}']
        object_list = [(object_id, object_type) for object_id in object_ids]
        scan_id = scene_id
        output_bboxes_id = [object_id for object_id in object_exist_dict[object_type]]
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id,None,output_bboxes_id))

    # 物体属性QA

    # 此处的文本描述还要改进,这里为了方便用比较生硬的语言
    for object_id,object_type in object_attribute_dict.keys():
        attribute_dict = object_attribute_dict[(object_id,object_type)]['attributes']
        common_attribute_dict = object_attribute_dict[(object_id,object_type)]['common_attribute']

        Question_and_Answers_set = list()


        for attribute_name in attribute_list_total:

            if attribute_name in attribute_dict.keys() and attribute_dict[attribute_name]!='':
                Qusetion = "Its {} is {}. ".format(attribute_name,attribute_dict[attribute_name])
                Answers = [attribute_dict[attribute_name]]
                if attribute_name in common_attribute_dict and common_attribute_dict[attribute_name]!='':
                    Answers.append(common_attribute_dict[attribute_name])
                Question_and_Answers_set.append((attribute_name,Qusetion,Answers))

        # 第一类属性QA: 给bboxes问属性
        for attribute_name,Qusetion,Answers in Question_and_Answers_set:
            question = f'What is {attribute_name} of this object?'
            answers = Answers
            object_list = [(object_id, object_type)]
            scan_id = scene_id
            input_bboxes_id = [object_id]
            QA_list.append(generate_qa_dict(question,answers,object_list,scan_id,input_bboxes_id))

        # 第二类属性QA：给一定属性描述，给出bboxes和另外的描述

        total_num = 4 # 控制总数
        require_indices = random.sample(list(range(len(Question_and_Answers_set))),total_num)
        for _index1 in require_indices:
            question = f'I will give you a description of an item in the room:'
            for _index2 in range(len(Question_and_Answers_set)):
                if _index1==_index2:
                    continue
                question+= Question_and_Answers_set[_index2][1]
            question += f'What is {attribute_name} of this object?'
            answers = Question_and_Answers_set[_index1][2]
            object_list = [(object_id, object_type)]
            scan_id = scene_id
            output_bboxes_id = [object_id]
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, None,output_bboxes_id))


    return QA_list

def generate_relation_QA(region_anno_dict,common_attribute_categorize,object_attribute_dict,sr3d_dict,scene_id,region_id):

    QA_list = []

    function_relations = region_anno_dict[1]

    large_class_dict = region_anno_dict[4]





    # 物体间位置的QA, 要求圈出bboxes并且回答有关属性的问题
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
    raw_space_list = sr3d_dict[scene_id]


    def process_text(text):

        _index = 0
        output_list = []
        while '[' in text[_index:] and ']' in text[_index:]:
            a = text[_index:].index('[')
            b = text[_index:].index(']')
            output_list.append(text[_index + a:_index + b + 1])
            if '[' in text[_index + a + 1:_index + b] or ']' in text[_index + a + 1:_index + b]:
                return ''
            _index = _index + b + 1

        for object_name in output_list:
            text = text.replace(object_name, 'the ' + object_name.split(' ')[1])
        return text

    for raw_space_item in raw_space_list:
        vg_text = process_text(raw_space_item["utterance"])
        if len(vg_text) == 0:
            continue
        target_id = raw_space_item["target_id"]
        target_type = raw_space_item["instance_type"]
        if (target_id, target_type) not in object_attribute_dict.keys():
            continue

        attribute_dict = object_attribute_dict[(target_id, target_type)]['attributes']
        common_attribute_dict = object_attribute_dict[(target_id, target_type)]['common_attribute']

        Question_and_Answers_set = list()

        for attribute_name in attribute_list_total:

            if attribute_name in attribute_dict.keys() and attribute_dict[attribute_name] != '':
                Qusetion = "What is {} of the {} ?".format(attribute_name,target_type)
                Answers = [attribute_dict[attribute_name]]
                if attribute_name in common_attribute_dict and common_attribute_dict[attribute_name] != '':
                    Answers.append(common_attribute_dict[attribute_name])
                Question_and_Answers_set.append((Qusetion, Answers))
        question,answers = random.sample(Question_and_Answers_set,1)[0]

        qa_dict = {
            "answers":answers ,
            "object_ids": [target_id]+raw_space_item["anchor_ids"],
            "object_names": [target_type] + raw_space_item["anchors_types"],
            "question": vg_text+'. '+question,
            "scan_id": scene_id,
            "input_bboxes_id": None,
            "output_bboxes_id": [raw_space_item["target_id"]]
        }

        QA_list.append(qa_dict)
    print(QA_list)


    # 物体间功能属性的QA

    for object_x_name, object_y_name in function_relations.keys():
        object_x_type, object_x_id = object_x_name[1:-1].split('_')
        object_y_type, object_y_id = object_y_name[1:-1].split('_')

        # 可能交换
        if np.random.uniform()>0.5:
            object_x_type, object_x_id = object_y_name[1:-1].split('_')
            object_y_type, object_y_id = object_x_name[1:-1].split('_')

        # 第一类问题，框两个物体问功能联系
        question = f'What is the function relationship between these two objects in the {region_id.split("_")[1]}?'
        answers = [function_relations[(object_x_name,object_y_name)].replace(object_x_name,object_x_type).replace(object_y_name,object_y_type)]
        object_list = [(object_x_id, object_x_type),(object_y_id,object_y_type)]
        scan_id = scene_id
        input_bboxes_id = [object_x_id,object_y_id]
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id))

        # 第二类问题，给一个物体和功能联系问另一个物体
        if (object_x_id,object_x_type) not in object_attribute_dict.keys():
            continue
        attribute_dict = object_attribute_dict[(object_x_id,object_x_type)]['attributes']
        common_attribute_dict = object_attribute_dict[(object_x_id,object_x_type)]['common_attribute']
        object_answer_list = []
        for attribute_name in attribute_list_total:
            if attribute_name in attribute_dict and attribute_dict[attribute_name]!='':
                Answers = [attribute_dict[attribute_name]]
                if attribute_name in common_attribute_dict and common_attribute_dict[attribute_name]!='':
                    Answers.append(common_attribute_dict[attribute_name])
                object_answer_list.append((attribute_name,Answers))

        total_num = 4
        object_qa_attr = random.sample(object_answer_list,total_num)
        object_qa_attr.append(('type',[object_x_type]))
        for attribute_name,answers in object_answer_list:
            question = f'In the {region_id.split("_")[1]}, this {object_y_type} and X have relationship in their function:{function_relations[(object_x_name,object_y_name)]}\
                       What is the {attribute_name} of X?'

            object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
            scan_id = scene_id
            input_bboxes_id = [object_y_id]
            output_bboxes_id = [object_x_id]
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, output_bboxes_id))

    # 框多个物体问大类作用
    for object_name_list in large_class_dict.keys():
        object_id_list = [object_name[1:-1].split('_')[1] for object_name in object_name_list]

        question = f'What is the joint function between these {len(object_name_list)} objects in the {region_id.split("_")[1]}?'
        answers = [large_class_dict[object_name_list]]
        object_list = [(object_name[1:-1].split('_')[1],object_name[1:-1].split('_')[0]) for object_name in object_name_list]
        scan_id = scene_id
        input_bboxes_id = object_id_list
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id))

    # 物体间各种属性的QA
    for attribute_name in common_attribute_categorize:


        # 第一类，比较QA，给定两个框，问在某种属性上是不是同种

        #数量较多要sample且保证正负样本一样多

        useful_keys = []



        for key_ in common_attribute_categorize[attribute_name].keys():
            if len(common_attribute_categorize[attribute_name][key_])>1:

                useful_keys.append(key_)
        # 正样本选取
        for key_ in useful_keys:
            (object_x_id, object_x_type),(object_y_id, object_y_type) = random.sample(common_attribute_categorize[attribute_name][key_], 2)
            object_x_id = int(object_x_id)
            object_y_id = int(object_y_id)
            question = f'Are these two objects same in {attribute_name}? Why?'
            answers = [f'No, the {object_x_type} and the {object_y_type} are both {key_}']
            object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
            scan_id = scene_id
            input_bboxes_id = [object_x_id, object_y_id]
            output_bboxes_id = None
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, output_bboxes_id))


        # 负样本选取，为了保证正负样本数一致sample
        import itertools
        key_pair = list(itertools.product(useful_keys,useful_keys))
        key_choices = []
        for key1,key2 in key_pair:
            if key1!=key2:
                key_choices.append((key1,key2))

        key_sample = random.sample(key_choices,len(useful_keys))


        for key1,key2 in key_sample:
            object_x_id,object_x_type = random.sample(common_attribute_categorize[attribute_name][key1],1)[0]
            object_x_id = int(object_x_id)
            object_y_id, object_y_type = random.sample(common_attribute_categorize[attribute_name][key2], 1)[0]
            object_y_id = int(object_y_id)
            question = f'Are these two objects same in {attribute_name}? Why?'
            answers = [f'No, the {object_x_type} is {key1} and the {object_y_type} is {key2}']
            object_list = [(object_x_id,object_x_type),(object_y_id, object_y_type)]
            scan_id = scene_id
            input_bboxes_id = [object_x_id,object_y_id]
            output_bboxes_id = None
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, output_bboxes_id))





        # 第二类，归类QA，给定一个框，问与它在某种属性上相同的物体

        for key_ in common_attribute_categorize[attribute_name].keys():
            if len(common_attribute_categorize[attribute_name][key_])>1:
                object_id, object_type = random.sample(common_attribute_categorize[attribute_name][key_],1)[0]
                object_id = int(object_id)
                question = f'What other objects are same as this object in the {attribute_name}?'
                answers = [f'Their same {attribute_name} is {key_}',f'{key_}']
                object_list = [(int(id_),type_) for id_,type_ in
                               common_attribute_categorize[attribute_name][key_]]
                scan_id = scene_id
                input_bboxes_id = [object_id]
                output_bboxes_id = [int(id_) for id_,type_ in
                               common_attribute_categorize[attribute_name][key_] if int(id_)!=object_id]
                QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id,output_bboxes_id))


    return QA_list



    
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

    ex_function_dict = {('<stool_29>', '<table_3>'): 'The <stool_29> is used for sitting at the <table_3>.',
                        ('<stool_30>', '<table_3>'): 'The <stool_30> is used for sitting at the <table_3>.',
                        ('<stool_31>', '<table_3>'): 'The <stool_31> is used for sitting at the <table_3>.',
                        ('<stool_32>', '<table_3>'): 'The <stool_32> is used for sitting at the <table_3>.', (
                        '<cup_154>',
                        '<table_3>'): 'The <cup_154> is placed on <table_3> to hold a beverage which can be enjoyed while sitting on the stools.',
                        ('<object_155>',
                         '<table_3>'): 'The <object_155> is placed on <table_3> for functional or decorative purposes.'}
    ex_special_dict = {
        '<table_3>': 'The <table_3> is the centerpiece of the dining area, serving as the main surface for meals and gatherings.'}
    ex_large_class_dict = {('<table_3>', '<stool_29>', '<stool_30>', '<stool_31>',
                            '<stool_32>'): 'Belong to the class of dining furniture, together they form a dining set used for eating and socializing.',
                           ('<cup_154>', '<table_3>'): 'Belong to the class of tableware.'}
    ex_region_anno_dict = [{}, ex_function_dict, {}, ex_special_dict, ex_large_class_dict]

    object_attribute_dict = get_object_attribute_dict(scene_id)

    common_attribute_categorize = generate_common_attribute_categorize(scene_id)
    # print(common_attribute_categorize)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


    # single QA:单个物体属性
    print(generate_attribute_QA(scene_id,object_attribute_dict))

    with open("attribute_QA.json", "w") as f:

        json.dump(generate_attribute_QA(scene_id,object_attribute_dict), f,cls=NpEncoder)


    with open("all_sr3d_relations.json", 'r', encoding='UTF-8') as f:
        sr3d_dict = json.load(f)
    #
    # # relation QA:多个物体关系
    print(generate_relation_QA(ex_region_anno_dict,common_attribute_categorize,object_attribute_dict,sr3d_dict,scene_id,region_id='1_dining region'))

    with open("relation_QA.json", "w") as f:

        json.dump(generate_relation_QA(ex_region_anno_dict,common_attribute_categorize,object_attribute_dict,sr3d_dict,scene_id,region_id='1_dining region'), f,cls=NpEncoder)











