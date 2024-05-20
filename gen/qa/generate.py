import os
import json
import numpy as np
import random
from utils.utils_read import read_annotation_pickle
from object_text_anno import translate
from utils.openai_api import mimic_chat_budget, get_content_groups_from_source_groups
from gen.vg.generate import generate_common_attribute_categorize


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
attribute_list_total = ["fine grained category","type",
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
All_VG_SUB_CLASS=['QA_Attribute_Presence','QA_Attribute_Quantity','QA_Attribute_single0','QA_Attribute_single1','QA_Relation_Function0',
                  'QA_Relation_Function1','QA_Relation_Function2','QA_Relation_Function3',"QA_Relation_Attribute0","QA_Relation_Attribute1",'QA_Relation_Space']
EXCLUDED_OBJECTS = ["wall", "ceiling", "floor", "pillar", "column", "roof", "ridge", "arch", "beam"]

DATA_ROOT = "data/"
SPLIT_DATA_ROOT = "splitted_infos/"

def check_attribute(text):
    if text is None or text=='' or text=="N/A" or not isinstance(text,str):
        return False
    return True
def strict_check(text):
    angle_list = []
    angle_dict = {}
    for _index1 in range(len(text)):
        if text[_index1] == '<':
            for _index2 in range(_index1+1,len(text)):
                if text[_index2]=='>':
                    break
            angle_list.append(text[_index1:_index2+1])
            if text[_index1:_index2+1] not in angle_dict.keys():
                angle_dict[text[_index1:_index2+1]] =1
            else:
                angle_dict[text[_index1:_index2 + 1]] += 1
    return angle_list,angle_dict
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
    print(fix_anno['object_ids'])
    return fix_anno
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
def generate_qa_dict(question,answers,object_list,scan_id,input_bboxes_id = None,input_bboxes_=None, output_bboxes_id = None,output_bboxes_=None,sub_class=''):
    qa_dict = {
        "sub_class":sub_class,
        "answers":answers,
        "object_ids": [t[0] for t in object_list],
        "object_names": [t[1] for t in object_list],
        "question":question,
        "scan_id":scan_id,
        "input_bboxes_id":input_bboxes_id,
        "input_bboxes":input_bboxes_,
        "output_bboxes_id":output_bboxes_id,
        "output_bboxes":output_bboxes_
    }

    return qa_dict


def generate_single_item_QA(json_path,scene_id):
    pass

def generate_attribute_QA(scene_id,object_attribute_dict):
    annotation_data = read_annotation_pickle(f'splitted_infos/{scene_id}.pkl')[scene_id]
    object_ids = annotation_data['object_ids']
    object_types = annotation_data['object_types']
    object_bboxes = annotation_data['bboxes']
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
        # object这种type要另外考虑
        if object_types[index_]== 'object':
            continue
        if object_types[index_] in object_exist_dict.keys():
            object_exist_dict[object_types[index_]].append(object_ids[index_])
        else:
            object_exist_dict[object_types[index_]]=[object_ids[index_]]

    object_type_ids_list = []
    for key_ in object_exist_dict.keys():
        object_type_ids_list.append((key_,object_exist_dict[key_]))
    object_type_ids = random.sample(object_type_ids_list,min(len(object_type_ids_list),QA_generate_num_for_one_scene['Presence and quantity']//2))

    pos_types = list(object_exist_dict)
    neg_types = [type_ for type_ in all_types if type_ not in pos_types and type_!='object']

    # 数量划分
    pos_num = min(len(pos_types),QA_generate_num_for_one_scene['Presence and quantity']//4)
    neg_num = QA_generate_num_for_one_scene['Presence and quantity']//2-pos_num
    pos_choice = random.sample(pos_types,pos_num)
    neg_choice = random.sample(neg_types,neg_num)


    # 存在性(pos+neg)
    for object_type in pos_choice:
        question = f'Is there a {object_type} in the room?'
        answers = ['Yes.','Yes, there is.']
        object_list = [(object_id, object_type) for object_id in object_exist_dict[object_type]]
        scan_id = scene_id


        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id,sub_class='QA_Attribute_Presence'))
    for object_type in neg_choice:
        question = f'Is there a {object_type} in the room?'
        answers = ['No.',"No, there isn't."]
        object_list = []
        scan_id = scene_id
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id,sub_class='QA_Attribute_Presence'))


    # 数量
    for object_type,object_ids_ in object_type_ids:
        question = f'How many {object_type}s are there in the room?'
        answers = [f'{len(object_ids_)}']
        object_list = [(object_id, object_type) for object_id in object_ids_]
        scan_id = scene_id
        QA_list.append(generate_qa_dict(question, answers, object_list, scan_id,sub_class='QA_Attribute_Quantity'))

    # 物体属性QA


    # 此处的文本描述还要改进,这里为了方便用比较生硬的语言
    for object_id,object_type in object_attribute_dict.keys():

        object_bbox = object_bboxes[list(object_ids).index(object_id)]
        attribute_dict = object_attribute_dict[(object_id,object_type)]['attributes']
        common_attribute_dict = object_attribute_dict[(object_id,object_type)]['common_attribute']

        Question_and_Answers_set = list()


        for attribute_name in attribute_list_total:

            if attribute_name in attribute_dict.keys() and check_attribute(attribute_dict[attribute_name]):

                Qusetion = "Its {} is {}. ".format(attribute_name,attribute_dict[attribute_name])
                Answers = [attribute_dict[attribute_name]]
                if attribute_name in common_attribute_dict and common_attribute_dict[attribute_name]!='':
                    Answers.append(common_attribute_dict[attribute_name])
                Question_and_Answers_set.append((attribute_name,Qusetion,Answers))
        total_num = 4 # 控制总数
        # 第一类属性QA: 给bboxes问属性
        for attribute_name,Qusetion,Answers in random.sample(Question_and_Answers_set,total_num):
            question = f'What is {attribute_name} of this object?'
            answers = Answers
            object_list = [(object_id, object_type)]
            scan_id = scene_id
            input_bboxes_id = [object_id]
            input_bboxes_ = [object_bbox]
            QA_list.append(generate_qa_dict(question,answers,object_list,scan_id,input_bboxes_id,input_bboxes_,sub_class='QA_Attribute_single0'))



        # 第二类属性QA：给一定属性描述，给出bboxes和另外的描述

        total_num = 4 # 控制总数
        require_indices = random.sample(list(range(len(Question_and_Answers_set))),total_num)
        for _index1 in require_indices:
            question = f'I will give you a description of an item in the room:'
            for _index2 in range(len(Question_and_Answers_set)):
                if _index1==_index2:
                    continue
                question+= Question_and_Answers_set[_index2][1]
            question += f'What is {Question_and_Answers_set[_index1][0]} of this object?'
            answers = Question_and_Answers_set[_index1][2]
            object_list = [(object_id, object_type)]
            scan_id = scene_id
            output_bboxes_id = [object_id]
            output_bboxes_ = [object_bbox]
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, None, None,output_bboxes_id,output_bboxes_,sub_class='QA_Attribute_single1'))


    return QA_list

def generate_relation_QA(region_anno_dict,annotation_data,common_attribute_categorize,object_attribute_dict,sr3d_dict,scene_id,region_id):

    QA_list = []

    object_ids = annotation_data[scene_id]['object_ids']


    object_types = annotation_data[scene_id]['object_ids']

    object_bboxes = annotation_data[scene_id]['bboxes']

    raw_space_list = filter_repeat_utterance(sr3d_dict[scene_id])


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

        all_types = raw_space_item["anchors_types"] + [raw_space_item["instance_type"]]
        skip_flag = False
        for o_type in all_types:
            if o_type in EXCLUDED_OBJECTS:
                skip_flag = True
        if skip_flag:
            continue
        target_id = raw_space_item["target_id"]
        target_type = raw_space_item["instance_type"]
        if (target_id, target_type) not in object_attribute_dict.keys():
            continue

        attribute_dict = object_attribute_dict[(target_id, target_type)]['attributes']
        common_attribute_dict = object_attribute_dict[(target_id, target_type)]['common_attribute']

        Question_and_Answers_set = list()

        for attribute_name in attribute_list_total:
            if attribute_name in ["fine grained category","type",
                                "coarse grained category","function"]:
                continue

            if attribute_name in attribute_dict.keys() and check_attribute(attribute_dict[attribute_name]):
                Qusetion = "What is {} of the {} ?".format(attribute_name,target_type)
                Answers = [attribute_dict[attribute_name]]
                if attribute_name in common_attribute_dict and common_attribute_dict[attribute_name] != '':
                    Answers.append(common_attribute_dict[attribute_name])
                Question_and_Answers_set.append((Qusetion, Answers))
        question,answers = random.sample(Question_and_Answers_set,1)[0]

        qa_dict = {
            "sub_class":"QA_Relation_Space",
            "answers":answers ,
            "object_ids": [target_id]+raw_space_item["anchor_ids"],
            "object_names": [target_type] + raw_space_item["anchors_types"],
            "question": vg_text+'. '+question,
            "scan_id": scene_id,
            "input_bboxes_id": None,
            "input_bboxes": None,
            "output_bboxes_id": [raw_space_item["target_id"]],
            "output_bboxes":[object_bboxes[list(object_ids).index(raw_space_item["target_id"])]]
        }

        QA_list.append(qa_dict)



    # 物体间各种属性的QA
    for attribute_name in common_attribute_categorize:


        # 第一类，比较QA，给定两个框，问在某种属性上是不是同种

        #数量较多要sample且保证正负样本一样多

        # size基于原始的bbox获取
        if attribute_name=='size':
            search_range = [object_ids[_index] for _index in range(len(object_ids)) if
                                object_types[_index] not in EXCLUDED_OBJECTS]
            object_ids_filter = [_id for _id in search_range]

            num_pair = max(1,len(object_ids_filter)//4)
            object_choices = random.sample(list(object_ids_filter),num_pair*2)
            for i in range(num_pair):
                object_x_id = object_choices[2*i]
                object_y_id = object_choices[2*i+1]
                object_x_type = object_types[list(object_ids).index(object_x_id)]
                object_y_type = object_types[list(object_ids).index(object_y_id)]
                object_x_bbox = object_bboxes[list(object_ids).index(object_x_id)]
                object_y_bbox = object_bboxes[list(object_ids).index(object_y_id)]

                object_x_volumn = object_x_bbox[3] * object_x_bbox[4] * object_x_bbox[5]
                object_y_volumn = object_y_bbox[3] * object_y_bbox[4] * object_y_bbox[5]

                threshold = 0.2

                # 如何定义比它大?(什么阈值合理)

                if (object_x_volumn < (1 + threshold) * object_y_volumn) and (object_x_volumn > (1-threshold) * object_y_volumn):
                    answers = ['Yes']
                else:
                    answers = ['No']

                question = f'Are these two objects similar in size?'
                object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
                scan_id = scene_id
                input_bboxes_id = [object_x_id, object_y_id]
                input_bboxes = [object_bboxes[list(object_ids).index(object_x_id)],
                                object_bboxes[list(object_ids).index(object_y_id)]]
                QA_list.append(
                    generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes, None, None,
                                     sub_class="QA_Relation_Attribute0"))
            num_query = min(3,len(object_ids_filter)//4)
            object_choices = random.sample(list(object_ids_filter), num_query)
            for i in range(num_query):
                anchor_id = object_choices[i]
                anchor_type = object_types[list(object_ids).index(anchor_id)]
                anchor_bbox = object_bboxes[list(object_ids).index(anchor_id)]

                anchor_volumn = anchor_bbox[3] * anchor_bbox[4] * anchor_bbox[5]

                answer_object = []

                for target_id in object_ids_filter:
                    if target_id==anchor_id:
                        continue
                    target_type = object_types[list(object_ids).index(target_id)]
                    target_bbox = object_bboxes[list(object_ids).index(target_id)]
                    target_volumn = target_bbox[3] * target_bbox[4] * target_bbox[5]
                    threshold = 0.2

                    # 如何定义比它大?(什么阈值合理)

                    if (target_volumn < (1 + threshold) * anchor_volumn) and (target_volumn > (
                            1 - threshold) * anchor_volumn):
                        answer_object.append((target_id,target_type))

                question = f'How many objects are similar to this object in size?'
                object_list = [(anchor_id, anchor_type)]+answer_object
                scan_id = scene_id
                input_bboxes_id = [anchor_id]
                input_bboxes = [anchor_bbox]
                answers = [len(answer_object)]
                QA_list.append(
                    generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes, None,
                                     None,
                                     sub_class="QA_Relation_Attribute1"))
            continue




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
            answers = [f'Yes, the {object_x_type} and the {object_y_type} are both {key_}']
            object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
            scan_id = scene_id
            input_bboxes_id = [object_x_id, object_y_id]
            input_bboxes = [object_bboxes[list(object_ids).index(object_x_id)],object_bboxes[list(object_ids).index(object_y_id)]]
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes,None,None,sub_class="QA_Relation_Attribute0"))


        # 负样本选取，为了保证正负样本数一致sample
        import itertools
        key_pair = list(itertools.product(useful_keys,useful_keys))
        key_choices = []
        for key1,key2 in key_pair:
            if key1!=key2:
                key_choices.append((key1,key2))

        key_sample = random.sample(key_choices,min(len(useful_keys),len(key_choices)))


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
            input_bboxes = [object_bboxes[list(object_ids).index(object_x_id)],
                            object_bboxes[list(object_ids).index(object_y_id)]]
            QA_list.append(
                generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes, None, None,
                                 sub_class="QA_Relation_Attribute0"))


    # 第二类，归类QA，给定一个框，问与它在某种属性上相同的物体

    for key_ in common_attribute_categorize[attribute_name].keys():
        if len(common_attribute_categorize[attribute_name][key_])>1:
            object_id, object_type = random.sample(common_attribute_categorize[attribute_name][key_],1)[0]
            object_id = int(object_id)
            question = f'How many other objects are the same as this object in the {attribute_name}?'

            object_list = [(int(id_),type_) for id_,type_ in
                           common_attribute_categorize[attribute_name][key_]]
            scan_id = scene_id
            input_bboxes_id = [object_id]
            output_bboxes_id = [int(id_) for id_,type_ in
                           common_attribute_categorize[attribute_name][key_] if int(id_)!=object_id]
            answers = [f'{len(output_bboxes_id)}']
            input_bboxes = [object_bboxes[list(object_ids).index(object_id)]]
            #output_bboxes = [object_bboxes[list(object_ids).index(id_)] for id_ in output_bboxes_id]
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id,input_bboxes,None,None,sub_class="QA_Relation_Attribute1"))

    # 物体间功能属性的QA
    print(region_anno_dict)
    for _id in region_anno_dict:
        print(_id)

        function_relations = region_anno_dict[_id]["annotation"][1]

        large_class_dict = region_anno_dict[_id]["annotation"][4]
        for object_x_name, object_y_name in function_relations.keys():
            object_x_type, object_x_id = object_x_name[1:-1].split('_')
            object_y_type, object_y_id = object_y_name[1:-1].split('_')

            # 可能交换
            if np.random.uniform() > 0.5:
                object_x_type, object_x_id = object_y_name[1:-1].split('_')
                object_y_type, object_y_id = object_x_name[1:-1].split('_')

            object_x_id = int(object_x_id)
            object_y_id = int(object_y_id)

            # 第一类问题，框两个物体问功能联系
            question = f'What is the function relationship between these two objects in the {region_id.split("_")[1]}?'
            answers = [
                function_relations[(object_x_name, object_y_name)].replace(object_x_name, object_x_type).replace(
                    object_y_name, object_y_type)]
            object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
            scan_id = scene_id
            input_bboxes_id = [object_x_id, object_y_id]

            input_bboxes = [object_bboxes[list(object_ids).index(object_x_id)],
                            object_bboxes[list(object_ids).index(object_y_id)]]
            QA_list.append(
                generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes, None, None,
                                 sub_class='QA_Relation_Function0'))

            # 第二类问题，给一个物体和功能联系问另一个物体
            if (object_x_id, object_x_type) not in object_attribute_dict.keys():
                continue
            attribute_dict = object_attribute_dict[(object_x_id, object_x_type)]['attributes']
            common_attribute_dict = object_attribute_dict[(object_x_id, object_x_type)]['common_attribute']
            object_answer_list = []
            for attribute_name in attribute_list_total:
                if attribute_name in attribute_dict and check_attribute(attribute_dict[attribute_name]):
                    Answers = [attribute_dict[attribute_name]]
                    if attribute_name in common_attribute_dict and common_attribute_dict[attribute_name] != '':
                        Answers.append(common_attribute_dict[attribute_name])
                    object_answer_list.append((attribute_name, Answers))

            total_num = 2
            object_qa_attr = random.sample(object_answer_list, total_num)

            for attribute_name, answers in object_qa_attr:
                function_text = function_relations[(object_x_name, object_y_name)].replace(object_x_name,'X').replace(object_y_name,object_y_type)
                question = f'In the {region_id.split("_")[1]}, this {object_y_type} and X have relationship in their function:{function_text} What is the {attribute_name} of X?'

                object_list = [(object_x_id, object_x_type), (object_y_id, object_y_type)]
                scan_id = scene_id
                input_bboxes_id = [object_y_id]
                input_bboxes = [object_bboxes[list(object_ids).index(object_y_id)]]

                QA_list.append(
                    generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes, None,
                                     None, sub_class='QA_Relation_Function1'))

        # 框多个物体问大类作用

        for object_name_list in large_class_dict.keys():
            object_id_list = [int(object_name[1:-1].split('_')[1]) for object_name in object_name_list]

            question = f'What is the joint function between these {len(object_name_list)} objects in the {region_id.split("_")[1]}?'

            answers = [large_class_dict[object_name_list]]
            object_list = [(int(object_name[1:-1].split('_')[1]), object_name[1:-1].split('_')[0]) for object_name in
                           object_name_list]
            scan_id = scene_id
            input_bboxes_id = object_id_list
            input_bboxes = [object_bboxes[list(object_ids).index(id_)] for id_ in object_id_list if id_ in list(object_ids)]
            # print(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes,
            #                                 sub_class="QA_Relation_Function2"))
            QA_list.append(generate_qa_dict(question, answers, object_list, scan_id, input_bboxes_id, input_bboxes,
                                            sub_class="QA_Relation_Function2"))

        # 区域的特点

        input_bboxes_id = region_anno_dict[_id]['objects']
        input_bboxes = [object_bboxes[list(object_ids).index(id_)] for id_ in input_bboxes_id]
        object_list = [(id_,object_types[list(object_ids).index(id_)]) for id_ in input_bboxes_id]
        region_features = region_anno_dict[_id]['annotation'][5]
        text = f"There is a {_id.split('_')[1]} in this room, these objects are in it. "
        for apart in region_features.keys():
            qa_text=text + f'What is the {apart} of this region?'
            about_list,_ = strict_check(region_features[apart])
            region_features_text = region_features[apart]
            for item_ in about_list:
                region_features_text = region_features_text.replace(item_,item_[1:-1].split('_')[0])
            answers = [region_features_text]

            QA_list.append(generate_qa_dict(qa_text, answers, object_list, scene_id, input_bboxes_id, input_bboxes,
                                            sub_class="QA_Relation_Function3"))


        #todo 情境QA
        region_QA = region_anno_dict[_id]['annotation'][6]
        for region_Q in region_QA.keys():
            if 'Q' in region_Q:
                region_A = region_Q.replace('Q','A')
                question = region_QA[region_Q]
                answer = region_QA[region_A]
                q_list, _ = strict_check(question)
                a_list, _ = strict_check(answer)
                object_list = []
                input_bboxes_id = []
                for item_ in q_list:
                    question = question.replace(item_, item_[1:-1].split('_')[0])
                    object_list.append((item_[1:-1].split('_')[1],item_[1:-1].split('_')[0]))
                    input_bboxes_id.append(item_[1:-1].split('_')[1])
                for item_ in a_list:
                    answer = answer.replace(item_, item_[1:-1].split('_')[0])
                    object_list.append((item_[1:-1].split('_')[1], item_[1:-1].split('_')[0]))

                input_bboxes = [object_bboxes[list(object_ids).index(id_)] for id_ in input_bboxes_id if id_ in list(object_ids)]

                QA_list.append(generate_qa_dict(question, [answer], object_list, scene_id, input_bboxes_id, input_bboxes,
                                                sub_class="QA_Relation_Function3"))


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
    #scene_id = "scene0000_00"
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
        if os.path.exists(os.path.join(region_text_annos_dir, region_id, 'struction_shujutang_czc.npy')):

            region_anno_dict[region_id] = {"annotation":np.load(os.path.join(region_text_annos_dir,region_id,'struction_shujutang_czc.npy'),allow_pickle=True),
                                                "objects":np.load(os.path.join(region_text_annos_dir,region_id,'object_filter.npy'))}


    # region_anno_dict['1_dinning region']["annotation"] = [{}, {
    #     ('<stool_29>', '<table_3>'): 'The <stool_29> is used for sitting at the <table_3>.',
    #     ('<stool_30>', '<table_3>'): 'The <stool_30> is used for sitting at the <table_3>.',
    #     ('<stool_31>', '<table_3>'): 'The <stool_31> is used for sitting at the <table_3>.',
    #     ('<stool_32>', '<table_3>'): 'The <stool_32> is used for sitting at the <table_3>.', ('<cup_154>',
    #                                                                                           '<table_3>'): 'The <cup_154> is placed on <table_3> to hold a beverage which can be enjoyed while sitting on the stools.',
    #     (
    #     '<object_155>', '<table_3>'): 'The <object_155> is placed on <table_3> for functional or decorative purposes.'},
    #                                                       {'wall': [],
    #                                                        'floor': ['<table_3>', '<stool_29>', '<stool_30>',
    #                                                                  '<stool_31>',
    #                                                                  '<stool_32>']},
    #                                                       {
    #                                                           '<table_3>': '<table_3> is the centerpiece of the dining area, serving as the main surface for meals and gatherings.',
    #                                                           '<stool_29>': '<stool_29> is part of the dining set, its position closest to the camera makes it prominent.',
    #                                                           '<stool_30>': '<stool_30> directly faces the living area, possibly the first choice for someone entering the dining area.',
    #                                                           '<stool_31>': "<stool_31> is tucked more towards the back, indicating less frequent use or reserved for additional guests.",
    #                                                           '<stool_32>': '<stool_32> is the furthest in the dining setup, possibly the last to be used or for someone who prefers a corner seat.',
    #                                                           '<cup_154>': "<cup_154> is the only item on the table, which suggests it's ready for use or has just been used.",
    #                                                           '<object_155>': "<object_155> is notably placed beside <cup_154>, implying it's part of a set or in use along with the cup."},
    #                                                       {('<table_3>', '<stool_29>', '<stool_30>', '<stool_31>',
    #                                                         '<stool_32>'): 'Belong to the class of dining furniture, together they form a dining set used for eating and socializing.',
    #                                                        ('<cup_154>',
    #                                                         '<table_3>'): 'Belong to the class of tableware, where <cup_154> is an item used during the activity of drinking, and <table_3> serves as the support surface for this activity.'},
    #                                                       {
    #                                                           'location and function description': 'The region appears to be a multi-functional space serving as a dining and living area, meant for eating, socializing, and relaxation.',
    #                                                           'space layout and dimensions': 'An open plan layout with the dining area marked by a long table and stools, suggesting a modern and casual living space. The exact dimensions cannot be determined from the photo.',
    #                                                           'doors,windows, walls, floors and ceilings': 'The space has visible wooden flooring, light-colored walls, and a ceiling that reflects a standard residential height. No doors or windows are visible in the image provided.',
    #                                                           'soft fitting elements and decorative details': 'The visible soft fittings include a blue sofa with cushions, indicating a comfortable lounging area. Decorative details are minimal, with personal items on the table and a potential shelf with items in the background.',
    #                                                           'lighting design': 'The lighting cannot be clearly seen, but the ambiance suggests the use of warm, diffused light, possibly from overhead or concealed sources to create a homey atmosphere.',
    #                                                           'color matching and style theme': 'The color theme is neutral, with the wood elements providing a warm tone against the white walls. This suggests a contemporary style with an emphasis on simplicity and functionality.',
    #                                                           'special features': 'The dining area features a distinctively long table paired with stools instead of traditional chairs, highlighting a casual and modern twist to dining furniture.'}]

    object_attribute_dict = get_object_attribute_dict(scene_id)

    common_attribute_categorize = generate_common_attribute_categorize(scene_id)
    with open("all_sr3d_relations.json", 'r', encoding='UTF-8') as f:
        sr3d_dict = json.load(f)

    #sr3d_dict = {scene_id: np.load('bbbbbb.py.npy', allow_pickle=True)}
    #print(sr3d_dict)




    # single QA:单个物体属性
    attribute_QA = generate_attribute_QA(scene_id,object_attribute_dict)


    # relation QA:多个物体关系

    relation_QA=generate_relation_QA(region_anno_dict,annotation_data,common_attribute_categorize,object_attribute_dict,sr3d_dict,scene_id,region_id='1_dining region')



    result_dict = attribute_QA+relation_QA
    count_dict = dict()
    for name_ in All_VG_SUB_CLASS:
        count_dict[name_] = 0

    for dict_ in result_dict:
        dict_['ID'] = dict_['sub_class'] + '_' + str(count_dict[dict_['sub_class']])
        if len(dict_['answers'])==2:
            if dict_['answers'][0].lower()==dict_['answers'][1].lower():
                dict_['answers'] = [dict_['answers'][0]]
        count_dict[dict_['sub_class']] += 1

    with open(f"{scene_id}_QA.json", "w") as f:

        json.dump(result_dict, f,cls=NpEncoder)











