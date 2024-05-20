import os
import json

import numpy as np
from tqdm import tqdm
from utils.openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups, num_tokens_from_string, get_full_response
def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    else:
        return False
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

def strict_translate(text,src_lang="Chinese", tgt_lang="English",max_try=3,show_out=False,translate_hint = ''):
    list_of_text,dict_of_text = strict_check(text)
    success = False
    try_ = 0
    while try_<max_try and not success:

        gen = translate(text,src_lang,tgt_lang,translate_hint = translate_hint)
        if show_out:
            print(gen)

        list_of_gen,dict_of_gen = strict_check(gen)
        for word_ in list_of_gen:
            gen = gen.replace(word_,word_.lower())
        list_of_gen, dict_of_gen = strict_check(gen)
        try_+=1
        success = True
        for k in dict_of_text.keys():
            if k not in dict_of_gen.keys() or dict_of_gen[k]!=dict_of_text[k]:
                success = False
        for k in dict_of_gen.keys():
            if k not in dict_of_text.keys() or dict_of_gen[k]!=dict_of_text[k]:
                success = False

        for ch in gen:
            if is_chinese(ch):
                success = False


    return gen,try_,success


def translate(text, src_lang="Chinese", tgt_lang="English", translate_hint = ''):
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
    system_prompt1 = f"You are an excellent translator, who does more than rigidly translating {src_lang} into {tgt_lang}. \
                    Your choice of words and phrases is natural and fluent. The expressions are easy to understand. \
                    The expected reader is a middle-school student.\
                    In the original text there will be some '<>'(such as <piano_1>), You must not translate the word in the '<>' and\
                    keep all '<>' unchanged in your result!"
    word_list,_ = strict_check(text)

    explain_text = 'For example, '
    for word_ in word_list:
        explain_text += f'regard "{word_}" as "{word_[1:-1].split("_")[0]}", '
    explain_text = explain_text[:-2]+' when translating. '


    if len(word_list)>0:
        system_prompt = f'You are an excellent translator, who does more than rigidly translating {src_lang} into {tgt_lang}. {translate_hint} You must keep the angle-bracketed words unchanged / unremoved in the result and regarded it as the object type in the "<>" when translating. ({explain_text}). Again, the angle-bracketed words should exist in the result. '
    else:
        system_prompt = f'You are an excellent translator, who does more than rigidly translating {src_lang} into {tgt_lang}. {translate_hint}'
    source_groups = [
        [user_message],
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    conversation = mimic_chat(content_groups, model="gpt-3.5-turbo-0125", system_prompt=system_prompt)
    for message in conversation:
        if message["role"] == "assistant":
            return message["content"]
def process_function_relation(function_relations):
    text_tuple_map = {}

    for object_tuple in function_relations.keys():
        mask_text = function_relations[object_tuple].replace(object_tuple[0], '<mask_1>').replace(object_tuple[1],
                                                                                                  '<mask_2>')
        if mask_text not in text_tuple_map:
            text_tuple_map[mask_text] = [object_tuple]
        else:
            text_tuple_map[mask_text].append(object_tuple)
    new_dict = {}
    for mask_text in text_tuple_map.keys():
        ex_tuple = text_tuple_map[mask_text][0]
        text = mask_text.replace('<mask_1>', ex_tuple[0]).replace('<mask_2>', ex_tuple[1])
        translate_hint = 'The text will describe the relationship of two objects. '
        text, a, b = strict_translate(text,translate_hint=translate_hint)
        print(text,a,b)
        if not b:
            continue


        for tuple_ in text_tuple_map[mask_text]:
            tuple_text = text.replace(ex_tuple[0], tuple_[0]).replace(ex_tuple[1], tuple_[1])
            if tuple_[0] not in tuple_text or tuple_[1] not in tuple_text:
                continue
            new_dict[tuple_] = tuple_text
    # 去除冗余的其它<>
    for object_tuple in new_dict.keys():

        arrow_list, _ = strict_check(new_dict[object_tuple])
        for arrow_name in arrow_list:
            if arrow_name not in object_tuple:
                new_dict[object_tuple] = new_dict[object_tuple].replace(arrow_name, arrow_name.split('_')[0][1:])
    return new_dict
def process_object_function(object_function_dict):
    for object_name in object_function_dict.keys():
        translate_hint = 'The text will describe the features of an object. '
        object_function_dict[object_name],a,b = strict_translate(object_function_dict[object_name],translate_hint=translate_hint)
        print(object_function_dict[object_name],a,b)
        if not b or object_name not in object_function_dict[object_name]:
            object_function_dict[object_name] = ''

    # 去除冗余的其它<>
    for object_name in object_function_dict.keys():
        arrow_list, _ = strict_check(object_function_dict[object_name])
        for arrow_name in arrow_list:
            if arrow_name!=object_name:
                object_function_dict[object_name] = object_function_dict[object_name].replace(arrow_name,arrow_name.split('_')[0][1:])
    return object_function_dict
def process_large_class_dict(large_class_dict):
    for object_list in large_class_dict.keys():
        translate_hint = 'The text will describe the function of some objects. '
        large_class_dict[object_list] = strict_translate(large_class_dict[object_list],translate_hint=translate_hint)[0]
    return large_class_dict
def process_region_feature_dict(region_feature_dict):
    for d_name in region_feature_dict.keys():
        translate_hint = 'The text will describe the features of a region of the house. '
        region_feature_dict[d_name] = strict_translate(region_feature_dict[d_name],translate_hint=translate_hint)[0]
    for d_name in region_feature_dict.keys():
        arrow_list, _ = strict_check(region_feature_dict[d_name])
        for arrow_name in arrow_list:
            region_feature_dict[d_name] = region_feature_dict[d_name].replace(arrow_name,arrow_name.split('_')[0][1:])

    return region_feature_dict
def process_QA_dict(QA_dict):
    for qa in QA_dict.keys():
        if 'Q' in qa:
            translate_hint = 'The text will a question. Translate the question from Chinese to English instead of answering the question. '
        else:
            flag= False

            # 没有中文自动跳过翻译
            for ch in QA_dict[qa]:
                if is_chinese(ch):
                    flag=True
            if not flag:
                continue
            translate_hint = ''

        QA_dict[qa],a,b = strict_translate(QA_dict[qa],translate_hint=translate_hint,max_try=8)
        print(QA_dict[qa],a,b)

    return QA_dict
def region_anno_translate_back(scene_id):
    for region_id in os.listdir(f'data/{scene_id}/region_views/'):
        if region_id[-4:]=='.png':
            continue
        region_npy_path = f'data/{scene_id}/region_views/{region_id}/struction_trans_shujutang_czc.npy'
        region_info = np.load(region_npy_path,allow_pickle=True)

        loc_relation_dict = region_info[0]
        print(region_info[1])
        function_relations = process_function_relation(region_info[1])
        print(function_relations)
        wall_floor_addition = region_info[2]
        print(region_info[3])
        object_function_dict = process_object_function(region_info[3])
        print(object_function_dict)

        print(region_info[4])
        large_class_dict = process_large_class_dict(region_info[4])
        print(large_class_dict)
        print(region_info[5])
        region_feature_dict = process_region_feature_dict(region_info[5])
        print(region_feature_dict)
        print(region_info[6])
        QA_dict = process_QA_dict(region_info[6])
        print(QA_dict)

        np.save(f'data/{scene_id}/region_views/{region_id}/struction_shujutang_czc.npy',[loc_relation_dict,function_relations,wall_floor_addition,object_function_dict,large_class_dict,region_feature_dict,QA_dict])







if __name__ == '__main__':

    #a,b,c = strict_translate('<lamp_22>为<couch_41>提供阅读或其他活动的光，所以<lamp_22>很好用',translate_hint='The text will describe the relationship of two objects. ')
    #print(a,b,c)
    scene_id = "1mp3d_0000_region10"
    region_anno_translate_back(scene_id)