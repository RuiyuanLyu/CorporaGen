import os
import json
from tqdm import tqdm
from openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups, num_tokens_from_string, get_response
from utils.utils_read import load_json


def translate(text, src_lang="English", tgt_lang="Chinese"):
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
    system_prompt = f"You are an excellent translator, who does more than rigidly translating {src_lang} into {tgt_lang} all the word in . In the original text there will be some '<>',you must not translate the word in the '<>'.Your choice of words and phrases is natural and fluent. The expressions are easy to understand. The expected reader is a middle-school student.Some original items in the text appear in the form of an angle bracket <name_id>,please keep them unchange in your result! And regard it as a noun with object type in the angle bracket(such as '<piano_1>' in the text, you should keep the '<piano_1>' unchange and view it as 'a piano').Notice you again, remember to keep the '<>'and words in it unchange.Again, don't translate the word in the '<>',this is very important, don't generate any new '<>' in your result!!!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ) while tanslating!"#Please keep all ‘<>’ and ids in them(such as <piano_01>)"
    source_groups = [
        [user_message],
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    conversation = mimic_chat(content_groups, model="gpt-3.5-turbo-0125", system_prompt=system_prompt)
    for message in conversation:
        if message["role"] == "assistant":
            return message["content"]

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

def strict_translate(text,src_lang,tgt_lang,max_try=8,show_out=False):
    list_of_text,dict_of_text = strict_check(text)

    success = False
    try_ = 0
    while try_<max_try and not success:

        gen = translate(text,src_lang,tgt_lang)
        if show_out:
            print(gen)

        list_of_gen,dict_of_gen = strict_check(gen)
        try_+=1
        success = True
        for k in dict_of_text.keys():
            if k not in dict_of_gen.keys() or dict_of_gen[k]!=dict_of_text[k]:
                success = False
        for k in dict_of_gen.keys():
            if k not in dict_of_text.keys() or dict_of_gen[k]!=dict_of_text[k]:
                success = False


    return gen,success

if __name__ == "__main__":

    text = "这个<bed_38>位于中央，作为睡眠区的焦点，上面放着一个<(舒适的)pillow_17>。床边放着一块<(柔软的)carpet_166>，增加了脚下的舒适度。床上方挂着一幅<picture_159>，增添了个人色彩。附近，<backpack_63>靠在床边，暗示着活动性和学习习惯。一个<object_165>，可能是床头柜，悄悄地靠在床边，使必需品近在手边。最后，床边附近的一个未知<object_62>可能用于储物或装饰用途。"
    out,_ = strict_translate(text,src_lang="Chinese", tgt_lang="English")

