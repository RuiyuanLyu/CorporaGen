import os
import json
from tqdm import tqdm
from utils.openai_api import mimic_chat, mimic_chat_budget, get_content_groups_from_source_groups, num_tokens_from_string, get_full_response
from utils.utils_read import load_json

def list_translate(texts, src_lang="English", tgt_lang="Chinese"):
    """
        Translates a list of text using the OpenAI API.
        Args:
            text: a list of text to be translated.
            src_lang: A string of the source language code.
            tgt_lang: A string of the target language code.
        Returns:
            A list of translated text strings.
    """
    user_message = str(texts)
    src_lang = src_lang.capitalize()
    tgt_lang = tgt_lang.capitalize()
    system_prompt = f'You are an excellent translator.I will give you a list of sentences, you are supposed to translate every sentence in the list from {src_lang} to {tgt_lang}. The result should be only a list (Use double quotation marks as string delimiters, like this:["here is a example","you are happy"]). Repeat: provide a list in the answer.'
    source_groups = [
        [user_message],
    ]
    content_groups = get_content_groups_from_source_groups(source_groups)
    conversation = mimic_chat(content_groups, model="gpt-3.5-turbo-0125", system_prompt=system_prompt)
    for message in conversation:
        if message["role"] == "assistant":
            return message["content"]
        
def disassemble_parts(texts:list[str], max_token_length=600, max_list_length=10):
    """
        disassembles the text into parts that are smaller than the maximum token length.
        Returns:
            A list of parts.
    """
    parts = []
    current_part = []
    current_token_length = 0
    last_i = 0
    for i, text in enumerate(texts):
        if current_token_length+num_tokens_from_string(text)>max_token_length or i-last_i>max_list_length:
            parts.append(current_part)
            current_part = []
            current_token_length = 0
            last_i = i
        current_part.append(text)
        current_token_length += num_tokens_from_string(text)
    if current_part:
        parts.append(current_part)
    return parts

def strict_list_translate_part(texts,src_lang="English", tgt_lang="Chinese",max_try=10):
    """
        Translates a list of text using the OpenAI API.
        Strictly checks the output to make sure the output is a list of strings.
        Returns:
            A list of translated text strings, num tries, and total token usage.
    """
    success = False
    _try = 0
    while not success and _try<max_try:
        _try+=1
        success = True
        out = list_translate(texts,src_lang, tgt_lang)
        a=b=-1
        for _index in range(len(out)):
            if out[_index]=='[':
                a = _index
        for _index in range(len(out)):
            if out[_index]==']':
                b = _index
        if a==-1 or b==-1:
            success = False
            continue
        list_out = out[a:b+1]
        try:
            texts_out = eval(list_out)
        except:
            success = False
            continue

        if len(texts_out)!=len(texts):
            success = False
    if success:
        return texts_out,_try
    else:
        return [],_try
        
def strict_list_translate(texts:list[str],src_lang="English", tgt_lang="Chinese",max_try=10):
    parts = disassemble_parts(texts)
    out_list = []
    max_num_tries = 0
    total_token_usage = {}
    for part in parts:
        part_translated, num_tries = strict_list_translate_part(part, src_lang, tgt_lang, max_try)
        out_list.extend(part_translated)
        max_num_tries = max(max_num_tries, num_tries)
    return out_list, max_num_tries

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


if __name__ == "__main__":

    text = ["serves as a surface for placing items and as a social hub for gatherings","show room's tidiness","provides sleeping"]
    out,_ = strict_list_translate(text,src_lang="English", tgt_lang="Chinese")
    out2,_ = strict_list_translate(out, src_lang="Chinese", tgt_lang="English")
    print(out,out2)
