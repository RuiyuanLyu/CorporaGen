
# -*- coding:utf-8 –*-
import os

import gradio as gr  # Now: gradio==3.50.2 Deprecated: gradio==3.44.0
import numpy as np
import json
from utils_read import read_annotation_pickles

# 最大可能包含的物品数
MAX_OBJECT_NUM =30

# 区域字典映射
REGIONS = {"起居室/会客区": "living region",
           "书房/工作学习区": "study region",
           "卧室/休息区": "resting region", # previously "sleeping region"
           "饭厅/进食区": "dinning region",
           "厨房/烹饪区": "cooking region",
           "浴室/洗澡区": "bathing region",
           "储藏区": "storage region",
           "厕所/洗手间": "toliet region",
           "走廊/通道": "corridor region",
           "空地": "open area region",
           "其它": "other region"}
REGIONS_tran = {REGIONS[key]: key for key in REGIONS.keys()}


scene_list = ['scene0000_00']
region_list = ['0_cooking region']
anno_full =read_annotation_pickles(["embodiedscan_infos_train_full.pkl", "embodiedscan_infos_val_full.pkl",
                                               "3rscan_infos_test_MDJH_aligned_full_10_visible.pkl","matterport3d_infos_test_full_10_visible.pkl"])

region_view_dir_name = 'region_view_test'



def read_str_list_from_json(file_name,encoding='gbk'):
    with open(file_name,'r',encoding=encoding) as f:
        out = json.load(f)
    result_=[]
    for key_ in out.keys():

        text = ''
        for t in out[key_].split('\n\n'):
            text += t
        result_.append(text)

    return result_


with gr.Blocks() as demo:

    def data_process(scene_id,region_id):
        if scene_id==None or region_id==None:
            help_check = gr.Dropdown([], visible=False)
            core_question = gr.Radio(label=f"Question 0", choices=["是", "否", "不要选这个选项"], value="不要选这个选项",
                                     info="这三段描述与图片对应区域是否对应", interactive=True, visible=False)


            return [],None,None,None,None,None,None,None,[],help_check,[],core_question,['','','']

        show_image_path =[f'data/{scene_id}/{region_view_dir_name}/{region_id}/'+img_file for img_file in os.listdir(f'data/{scene_id}/{region_view_dir_name}/{region_id}') if img_file[-4:]=='.jpg']


        English_annotation_path = f'data/{scene_id}/{region_view_dir_name}/{region_id}/English.json'
        Chineses_annotation_path = f'data/{scene_id}/{region_view_dir_name}/{region_id}/Chinese.json'

        id_list = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/object_filter.npy')
        anno_single = anno_full.get(scene_id)
        object_list = [f'<{anno_single["object_types"][list(anno_single["object_ids"]).index(int(id_))]}_{int(id_)}>' for id_ in id_list]


        Chinese_annotation = read_str_list_from_json(Chineses_annotation_path,encoding='utf-8')

        full_text = Chinese_annotation[0]+'\n'+Chinese_annotation[1]+'\n'+Chinese_annotation[2]


        region_info_text = f'选取的区域为{REGIONS_tran[region_id.split("_")[1]]}，总共有{len(show_image_path)}张展示图片，包含的物体个数为{len(id_list)}，分别是：'

        for object_ in object_list:
            region_info_text+=object_+'，'
        region_info_text=region_info_text[:-1]+';'
        region_text_anno_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/region_text_anno_info.npy',allow_pickle=True).item()
        forget_id_list = region_text_anno_info['forget_id']
        forget_object_list = [f'<{anno_single["object_types"][list(anno_single["object_ids"]).index(int(id_))]}_{int(id_)}>'
                       for id_ in forget_id_list]

        if len(forget_object_list)>0:
            region_info_text+= '在下面的文本中有这些遗漏的物体需要补充：'
            for forget_object in forget_object_list:
                region_info_text+=forget_object+'，'
        region_info_text = region_info_text[:-1] + '。'

        radio_result_dict0 = {}

        help_check = gr.Dropdown(object_list,visible=True,interactive=True)
        core_question = gr.Radio(label=f"Question 0", choices=["是", "否", "不要选这个选项"], value="不要选这个选项",
                                 info="这三段描述与图片对应区域是否对应", interactive=True, visible=True)



        return show_image_path,gr.update(value=show_image_path[0]),full_text,None,region_info_text,*Chinese_annotation,radio_result_dict0,help_check,object_list,core_question,Chinese_annotation

    def img_change(show_image_index,show_image_path):
        img_change_path = show_image_path[(show_image_index+1)%len(show_image_path)]
        return show_image_index+1, gr.update(value= img_change_path)

    def textbox_refresh(core,object_name_id_list):
        object_with_adj = []
        if core=='是':

            for name_id in object_name_id_list:
                r = gr.Textbox(label=name_id,info=f'关于{name_id}的描述',visible=True,interactive=True)
                object_with_adj.append(r)
            for _ in range(MAX_OBJECT_NUM-len(object_name_id_list)):
                object_with_adj.append(gr.Textbox(visible=False))
        else:
            for i in range(MAX_OBJECT_NUM):
                r = gr.Textbox(label=f'{i}', visible=False)
                object_with_adj.append(r)

        return object_with_adj


    def dict_refresh(core_question, radio_result_dict):
        radio_result_dict['all'] = [core_question]
        allow_save = False
        for part in radio_result_dict.keys():
            for choice in radio_result_dict[part]:
                if choice == "不要选这个选项":
                    allow_save = False
        if radio_result_dict['all'][0] == '否':
            allow_save = True
        b = gr.Button("保存标注", visible=allow_save)
        return radio_result_dict,b

    def text_reset(orginal_text):

        return orginal_text[0],orginal_text[1],orginal_text[2]
    def annotation_save(scene_id,region_id,object_text,group_text,region_text,radio_result_dict):
        info_json = json.dumps(radio_result_dict, sort_keys=False, indent=4, separators=(',', ': '))
        with open(f'data/{scene_id}/{region_view_dir_name}/{region_id}/result_dict.json', 'w',encoding='utf-8') as f:
            f.write(info_json)
        annotation_json = json.dumps({'object':object_text,'group':group_text,'region':region_text})
        with open(f'data/{scene_id}/{region_view_dir_name}/{region_id}/result_annotation.json', 'w',encoding='utf-8') as f:
            f.write(annotation_json)

        radio_result_dict0 = {}

        return None,None,None,[],0,radio_result_dict0,None
    def main_reset():
        core_question = gr.Radio(label=f"Question 0", choices=["是", "否", "不要选这个选项"], value="不要选这个选项",
                                 info="这三段描述与图片对应区域是否对应", interactive=True, visible=True)
        object_text = gr.Textbox(interactive=False)
        group_text = gr.Textbox(interactive=False)
        region_text = gr.Textbox(interactive=False)
        Save_button = gr.Button("保存标注",visible=False)
        return core_question,object_text,group_text,region_text,Save_button


    def image_match(scene_id,region_id,help_check):
        id_ = int(help_check[1:-1].split('_')[1])
        path_ = f'data/{scene_id}/painted_objects'
        target = None
        for file in os.listdir(path_):
            if file[-4:] == '.jpg' and int(file[:3])==id_:
                target = path_+'/'+file
        if target!= None:
            return gr.update(value= target)
        else:
            return None
    def adj_update(r,label,obj_adj_dict):

        if r!=None:
            obj_adj_dict[label] = r

        return obj_adj_dict


    def text_process(obj_adj_dict, object_text, group_text, region_text,object_name_id_list):
        for obj_name_id in obj_adj_dict.keys():

            adj_word = obj_adj_dict[obj_name_id]
            if adj_word=='':
                continue
            obj_name = object_name_id_list[int(obj_name_id)]
            object_text = object_text.replace(obj_name,'<('+adj_word+')'+obj_name[1:-1]+'>')
            group_text = group_text.replace(obj_name, '<(' + adj_word + ')' + obj_name[1:-1] + '>')
            region_text = region_text.replace(obj_name, '<(' + adj_word + ')' + obj_name[1:-1] + '>')
        return object_text, group_text, region_text

    def text_sync(object_text_1,object_text):
        if object_text_1!=object_text:
            return object_text_1
        else:
            return object_text





    show_image_index = gr.State(0)
    show_image_path = gr.State([])
    radio_result_dict = gr.State({})
    object_name_id_list = gr.State([])
    original_text = gr.State(['','',''])

    obj_adj_dict = gr.State({})
    scene_choice = gr.Dropdown(scene_list, label="对应场景")
    region_choice = gr.Dropdown(region_list, label="对应区域")
    region_info_text = gr.Textbox(label="区域信息",value="请选择区域")
    with gr.Row():
        show_image = gr.Image(label='区域场景展示')
        with gr.Column():
            English_text_box = gr.Textbox(label='文本标注',interactive=False)
            Chinese_text_box = gr.Textbox(label='文本翻译', interactive=False)


    core_question = gr.Radio(label=f"Question 0", choices=["是","否","不要选这个选项"], value="不要选这个选项", info="这三段描述与图片对应区域是否对应", interactive=True, visible=False)

    # object level
    object_text = gr.Textbox(info='object level', interactive=True)

    # group level
    group_text = gr.Textbox(info='group level',interactive=True)

    # region level
    region_text = gr.Textbox(info='region level',interactive=True)
    Reset_button = gr.Button("重置文本")
    Save_button = gr.Button("保存标注",visible=True)
    help_check = gr.Dropdown([], visible=False)
    help_look = gr.Image()

    with gr.Row():
        object_text_1 = gr.Textbox(info='object level', interactive=True)

        object_with_adj = []
        with gr.Column():
            for i in range(MAX_OBJECT_NUM):
                r = gr.Textbox(label=f'{i}',visible=False)
                object_with_adj.append(r)
    match_button = gr.Button('形容词匹配')

    region_choice.change(fn=data_process, inputs=[scene_choice,region_choice],outputs=[show_image_path,show_image,English_text_box,Chinese_text_box,region_info_text,object_text,group_text,region_text,radio_result_dict,help_check,object_name_id_list,core_question,original_text])
    show_image.select(fn=img_change,inputs=[show_image_index,show_image_path],outputs=[show_image_index,show_image])
    core_question.change(fn=textbox_refresh, inputs=[core_question,object_name_id_list],outputs=object_with_adj)
    Reset_button.click(fn=text_reset,inputs=[original_text],outputs=[object_text,group_text,region_text])
    Save_button.click(fn=annotation_save,inputs=[scene_choice,region_choice,object_text,group_text,region_text,radio_result_dict],
                      outputs=[scene_choice,region_choice,show_image,show_image_path,show_image_index,radio_result_dict,help_look])
    Save_button.click(fn=main_reset,inputs=[],outputs=[core_question,object_text,group_text,region_text,Save_button])

    match_button.click(fn=text_process,inputs=[obj_adj_dict,object_text,group_text,region_text,object_name_id_list],outputs=[object_text,group_text,region_text] )
    help_check.select(fn=image_match, inputs=[scene_choice,region_choice,help_check],outputs=[help_look])
    object_text_1.change(fn=text_sync, inputs=[object_text_1,object_text],outputs=[object_text])
    object_text.change(fn=text_sync, inputs=[object_text, object_text_1], outputs=[object_text_1])
    for r in object_with_adj:
        r.change(fn=adj_update,inputs=[r,gr.State(r.label),obj_adj_dict],outputs=[obj_adj_dict])



demo.queue(concurrency_count=20)

if __name__ == "__main__":

    demo.launch(show_error=True)

