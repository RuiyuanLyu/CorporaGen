# -*- coding:utf-8 –*-
import os

import gradio as gr  # Now: gradio==3.50.2 Deprecated: gradio==3.44.0
import numpy as np
import json

# 最大可能包含的物品数
MAX_OBJECT_NUM =30

# 区域名称映射
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

# 位置关系映射
RELATIONS = {'in':'在里面',
            'on':'在上面',
            'above':'在上方',
            'under':'在下方',
            'closed by':'紧贴着',
            'next to':'相邻着',
            'hanging/lying on':'悬挂在/倚靠在',}
RELATIONS_tran = {RELATIONS[key]: key for key in RELATIONS.keys()}

# 区域特点映射
FEATURES = {'location and function description':'区域定位和功能描述',
            'space layout and dimensions':'空间布局与尺寸',
            'doors,windows, walls, floors and ceilings':'门窗, 墙面,地面与天花板',
            'soft fitting elements and decorative details':'软装元素与装饰细节',
            'lighting design':'照明设计',
            'color matching and style theme':'色彩搭配与风格主题',
            'special features':'特殊化特点'}

FEATURES_tran = {FEATURES[key]: key for key in FEATURES.keys()}

scene_list = ['scene0000_00']
region_list = ['4_toliet region']

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
            check_id_choice = gr.Dropdown([], visible=False)
            loc_relation_frame = gr.Dataframe(visible=False)
            logic_relation_frame = gr.Dataframe(visible=False)
            large_class_frame = gr.Dataframe(visible=False)
            wall_text = gr.Textbox(visible=False)
            floor_text = gr.Textbox(visible=False)

            return [],loc_relation_frame,logic_relation_frame,large_class_frame,check_id_choice,wall_text,floor_text,None,None,None,None,None,[{}]*6,[{}]*6,'请选择区域',None,None,None,None

        show_image_path =[f'data/{scene_id}/{region_view_dir_name}/{region_id}/'+img_file for img_file in os.listdir(f'data/{scene_id}/{region_view_dir_name}/{region_id}') if img_file[-4:]=='.jpg']
        region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans.npy',allow_pickle=True)
        loc_relation_dict = region_info[0]
        logic_relation_dict = region_info[1]
        wall_floor_addition = region_info[2]
        object_function_dict = region_info[3]
        large_class_dict = region_info[4]
        region_feature_dict = region_info[5]


        loc_relation_list = []
        for object_tuple in loc_relation_dict.keys():
            loc_relation_list.append([object_tuple[0],object_tuple[1],RELATIONS[loc_relation_dict[object_tuple]]])
        loc_relation_frame = gr.Dataframe(
            label= '物体间位置关系表格',
            headers=["物体A", "物体B", "物体A相对于物体B位置关系"],
            datatype=["str", "str", "str"],
            row_count=len(loc_relation_dict.keys()),
            col_count=(3, "fixed"),
            value=loc_relation_list,
            visible=True,
            interactive=True
        )
        logic_relation_list = []
        for object_tuple in logic_relation_dict.keys():
            logic_relation_list.append([object_tuple[0], object_tuple[1], logic_relation_dict[object_tuple]])
        logic_relation_frame = gr.Dataframe(
            label='物体间逻辑关系表格',
            headers=["物体A", "物体B", "物体A相对于物体B逻辑关系"],
            datatype=["str", "str", "str"],
            row_count=len(logic_relation_dict.keys()),
            col_count=(3, "fixed"),
            value=logic_relation_list,
            visible=True,
            interactive=True
        )
        large_class_list = []
        cnt=0
        for object_tuple in large_class_dict.keys():
            cnt+=1
            object_text = ''
            for t in object_tuple:
                object_text+=(t+'/')
            large_class_list.append([object_text[:-1], large_class_dict[object_tuple]])

        large_class_frame = gr.Dataframe(
            label='物体大类/功能组表格',
            headers=["物体列表", "描述"],
            datatype=["str", "str"],
            row_count=cnt,
            col_count=(2, "fixed"),
            value=large_class_list,
            visible=True,
            interactive=True
        )

        wall_addition_str = floor_addition_str = ''
        for object_name in wall_floor_addition['wall']:
            wall_addition_str += f'{object_name}, '
        for object_name in wall_floor_addition['floor']:
            floor_addition_str += f'{object_name}, '
        wall_text = gr.Textbox(label='在地面上的物体',value=wall_addition_str[:-2],visible=True,interactive=True)
        floor_text = gr.Textbox(label='在墙上的物体',value=floor_addition_str[:-2],visible=True,interactive=True)



        object_list = object_function_dict.keys()
        region_info_text = f'选取的区域为{REGIONS_tran[region_id.split("_")[1]]}，总共有{len(show_image_path)}张展示图片，包含的物体个数为{len(object_list)}，分别是：'
        for object_ in object_list:
            region_info_text+=object_+'，'
        region_info_text=region_info_text[:-1]+'。'
        check_id_choice = gr.Dropdown(object_list,visible=True,interactive=True)

        return show_image_path,loc_relation_frame,logic_relation_frame,large_class_frame,check_id_choice,wall_text,floor_text,None,gr.update(value=show_image_path[0]),gr.update(value=show_image_path[0]),gr.update(value=show_image_path[0]),gr.update(value=show_image_path[0]),\
               [{}]*6,region_info,region_info_text,None,None,None,None

    def object_function_process(scene_id,region_id):
        '''
            update the object function text
        '''
        if scene_id==None or region_id==None:
            object_function_text = []
            for _ in range(MAX_OBJECT_NUM):
                object_function_text.append(gr.Textbox(visible=False))

            return object_function_text

        region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans.npy',allow_pickle=True)
        object_function_dict = region_info[3]
        object_function_text = []
        for object_name in object_function_dict.keys():
            object_function_text.append(gr.Textbox(label=f'{object_name}在区域的角色',value=object_function_dict[object_name],visible=True,interactive=True))
        for _ in range(MAX_OBJECT_NUM-len(object_function_dict.keys())):
            object_function_text.append(gr.Textbox(visible=False))
        return object_function_text

    def region_feature_process(scene_id,region_id):
        '''
            update the region feature text
        '''
        if scene_id==None or region_id==None:
            region_feature_text = []
            for d_name in FEATURES_tran:
                region_feature_text.append(
                    gr.Textbox(label=f'{d_name}',visible=False))

            return region_feature_text

        region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans.npy',allow_pickle=True)
        region_feature_dict = region_info[5]
        region_feature_text = []
        for d_name in region_feature_dict.keys():
            region_feature_text.append(gr.Textbox(label=f'{FEATURES[d_name]}',value=region_feature_dict[d_name],visible=True,interactive=True))

        return region_feature_text


    def img_change(show_image_index,show_image_path):
        '''
            show the next image while clicking the button.
        '''
        img_change_path = show_image_path[(show_image_index+1)%len(show_image_path)]
        return show_image_index+1, gr.update(value= img_change_path),gr.update(value= img_change_path),gr.update(value= img_change_path)



    def image_match(scene_id,region_id,help_check):
        '''
            show the high-quality photo of an object while choosing the id.
        '''
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

    def loc_info_gen(loc_relation_frame,logic_relation_frame,wall_text,floor_text,region_info,Initial_region_info):

        dataframe_success = True
        # check for loc_relation_frame
        stop_slot = -1
        for _index in range(len(loc_relation_frame['物体A'])):

            if loc_relation_frame['物体A'][_index]=='' and  loc_relation_frame['物体B'][_index]=='' and loc_relation_frame['物体A相对于物体B位置关系'][_index]=='':
                stop_slot =_index
                break

        if stop_slot == -1:
            t_cnt = len(loc_relation_frame['物体A'])
        else:
            t_cnt = stop_slot


        object_dict = list(Initial_region_info[3].keys())
        for _index in range(t_cnt):
            if loc_relation_frame['物体A'][_index] not in object_dict or  loc_relation_frame['物体B'][_index] not in object_dict or loc_relation_frame['物体A相对于物体B位置关系'][_index] not in RELATIONS_tran.keys():
                dataframe_success = False

        # check for logic_relation_frame
        stop_slot = -1
        for _index in range(len(logic_relation_frame['物体A'])):

            if logic_relation_frame['物体A'][_index] == '' and logic_relation_frame['物体B'][_index] == '' and \
                    logic_relation_frame['物体A相对于物体B逻辑关系'][_index] == '':
                stop_slot = _index
                break

        if stop_slot == -1:
            t_cnt2 = len(logic_relation_frame['物体A'])
        else:
            t_cnt2 = stop_slot

        for _index in range(t_cnt2):
            if logic_relation_frame['物体A'][_index] not in object_dict or logic_relation_frame['物体B'][
                _index] not in object_dict:
                dataframe_success = False

        list_success = True

        wall_list_str = '['
        for s in wall_text:
            if s == '<':
                wall_list_str+="'<"
            elif s == '>':
                wall_list_str+=">'"
            else:
                wall_list_str+=s
        wall_list_str+=']'
        floor_list_str = '['
        for s in floor_text:
            if s == '<':
                floor_list_str += "'<"
            elif s == '>':
                floor_list_str += ">'"
            else:
                floor_list_str += s
        floor_list_str += ']'
        try:
            wall_list = eval(wall_list_str)
            floor_list = eval(floor_list_str)
        except:
            gr.Info('输入文本框的格式错误！！')

            return region_info, 'Fail!'

        for t in [*wall_list,*floor_list]:
            if t not in object_dict:
                list_success = False


        if not dataframe_success or not list_success:
            gr.Info('输入表格/文本框的数据错误！！')
            return region_info,'Fail!'
        else:

            for _index in range(t_cnt):
                 A,B,R = loc_relation_frame['物体A'][_index],loc_relation_frame['物体B'][_index],loc_relation_frame['物体A相对于物体B位置关系'][_index]
                 region_info[0][(A,B)] = R
            for _index in range(t_cnt2):
                 A,B,R = logic_relation_frame['物体A'][_index],logic_relation_frame['物体B'][_index],logic_relation_frame['物体A相对于物体B逻辑关系'][_index]
                 region_info[2][(A,B)] = R
            region_info[1]['wall'] = wall_list
            region_info[1]['floor'] = floor_list
            return region_info,str(region_info[0])+'\n'+str(region_info[1])+'\n'+str(region_info[2])

    def loc_reset(Initial_region_info):

        loc_relation_dict = region_info[0]
        logic_relation_dict = region_info[1]
        wall_floor_addition = region_info[2]
        object_function_dict = region_info[3]
        large_class_dict = region_info[4]
        region_feature_dict = region_info[5]

        loc_relation_list = []
        for object_tuple in loc_relation_dict.keys():
            loc_relation_list.append([object_tuple[0], object_tuple[1], RELATIONS[loc_relation_dict[object_tuple]]])
        loc_relation_frame = gr.Dataframe(
            label='物体间位置关系表格',
            headers=["物体A", "物体B", "物体A相对于物体B位置关系"],
            datatype=["str", "str", "str"],
            row_count=len(loc_relation_dict.keys()),
            col_count=(3, "fixed"),
            value=loc_relation_list,
            visible=True,
            interactive=True
        )
        logic_relation_list = []
        for object_tuple in logic_relation_dict.keys():
            logic_relation_list.append([object_tuple[0], object_tuple[1], logic_relation_dict[object_tuple]])
        logic_relation_frame = gr.Dataframe(
            label='物体间逻辑关系表格',
            headers=["物体A", "物体B", "物体A相对于物体B逻辑关系"],
            datatype=["str", "str", "str"],
            row_count=len(logic_relation_dict.keys()),
            col_count=(3, "fixed"),
            value=logic_relation_list,
            visible=True,
            interactive=True
        )

        wall_addition_str = floor_addition_str = ''
        for object_name in wall_floor_addition['wall']:
            wall_addition_str += f'{object_name}, '
        for object_name in wall_floor_addition['floor']:
            floor_addition_str += f'{object_name}, '
        wall_text = gr.Textbox(label='在地面上的物体', value=wall_addition_str[:-2], visible=True, interactive=True)
        floor_text = gr.Textbox(label='在墙上的物体', value=floor_addition_str[:-2], visible=True, interactive=True)
        return loc_relation_frame,logic_relation_frame,wall_text,floor_text

    def obj_info_gen(*object_set):


        output_dict = {}
        for _index in range(len(object_set[-1][3].keys())):
            output_dict[list(object_set[-1][3].keys())[_index]] = object_set[_index]
        region_info = object_set[-2]
        region_info[3] = output_dict

        return region_info,str(output_dict)
    def obj_reset(Initial_region_info):
        object_function_dict = Initial_region_info[3]
        object_function_text = []
        for object_name in object_function_dict.keys():
            object_function_text.append(
                gr.Textbox(label=f'{object_name}的作用', value=object_function_dict[object_name], visible=True,
                           interactive=True))
        for _ in range(MAX_OBJECT_NUM - len(object_function_dict.keys())):
            object_function_text.append(gr.Textbox(visible=False))
        return object_function_text


    def large_class_info_gen(large_class_frame,region_info,Initial_region_info):

        object_dict = list(Initial_region_info[3].keys())
        stop_slot = -1
        dataframe_success = True
        for _index in range(len(large_class_frame['物体列表'])):

            if large_class_frame['物体列表'][_index] == '' and large_class_frame['描述'][_index] == '' :
                stop_slot = _index
                break

        if stop_slot == -1:
            t_cnt = len(large_class_frame['物体A'])
        else:
            t_cnt = stop_slot

        for _index in range(t_cnt):
            object_list = large_class_frame['物体列表'][_index].split('/')
            for object_ in object_list:
                if object_ not in object_dict:
                    dataframe_success = False

        if not dataframe_success:
            gr.Info('输入表格/文本框的数据错误！！')
            return region_info,'Fail!'
        else:
            for _index in range(t_cnt):
                object_tuple = tuple(large_class_frame['物体列表'][_index].split('/'))
                region_info[4][object_tuple] = large_class_frame['描述'][_index]

        return region_info,str(region_info[4])

    def large_class_reset(Initial_region_info):

        large_class_dict = Initial_region_info[4]



        large_class_list = []
        for object_tuple in large_class_dict.keys():
            object_text = ''
            for t in object_tuple:
                object_text += (t + '/')
            large_class_list.append([object_text[:-1], large_class_dict[object_tuple]])
        large_class_frame = gr.Dataframe(
            label='物体大类/功能组表格',
            headers=["物体列表", "描述"],
            datatype=["str", "str"],
            row_count=len(large_class_dict.keys()),
            col_count=(2, "fixed"),
            value=large_class_list,
            visible=True,
            interactive=True
        )
        return large_class_frame



    def reg_info_gen(*region_set):

        output_dict = {}
        for _index in range(len(FEATURES)):
            output_dict[list(FEATURES.keys())[_index]] = region_set[_index]
        region_info = region_set[-1]
        region_info[5] = output_dict

        return region_info,str(output_dict)

    def reg_reset(Initial_region_info):


        region_feature_dict = Initial_region_info[5]
        region_feature_text = []
        for d_name in region_feature_dict.keys():
            region_feature_text.append(
                gr.Textbox(label=f'{FEATURES[d_name]}', value=region_feature_dict[d_name], visible=True,
                           interactive=True))

        return region_feature_text

    def annotation_save(scene_id,region_id,region_info):
        np.save(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_anno.npy', region_info)
        return None,None



    show_image_index = gr.State(0)
    show_image_path = gr.State([])
    region_info = gr.State([{},{},{},{}])
    Initial_region_info = gr.State([])

    scene_choice = gr.Dropdown(scene_list, label="对应场景")
    region_choice = gr.Dropdown(region_list, label="对应区域")
    region_info_text = gr.Textbox(label="区域信息",value="请选择区域")
    with gr.Row():
        show_image = gr.Image(label='区域场景展示')
        loc_relation_info = gr.Textbox(label='物体间位置关系',interactive=False)



    # location relationship
    loc_relation_frame = gr.DataFrame(visible=False)
    # wall/floor addition
    wall_text = gr.Textbox(visible=False)
    floor_text = gr.Textbox(visible=False)
    # logic_relationship
    logic_relation_frame = gr.DataFrame(visible=False)

    loc_relation_gen_button = gr.Button("生成标注(每次更改完必点)")
    loc_relation_reset_button = gr.Button("重置标注")

    with gr.Row():
        show_image_copy1 = gr.Image(label='区域场景展示')
        object_function_info = gr.Textbox(label='物体角色',interactive=False)


    # object functions
    object_function_text = []
    for _ in range(MAX_OBJECT_NUM):
        object_function_text.append(gr.Textbox(visible=False))

    object_function_gen_button = gr.Button("生成标注(每次更改完必点)")
    object_function_reset_button = gr.Button("重置标注")


    with gr.Row():
        show_image_copy2 = gr.Image(label='区域场景展示')
        large_class_info = gr.Textbox(label='大类/功能组',interactive=False)

    # large class
    large_class_frame = gr.DataFrame(visible=False)
    large_class_gen_button = gr.Button("生成标注(每次更改完必点)")
    large_class_reset_button = gr.Button("重置标注")


    with gr.Row():
        show_image_copy3 = gr.Image(label='区域场景展示')
        region_feature_info = gr.Textbox(label='区域特点',interactive=False)


    # region features
    region_feature_text = []
    for d_name in FEATURES_tran.keys():
        region_feature_text.append(gr.Textbox(label=d_name,value=FEATURES_tran[d_name],visible=False))
    region_feature_gen_button = gr.Button("生成标注(每次更改完必点)")
    region_feature_reset_button = gr.Button("重置标注")

    Save_button = gr.Button("保存标注",visible=True)
    check_id_choice = gr.Dropdown([], visible=False)
    check_id_image = gr.Image()



    region_choice.change(fn=data_process, inputs=[scene_choice,region_choice],outputs=[show_image_path,loc_relation_frame,logic_relation_frame,large_class_frame,check_id_choice,wall_text,floor_text,check_id_image,
                                                                                       show_image,show_image_copy1,show_image_copy2,show_image_copy3,region_info,Initial_region_info,region_info_text,
                                                                                       loc_relation_info,object_function_info,large_class_info,region_feature_info])
    region_choice.change(fn=object_function_process, inputs=[scene_choice,region_choice], outputs=object_function_text)
    region_choice.change(fn=region_feature_process, inputs=[scene_choice, region_choice], outputs=region_feature_text)

    show_image.select(fn=img_change,inputs=[show_image_index,show_image_path],outputs=[show_image_index,show_image,show_image_copy1,show_image_copy2,show_image_copy3])
    show_image_copy1.select(fn=img_change,inputs=[show_image_index,show_image_path],outputs=[show_image_index,show_image,show_image_copy1,show_image_copy2,show_image_copy3])
    show_image_copy2.select(fn=img_change, inputs=[show_image_index, show_image_path], outputs=[show_image_index, show_image,show_image_copy1,show_image_copy2,show_image_copy3])
    show_image_copy3.select(fn=img_change, inputs=[show_image_index, show_image_path],outputs=[show_image_index, show_image, show_image_copy1, show_image_copy2,show_image_copy3])

    loc_relation_gen_button.click(fn=loc_info_gen,inputs=[loc_relation_frame,logic_relation_frame,wall_text,floor_text,region_info,Initial_region_info],outputs=[region_info,loc_relation_info])
    loc_relation_reset_button.click(fn=loc_reset,
                                  inputs=[Initial_region_info],
                                  outputs=[loc_relation_frame,logic_relation_frame, wall_text, floor_text])

    object_function_gen_button.click(fn=obj_info_gen,inputs=[*object_function_text,region_info,Initial_region_info],outputs=[region_info,object_function_info])
    object_function_reset_button.click(fn=obj_reset,inputs=[Initial_region_info],outputs=object_function_text)

    large_class_gen_button.click(fn=large_class_info_gen,inputs=[large_class_frame,region_info,Initial_region_info], outputs=[region_info, large_class_info])
    large_class_reset_button.click(fn=large_class_reset,inputs=[Initial_region_info],outputs=[large_class_frame])

    region_feature_gen_button.click(fn=reg_info_gen,inputs=[*region_feature_text,region_info],outputs=[region_info,region_feature_info])
    region_feature_reset_button.click(fn=reg_reset,inputs=[Initial_region_info],outputs=region_feature_text)


    check_id_choice.select(fn=image_match, inputs=[scene_choice,region_choice,check_id_choice],outputs=[check_id_image])

    Save_button.click(fn=annotation_save, inputs=[scene_choice,region_choice,region_info],outputs=[scene_choice,region_choice])



demo.queue(concurrency_count=20)

if __name__ == "__main__":

    demo.launch(show_error=True)

