# -*- coding:utf-8 –*-
import os
import re
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
            'on':'在上面（有支撑）',
            'above':'在上方（没支撑）',
            'under':'在下方',
            'closed by':'紧贴着',
            'next to':'相邻着（不紧贴）',
            'hanging/lying on':'悬挂在/倚靠在',}
RELATIONS_tran = {RELATIONS[key]: key for key in RELATIONS.keys()}
def relation_cn(text):
    return RELATIONS.get(text, text)
def relation_en(text):
    return RELATIONS_tran.get(text, text)

# 区域特点映射
FEATURES = {'location and function description':'区域定位和功能描述',
            'space layout and dimensions':'空间布局与尺寸',
            'doors,windows, walls, floors and ceilings':'门窗, 墙面,地面与天花板',
            'soft fitting elements and decorative details':'软装元素与装饰细节',
            'lighting design':'照明设计',
            'color matching and style theme':'色彩搭配与风格主题',
            'special features':'特殊化特点'}

FEATURES_tran = {FEATURES[key]: key for key in FEATURES.keys()}

# 物品类型映射
from utils.utils_read import load_json
OBJECTS = load_json('object_type_translation.json')
OBJECTS = {k.lower(): v for k, v in OBJECTS.items()}
OBJECTS_tran = {OBJECTS[key]: key for key in OBJECTS.keys()}

def translate_object_type(object_name:str):
    str_to_check = re.sub(r'[\d<>_]', '', object_name.lower())
    return OBJECTS.get(str_to_check, object_name)

region_view_dir_name = 'region_views'
def _get_regions(scene_id):
    regions = os.listdir(f'data/{scene_id}/{region_view_dir_name}')
    regions = [region for region in regions if region.endswith('region') and os.path.exists(f'data/{scene_id}/{region_view_dir_name}/{region}/struction_trans.npy')]
    regions.sort()
    return regions

SCENE_LIST = os.listdir('data')
SCENE_LIST = [name for name in SCENE_LIST if name.startswith('scene') or name.startswith('1mp3d') or name.startswith('3rscan')]
SCENE_LIST = [name for name in SCENE_LIST if os.path.isdir(os.path.join('data',name, region_view_dir_name))]
SCENE_LIST = [name for name in SCENE_LIST if len(_get_regions(name))>0]
SCENE_LIST.sort()




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
    ###############################################################################################3
    ## functions definition here    

    def check_user_name_validity(user_name):
        if len(user_name) == 0 or ' ' in user_name or not user_name[0].isalpha():
            gr.Warning("用户名不合法。请首位必须为字母，并不要带空格。请重新输入。")
            return False
        return True


    def lock_user_name(user_name, user_name_locked):
        if check_user_name_validity(user_name):
            user_name = user_name.strip()
            user_name = gr.Textbox(label="用户名", value=user_name, interactive=False)
            user_name_locked = True
        return user_name, user_name_locked


    def get_region_options(scene_id):
        regions = _get_regions(scene_id)
        if len(regions) == 0:
            gr.Info(f"场景{scene_id}暂时没有区域信息，未来会补充。请切换到其他场景。")
        return gr.Dropdown(regions, label="需要修改的区域")

    def data_process(scene_id,region_id,user_name):
        if scene_id==None or region_id==None:
            check_id_choice = gr.Dropdown([], visible=False)
            loc_relation_frame = gr.Dataframe(visible=False)
            logic_relation_frame = gr.Dataframe(visible=False)
            large_class_frame = gr.Dataframe(visible=False)
            wall_text = gr.Textbox(visible=False)
            floor_text = gr.Textbox(visible=False)

            return [],loc_relation_frame,logic_relation_frame,large_class_frame,check_id_choice,wall_text,floor_text,None,None,None,None,None,[{}]*6,[{}]*6,'请选择区域',None,None,None,None,None

        show_image_path =[f'data/{scene_id}/{region_view_dir_name}/{region_id}/'+img_file for img_file in os.listdir(f'data/{scene_id}/{region_view_dir_name}/{region_id}') if img_file[-4:]=='.jpg']
        if os.path.exists(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy'):

            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy',allow_pickle=True)
            gr.Info(f"检测到标注已经修改！已加载修改后的信息")
        else:
            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans.npy',allow_pickle=True)
        loc_relation_dict = region_info[0]
        logic_relation_dict = region_info[1]
        wall_floor_addition = region_info[2]
        object_function_dict = region_info[3]
        large_class_dict = region_info[4]
        region_feature_dict = region_info[5]


        loc_relation_list = []

        for object_tuple in loc_relation_dict.keys():
            if loc_relation_dict[object_tuple] in RELATIONS.keys():
                loc_relation_list.append([object_tuple[0], translate_object_type(object_tuple[0]), object_tuple[1], translate_object_type(object_tuple[1]), relation_cn(loc_relation_dict[object_tuple])])
            else:
                loc_relation_list.append([object_tuple[0], translate_object_type(object_tuple[0]), object_tuple[1],
                                          translate_object_type(object_tuple[1]),
                                          loc_relation_dict[object_tuple]])
        loc_relation_list.append(['','','','',''])

        loc_relation_frame = gr.Dataframe(
            label='物体间位置关系表格',
            headers=["物体A", "翻译A", "物体B", "翻译B", "物体A相对于物体B位置关系"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=len(loc_relation_dict.keys()),
            col_count=(5, "fixed"),
            value=loc_relation_list,
            visible=True,
            interactive=True
        )
        logic_relation_list = []
        for object_tuple in logic_relation_dict.keys():
            logic_relation_list.append([object_tuple[0], translate_object_type(object_tuple[0]), object_tuple[1], translate_object_type(object_tuple[1]), logic_relation_dict[object_tuple]])
        logic_relation_list.append(['','','','',''])
        logic_relation_frame = gr.Dataframe(
            label='物体间逻辑关系表格',
            headers=["物体A", "翻译A", "物体B", "翻译B", "物体A相对于物体B逻辑关系"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=len(logic_relation_dict.keys()),
            col_count=(5, "fixed"),
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
        large_class_list.append(['',''])

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
        wall_text = gr.Textbox(label='在墙上的物体',value=wall_addition_str[:-2],visible=True,interactive=True)
        floor_text = gr.Textbox(label='在地面上的物体',value=floor_addition_str[:-2],visible=True,interactive=True)



        object_list = object_function_dict.keys()
        region_info_text = f'选取的区域为{REGIONS_tran[region_id.split("_")[1]]}，总共有{len(show_image_path)}张展示图片，包含的物体个数为{len(object_list)}，分别是：'
        for object_ in object_list:
            region_info_text+=object_+translate_object_type(object_)+'，'
        region_info_text=region_info_text[:-1]+'。'
        check_id_choice = gr.Dropdown(object_list,visible=True,interactive=True)

        return show_image_path,loc_relation_frame,logic_relation_frame,large_class_frame,check_id_choice,wall_text,floor_text,None,show_image_path[0],show_image_path[0],show_image_path[0],show_image_path[0],\
               [{}]*6,region_info,region_info_text,None,None,None,None,None

    def object_function_process(scene_id,region_id,user_name):
        '''
            update the object function text
        '''
        if scene_id==None or region_id==None:
            object_function_text = []
            for _ in range(MAX_OBJECT_NUM):
                object_function_text.append(gr.Textbox(visible=False))

            return object_function_text

        if os.path.exists(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy'):
            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy',allow_pickle=True)

        else:
            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans.npy',allow_pickle=True)
        object_function_dict = region_info[3]
        object_function_text = []
        for object_name in object_function_dict.keys():
            object_function_text.append(gr.Textbox(label=f'{object_name}在区域的角色',value=object_function_dict[object_name],visible=True,interactive=True))
        for _ in range(MAX_OBJECT_NUM-len(object_function_dict.keys())):
            object_function_text.append(gr.Textbox(visible=False))
        return object_function_text

    def region_feature_process(scene_id,region_id,user_name):
        '''
            update the region feature text
        '''
        if scene_id==None or region_id==None:
            region_feature_text = []
            for d_name in FEATURES_tran:
                region_feature_text.append(
                    gr.Textbox(label=f'{d_name}',visible=False))

            return region_feature_text

        if os.path.exists(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy'):

            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy',
                                  allow_pickle=True)

        else:
            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans.npy',
                                  allow_pickle=True)

        region_feature_dict = region_info[5]
        region_feature_text = []
        for d_name in region_feature_dict.keys():
            region_feature_text.append(gr.Textbox(label=f'{FEATURES[d_name]}',value=region_feature_dict[d_name],visible=True,interactive=True))

        return region_feature_text
    def Q_A_process(scene_id,region_id,user_name):
        '''
            update the region feature text
        '''
        if scene_id==None or region_id==None:


            Space_Q1 = gr.Textbox(label='有关空间的问题1', visible=False)
            Space_A1 = gr.Textbox(label='有关空间的回答1', visible=False)

            Space_Q2 = gr.Textbox(label='有关空间的问题2', visible=False)
            Space_A2 = gr.Textbox(label='有关空间的回答2', visible=False)

            Explain_Q1 = gr.Textbox(label='有关解释的问题1', visible=False)
            Explain_A1 = gr.Textbox(label='有关解释的回答1', visible=False)

            Explain_Q2 = gr.Textbox(label='有关解释的问题2', visible=False)
            Explain_A2 = gr.Textbox(label='有关解释的回答2', visible=False)

            Situate_Q1 = gr.Textbox(label='有关情境的问题1', visible=False)
            Situate_A1 = gr.Textbox(label='有关情境的回答1', visible=False)

            Situate_Q2 = gr.Textbox(label='有关情境的问题2', visible=False)
            Situate_A2 = gr.Textbox(label='有关情境的回答2', visible=False)


            return Space_Q1,Space_A1,Space_Q2,Space_A2,Explain_Q1,Explain_A1,Explain_Q2,Explain_A2,Situate_Q1,Situate_A1,Situate_Q2,Situate_A2

        if os.path.exists(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy'):

            region_info = np.load(f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy',
                                  allow_pickle=True)

            Space_Q1 = gr.Textbox(label='有关空间的问题1',value=region_info[-1]['Space_Q1'], visible=True, interactive=True)
            Space_A1 = gr.Textbox(label='有关空间的回答1',value=region_info[-1]['Space_A1'], visible=True, interactive=True)

            Space_Q2 = gr.Textbox(label='有关空间的问题2',value=region_info[-1]['Space_Q2'], visible=True, interactive=True)
            Space_A2 = gr.Textbox(label='有关空间的回答2',value=region_info[-1]['Space_A2'], visible=True, interactive=True)

            Explain_Q1 = gr.Textbox(label='有关解释的问题1',value=region_info[-1]['Explain_Q1'], visible=True, interactive=True)
            Explain_A1 = gr.Textbox(label='有关解释的回答1',value=region_info[-1]['Explain_A1'], visible=True, interactive=True)

            Explain_Q2 = gr.Textbox(label='有关解释的问题2',value=region_info[-1]['Explain_Q2'], visible=True, interactive=True)
            Explain_A2 = gr.Textbox(label='有关解释的回答2',value=region_info[-1]['Explain_A2'], visible=True, interactive=True)

            Situate_Q1 = gr.Textbox(label='有关情境的问题1',value=region_info[-1]['Situate_Q1'], visible=True, interactive=True)
            Situate_A1 = gr.Textbox(label='有关情境的回答1',value=region_info[-1]['Situate_A1'], visible=True, interactive=True)

            Situate_Q2 = gr.Textbox(label='有关情境的问题2',value=region_info[-1]['Situate_Q2'], visible=True, interactive=True)
            Situate_A2 = gr.Textbox(label='有关情境的回答2',value=region_info[-1]['Situate_A2'], visible=True, interactive=True)

            return Space_Q1, Space_A1, Space_Q2, Space_A2, Explain_Q1, Explain_A1, Explain_Q2, Explain_A2, Situate_Q1, Situate_A1, Situate_Q2, Situate_A2


        else:
            Space_Q1 = gr.Textbox(label='有关空间的问题1', visible=True,interactive=True)
            Space_A1 = gr.Textbox(label='有关空间的回答1', visible=True,interactive=True)

            Space_Q2 = gr.Textbox(label='有关空间的问题2', visible=True,interactive=True)
            Space_A2 = gr.Textbox(label='有关空间的回答2', visible=True,interactive=True)

            Explain_Q1 = gr.Textbox(label='有关解释的问题1', visible=True,interactive=True)
            Explain_A1 = gr.Textbox(label='有关解释的回答1', visible=True,interactive=True)

            Explain_Q2 = gr.Textbox(label='有关解释的问题2', visible=True,interactive=True)
            Explain_A2 = gr.Textbox(label='有关解释的回答2', visible=True,interactive=True)

            Situate_Q1 = gr.Textbox(label='有关情境的问题1', visible=True,interactive=True)
            Situate_A1 = gr.Textbox(label='有关情境的回答1', visible=True,interactive=True)

            Situate_Q2 = gr.Textbox(label='有关情境的问题2', visible=True,interactive=True)
            Situate_A2 = gr.Textbox(label='有关情境的回答2', visible=True,interactive=True)

            return Space_Q1,Space_A1,Space_Q2,Space_A2,Explain_Q1,Explain_A1,Explain_Q2,Explain_A2,Situate_Q1,Situate_A1,Situate_Q2,Situate_A2

    def Q_A_gen(Space_Q1, Space_A1, Space_Q2, Space_A2, Explain_Q1, Explain_A1, Explain_Q2, Explain_A2, Situate_Q1, Situate_A1, Situate_Q2, Situate_A2,region_info):
        Q_A_dict = {}
        Q_A_dict['Space_Q1'] = Space_Q1
        Q_A_dict['Space_A1'] = Space_A1
        Q_A_dict['Space_Q2'] = Space_Q2
        Q_A_dict['Space_A2'] = Space_A2

        Q_A_dict['Explain_Q1'] = Explain_Q1
        Q_A_dict['Explain_A1'] = Explain_A1
        Q_A_dict['Explain_Q2'] = Explain_Q2
        Q_A_dict['Explain_A2'] = Explain_A2

        Q_A_dict['Situate_Q1'] = Situate_Q1
        Q_A_dict['Situate_A1'] = Situate_A1
        Q_A_dict['Situate_Q2'] = Situate_Q2
        Q_A_dict['Situate_A2'] = Situate_A2
        for key_ in Q_A_dict:
            if 'Q' in key_:
                if (Q_A_dict[key_]=='' and Q_A_dict[key_.replace('Q','A')]!='') or (Q_A_dict[key_]!=''  and Q_A_dict[key_.replace('Q','A')]==''):
                    gr.Info('问题/答案不能只有一个不为空')
                    return region_info,'Fail'

        if len(region_info)>6:
            region_info[6]=Q_A_dict
        else:
            region_info.append(Q_A_dict)

        return region_info,Q_A_dict



    def img_change(show_image_index,show_image_path):
        '''
            show the next image while clicking the button.
            return 4 copies of the image path.
        '''
        img_change_path = show_image_path[(show_image_index+1)%len(show_image_path)]
        return show_image_index+1, img_change_path, img_change_path,img_change_path, img_change_path



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
            return target
        else:
            return None

    def loc_info_gen(loc_relation_frame,logic_relation_frame,wall_text,floor_text,region_info,Initial_region_info):

        dataframe_success = True
        object_dict = list(Initial_region_info[3].keys())
        # check for loc_relation_frame

        for _index in range(len(loc_relation_frame['物体A'])):

            if not(loc_relation_frame['物体A'][_index]=='' and  loc_relation_frame['物体B'][_index]=='' and loc_relation_frame['物体A相对于物体B位置关系'][_index]==''):
                if loc_relation_frame['物体A'][_index] not in object_dict or loc_relation_frame['物体B'][
                    _index] not in object_dict or loc_relation_frame['物体A相对于物体B位置关系'][
                    _index] not in RELATIONS_tran.keys():
                    dataframe_success = False

        for _index in range(len(logic_relation_frame['物体A'])):

            if not(logic_relation_frame['物体A'][_index] == '' and logic_relation_frame['物体B'][_index] == '' and \
                    logic_relation_frame['物体A相对于物体B逻辑关系'][_index] == ''):
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
            loc_dict = {}
            log_dict = {}
            wf_dict = {}
            for _index in range(len(loc_relation_frame['物体A'])):

                if not (loc_relation_frame['物体A'][_index] == '' and loc_relation_frame['物体B'][_index] == '' and
                        loc_relation_frame['物体A相对于物体B位置关系'][_index] == ''):
                    A, B, R = loc_relation_frame['物体A'][_index], loc_relation_frame['物体B'][_index], \
                              loc_relation_frame['物体A相对于物体B位置关系'][_index]

                    loc_dict[(A, B)] = R

            for _index in range(len(logic_relation_frame['物体A'])):

                if not (logic_relation_frame['物体A'][_index] == '' and logic_relation_frame['物体B'][_index] == '' and
                        logic_relation_frame['物体A相对于物体B逻辑关系'][_index] == ''):
                    A, B, R = logic_relation_frame['物体A'][_index], logic_relation_frame['物体B'][_index], \
                              logic_relation_frame['物体A相对于物体B逻辑关系'][_index]
                    log_dict[(A, B)] = R

            wf_dict['wall'] = wall_list
            wf_dict['floor'] = floor_list
            region_info[0] = loc_dict
            region_info[2] = wf_dict
            region_info[1] = log_dict


            return region_info,str(region_info[0])+'\n'+str(region_info[1])+'\n'+str(region_info[2])

    def loc_reset(Initial_region_info):

        loc_relation_dict = Initial_region_info[0]
        logic_relation_dict = Initial_region_info[1]
        wall_floor_addition = Initial_region_info[2]

        loc_relation_list = []

        for object_tuple in loc_relation_dict.keys():
            if loc_relation_dict[object_tuple] in RELATIONS.keys():
                loc_relation_list.append([object_tuple[0], translate_object_type(object_tuple[0]), object_tuple[1],
                                          translate_object_type(object_tuple[1]),
                                          relation_cn(loc_relation_dict[object_tuple])])
            else:
                loc_relation_list.append([object_tuple[0], translate_object_type(object_tuple[0]), object_tuple[1],
                                          translate_object_type(object_tuple[1]),
                                          loc_relation_dict[object_tuple]])
        loc_relation_list.append(['', '', '', '', ''])

        loc_relation_frame = gr.Dataframe(
            label='物体间位置关系表格',
            headers=["物体A", "翻译A", "物体B", "翻译B", "物体A相对于物体B位置关系"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=len(loc_relation_dict.keys()),
            col_count=(5, "fixed"),
            value=loc_relation_list,
            visible=True,
            interactive=True
        )
        logic_relation_list = []
        for object_tuple in logic_relation_dict.keys():
            logic_relation_list.append([object_tuple[0], translate_object_type(object_tuple[0]), object_tuple[1],
                                        translate_object_type(object_tuple[1]), logic_relation_dict[object_tuple]])
        loc_relation_list.append(['', '', '', '', ''])
        logic_relation_frame = gr.Dataframe(
            label='物体间逻辑关系表格',
            headers=["物体A", "翻译A", "物体B", "翻译B", "物体A相对于物体B逻辑关系"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=len(logic_relation_dict.keys()),
            col_count=(5, "fixed"),
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

        dataframe_success = True
        for _index in range(len(large_class_frame['物体列表'])):

            if not(large_class_frame['物体列表'][_index] == '' and large_class_frame['描述'][_index] == '' ):
                object_list = large_class_frame['物体列表'][_index].split('/')
                for object_ in object_list:
                    if object_ not in object_dict:
                        dataframe_success = False





        if not dataframe_success:
            gr.Info('输入表格/文本框的数据错误！！')
            return region_info,'Fail!'
        else:
            for _index in range(len(large_class_frame['物体列表'])):

                if not (large_class_frame['物体列表'][_index] == '' and large_class_frame['描述'][_index] == ''):
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
        large_class_list.append(['',''])
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
        #print('regsave:',region_info)

        return region_info,str(output_dict)

    def reg_reset(Initial_region_info):
        region_feature_dict = Initial_region_info[5]
        region_feature_text = []
        for d_name in region_feature_dict.keys():
            region_feature_text.append(
                gr.Textbox(label=f'{FEATURES[d_name]}', value=region_feature_dict[d_name], visible=True,
                           interactive=True))

        return region_feature_text

    def annotation_save(scene_id,region_id,region_info,user_name,loc_relation_info,object_function_info,large_class_info,region_feature_info):
        texts =[loc_relation_info,object_function_info,large_class_info,region_feature_info]

        for t in texts:
            if t==None or t=='' or t=='Fail':
                gr.Info(f'有没生成标注或生成失败的部分!')
                return scene_id,region_id
        save_path = f'data/{scene_id}/{region_view_dir_name}/{region_id}/struction_trans_{user_name}.npy'
        np.save(save_path, region_info)
        gr.Info(f'标注已经保存到{save_path}!')
        return scene_id,None

    ###############################################################################################3
    ## UI definition here    

    show_image_index = gr.State(0)
    show_image_path = gr.State([])
    region_info = gr.State([{},{},{},{},{},{}])
    Initial_region_info = gr.State([])
    user_name_locked = gr.State(False)

    with gr.Row():
        user_name = gr.Textbox(label="用户名", value="", placeholder="在此输入用户名，首位必须为字母，不要带空格。")
        # view_only = gr.Checkbox(label="只读模式", value=False)
        confirm_user_name_btn = gr.Button(value="确认并锁定用户名（刷新网页才能重置用户名）", label="确认用户名")
        # check_annotated_btn = gr.Button(value="查看未标注场景数（目前还没做）", label="查看未标注")

    with gr.Row():
        scene_choice = gr.Dropdown(SCENE_LIST, label="需要修改的场景")
        region_choice = gr.Dropdown(['测试区域'], label="需要修改的区域")
    region_info_text = gr.Textbox(label="区域信息",value="请选择区域")
    ###########################
    # object relation related
    ###########################
    with gr.Row():
        show_image = gr.Image(label='区域场景展示')
        with gr.Column():
            loc_relation_frame = gr.DataFrame(visible=False)
            logic_relation_frame = gr.DataFrame(visible=False)
    with gr.Row():
        wall_text = gr.Textbox(visible=False)
        floor_text = gr.Textbox(visible=False)

    loc_relation_info = gr.Textbox(label='物体间位置关系',interactive=False)
    loc_relation_gen_button = gr.Button("生成位置和逻辑关系标注(每次更改完必点)")
    loc_relation_reset_button = gr.Button("重置位置和逻辑关系标注")

    ###########################
    # object function related
    ###########################
    with gr.Row():
        show_image_copy1 = gr.Image(label='区域场景展示')
        with gr.Column():
            object_function_text = []
            for _ in range(MAX_OBJECT_NUM):
                object_function_text.append(gr.Textbox(visible=False))

    object_function_info = gr.Textbox(label='物体角色',interactive=False)
    object_function_gen_button = gr.Button("生成物体功能标注(每次更改完必点)")
    object_function_reset_button = gr.Button("重置物体功能标注")

    ###########################
    # object group (large class) related
    ###########################
    with gr.Row():
        show_image_copy2 = gr.Image(label='区域场景展示')
        large_class_frame = gr.DataFrame(visible=False)
    large_class_info = gr.Textbox(label='大类/功能组',interactive=False)
    large_class_gen_button = gr.Button("生成大类标注(每次更改完必点)")
    large_class_reset_button = gr.Button("重置大类标注")

    ###########################
    # region features related
    ###########################
    with gr.Row():
        show_image_copy3 = gr.Image(label='区域场景展示')
        with gr.Column():
            region_feature_text = []
            for d_name in FEATURES_tran.keys():
                region_feature_text.append(gr.Textbox(label=d_name,value=FEATURES_tran[d_name],visible=False))

    region_feature_info = gr.Textbox(label='区域特点',interactive=False)
    region_feature_gen_button = gr.Button("生成区域特征标注(每次更改完必点)")
    region_feature_reset_button = gr.Button("重置标注")


    # 空间QA
    with gr.Row():
        with gr.Column():
            Space_Q1 = gr.Textbox(label='有关空间的问题1',visible=False)
            Space_A1 = gr.Textbox(label='有关空间的回答1',visible=False)
        with gr.Column():
            Space_Q2 = gr.Textbox(label='有关空间的问题2', visible=False)
            Space_A2 = gr.Textbox(label='有关空间的回答2', visible=False)
    # 解释QA
    with gr.Row():
        with gr.Column():
            Explain_Q1 = gr.Textbox(label='有关解释的问题1', visible=False)
            Explain_A1 = gr.Textbox(label='有关解释的回答1', visible=False)
        with gr.Column():
            Explain_Q2 = gr.Textbox(label='有关解释的问题2', visible=False)
            Explain_A2 = gr.Textbox(label='有关解释的回答2', visible=False)
    # 情境QA
    with gr.Row():
        with gr.Column():
            Situate_Q1 = gr.Textbox(label='有关情境的问题1', visible=False)
            Situate_A1 = gr.Textbox(label='有关情境的回答1', visible=False)
        with gr.Column():
            Situate_Q2 = gr.Textbox(label='有关情境的问题2', visible=False)
            Situate_A2 = gr.Textbox(label='有关情境的回答2', visible=False)
    Q_and_A_info = gr.Textbox(interactive=False)
    Q_and_A_gen_button = gr.Button("生成问题回答标注(每次更改完必点)")


    Save_button = gr.Button("保存标注",visible=True)
    check_id_choice = gr.Dropdown([], visible=False)
    check_id_image = gr.Image()

    ###############################################################################################3
    ## function connection here    

    confirm_user_name_btn.click(lock_user_name, inputs=[user_name, user_name_locked],
                                outputs=[user_name, user_name_locked])
    scene_choice.select(fn=get_region_options, inputs=[scene_choice], outputs=[region_choice])
    scene_choice.change(fn=get_region_options, inputs=[scene_choice], outputs=[region_choice])
    region_choice.change(fn=data_process, inputs=[scene_choice,region_choice,user_name],outputs=[show_image_path,loc_relation_frame,logic_relation_frame,large_class_frame,check_id_choice,wall_text,floor_text,check_id_image,
                                                                                       show_image,show_image_copy1,show_image_copy2,show_image_copy3,region_info,Initial_region_info,region_info_text,
                                                                                       loc_relation_info,object_function_info,large_class_info,region_feature_info,Q_and_A_info])
    region_choice.change(fn=object_function_process, inputs=[scene_choice,region_choice,user_name], outputs=object_function_text)
    region_choice.change(fn=region_feature_process, inputs=[scene_choice, region_choice,user_name], outputs=region_feature_text)
    region_choice.change(fn=Q_A_process,inputs=[scene_choice, region_choice,user_name], outputs=[Space_Q1,Space_A1,Space_Q2,Space_A2,Explain_Q1,Explain_A1,Explain_Q2,Explain_A2,Situate_Q1,Situate_A1,Situate_Q2,Situate_A2])

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

    Q_and_A_gen_button.click(fn=Q_A_gen, inputs=[Space_Q1,Space_A1,Space_Q2,Space_A2,Explain_Q1,Explain_A1,Explain_Q2,Explain_A2,Situate_Q1,Situate_A1,Situate_Q2,Situate_A2,region_info],
                                 outputs=[region_info, Q_and_A_info])




    check_id_choice.select(fn=image_match, inputs=[scene_choice,region_choice,check_id_choice],outputs=[check_id_image])

    Save_button.click(fn=annotation_save, inputs=[scene_choice,region_choice,region_info,user_name,loc_relation_info,object_function_info,large_class_info,region_feature_info],outputs=[scene_choice,region_choice])



demo.queue(concurrency_count=20)

if __name__ == "__main__":

    demo.launch(show_error=True, server_port=7850)

