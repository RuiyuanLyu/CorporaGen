####################  使用方式  #######################################
# 目的是给定当前俯视图，通过在图上选点框定图上的功能区域
# 单个scene的标注流程
# 1. 点击"Input file”框，根据文件名选择需要标注的图片
# 2. 不断重复以下流程，直到标完所有功能区
#   a. 拖动“Vertex Number”滑块，选择多边形的顶点个数
#   b. 在”Image”框中点Vertex Number个点，即选中了多边形
#   c. 在”Lable”框中选择label，即该区域提供的功能
#   d. 点击“Annotate”，完成一个多边形的标注
#     i. “Selected Polygon”和“Label”会被自动清空
#     ii. 如果顶点的数量不匹配，或者没有label则会清空当前多边形
# 3. 点击“Save to file”，整张图片的标注数据会被保存
# 其他说明
# 1. “Clear”会清空对于当前图片*所有*的标注，谨慎点击
# 2. 当前标注会展现在“Annotation History”中
######################################################################
#-*- coding:utf-8 –*-
import os

import cv2
import gradio as gr  # gradio==3.44.0
import numpy as np
from math import ceil
import json
import open3d as o3d
from utils_read import read_annotation_pickle
global file_name
global click_evt_list
global vertex_list
global annotation_list

global poly_done
global poly_image
import copy

# for undo
global item_dict_list
global enable_undo
global init_item_dict

file_name = None
poly_done = False
global store_vertex_list
vertex_list = []
annotation_list = []

item_dict_list=[]
init_item_dict = {}


init_item_dict['out_image'] = None
init_item_dict['show_json'] = None
init_item_dict['poly_done'] = False

init_item_dict['vertex_list'] = []
init_item_dict['annotation_list'] = []

item_dict_list.append(init_item_dict)

def lang_translation(region_name):
    if region_name=="起居室":
        return "living region"
    if region_name=="书房":
        return "study region"
    if region_name=="卧室":
        return "sleeping region"
    if region_name=="饭厅":
        return "dinning region"
    if region_name=="厨房":
        return "cooking region"
    if region_name=="浴室":
        return "bathing region"
    if region_name=="储藏室":
        return "storage region"
    if region_name=="厕所":
        return "restroom region"
    if region_name=="其它":
        return "others"
    return None

with gr.Blocks() as demo:

    input_file = gr.File(label="Input file")
    total_vertex_num = gr.Slider(
        label="Vertex Number",
        info="How different corners can be in a segment.",
        minimum=3,
        maximum=10,
        step=1,
        value=4,
    )

    def get_file_path(file):
        global file_name
        if file == None:
            return None, None
        file_name = os.path.basename(file.name)
        return gr.update(value=file.name), gr.update(value=file.name)

    with gr.Row():
        input_img = gr.Image(label="Image")
        output_img = gr.Image(label="Selected Polygon")

    label = gr.Radio(
        [
            "起居室","书房","卧室","饭厅","厨房","浴室","储藏室","厕所","其它"
        ],
        label="label",
        info="definition of this region",
    )

    def is_in_poly(p, poly):
        px, py = p
        is_in = False
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:
                    is_in = True
                    break
                elif x > px:
                    is_in = not is_in
        return is_in

    def draw_polygon(img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j][0] += is_in_poly((i, j), vertex_list) * int(255 * 0.4)
        return img

    def draw_dot(img,out_img,vertex_num, evt: gr.SelectData):
        global click_evt_list
        global poly_done
        global poly_image
        global vertex_list

        global enable_undo
        w, h, c = img.shape
        size = ceil(max([w, h]) * 0.01)

        out = img.copy() * 0.6


        if not poly_done:
            #print(item_dict_list[-1]['vertex_list'],item_dict_list[-1]['poly_done'],vertex_list,poly_done)
            new_item_dict = copy.deepcopy(item_dict_list[-1])
            new_item_dict['poly_done'] = poly_done
            new_item_dict['out_image'] = out_img


            new_item_dict['vertex_list'] = copy.deepcopy(vertex_list)
            item_dict_list.append(new_item_dict)
            vertex_list.append([evt.index[1], evt.index[0]])



            if len(vertex_list) == vertex_num:
                #print("OK!")

                for vertex in vertex_list:

                    out[
                        max(vertex[0] - size, 0) : min(
                            vertex[0] + size, out.shape[0] - 1
                        ),
                        max(vertex[1] - size, 0) : min(
                            vertex[1] + size, out.shape[1] - 1
                        ),
                    ] = np.array([255, 0, 0]).astype(np.uint8)

                new_out = draw_polygon(out.copy())
                poly_done = True
                new_out = new_out.astype(np.uint8)
                for object_id in object_ids:
                    k = list(object_ids).index(object_id)
                    o_x = bboxes[k][0]
                    o_y = bboxes[k][1]
                    o_x, o_y = get_position_in_render_image((o_x, o_y))
                    # global vertex_list
                    if object_types[k] not in exclude_type and is_in_poly(
                        (o_x, o_y), vertex_list
                    ):

                        new_out[
                            max(o_x - 1, 0) : min(o_x + 1, out.shape[0] - 1),
                            max(o_y - 1, 0) : min(o_y + 1, out.shape[1] - 1),
                        ] = np.array([0, 0, 255]).astype(np.uint8)
                poly_image = new_out
                global store_vertex_list
                store_vertex_list = copy.deepcopy(vertex_list)
                #print(store_vertex_list)
                vertex_list = []
                #print('before return', poly_done, vertex_list)

                return new_out
            for vertex in vertex_list:
                out[
                max(vertex[0] - size, 0): min(
                    vertex[0] + size, out.shape[0] - 1
                ),
                max(vertex[1] - size, 0): min(
                    vertex[1] + size, out.shape[1] - 1
                ),
                ] = np.array([255, 0, 0]).astype(np.uint8)
            out = out.astype(np.uint8)
            #print('before return',poly_done,vertex_list)
            return out


        else:
            return poly_image

    def get_position_in_render_image(point, ratio=20):

        x, y = point
        pixel_x = int((x - min_x) * ratio)
        pixel_y = int((y - min_y) * ratio)
        return pixel_y, pixel_x

    def new_draw_dot(img, evt: gr.SelectData):
        out = img.copy()
        min_distance = np.inf
        min_object_id = 0
        m_x = evt.index[1]
        m_y = evt.index[0]
        w, h, c = img.shape
        size = ceil(max([w, h]) * 0.01)
        for object_id in useful_object.keys():

            for index_ in range(len(object_ids)):
                if object_ids[index_] == object_id:
                    kkk = index_

                    break

            o_x = bboxes[kkk][0]
            o_y = bboxes[kkk][1]
            o_x, o_y = get_position_in_render_image((o_x, o_y))
            if (o_x - m_x) ** 2 + (o_y - m_y) ** 2 < min_distance:
                min_distance = (o_x - m_x) ** 2 + (o_y - m_y) ** 2
                min_object_id = object_id
                p = (o_x, o_y)

        out[
            max(p[0] - size, 0) : min(p[0] + size, out.shape[0] - 1),
            max(p[1] - size, 0) : min(p[1] + size, out.shape[1] - 1),
        ] = np.array([0, 0, 255]).astype(np.uint8)
        detail_img = gr.update(value=useful_object[min_object_id])
        return out, detail_img

    def annotate(label, output_img,show_json):
        global poly_done
        global vertex_list
        global click_evt_list
        global annotation_list
        new_item_dict = copy.deepcopy(item_dict_list[-1])
        new_item_dict['poly_done'] = poly_done
        new_item_dict['out_image'] = output_img
        new_item_dict['show_json'] = show_json
        new_item_dict['annotation_list'] = annotation_list
        global store_vertex_list


        new_item_dict['vertex_list'] = vertex_list
        item_dict_list.append(new_item_dict)

        if poly_done and label != None:


            annotation = {}
            annotation["id"] = len(annotation_list)
            annotation["label"] = lang_translation(label)


            annotation["vertex"] = store_vertex_list
            annotation_list.append(annotation)
            vertex_list = []

            poly_done = False

            return None, None, annotation_list
        else:

            gr.Info('Vertex num not match or have no label!Unable to annotation.')

            return label, output_img, annotation_list

    def clear(output_img, show_json):
        global annotation_list
        global poly_done
        global click_evt_list
        global vertex_list

        new_item_dict = copy.deepcopy(item_dict_list[-1])
        new_item_dict['poly_done'] = poly_done
        new_item_dict['out_image'] = output_img
        new_item_dict['show_json'] = show_json
        new_item_dict['annotation_list'] = annotation_list

        new_item_dict['vertex_list'] = vertex_list
        item_dict_list.append(new_item_dict)
        annotation_list = []
        vertex_list = []

        poly_done = False
        return None, None

    def undo(output_img, show_json):
        #print('undo!!!!')
        if len(item_dict_list)==1:
            return output_img, show_json
        else:
            global annotation_list
            global poly_done
            global vertex_list
            global click_evt_list
            output_img = item_dict_list[-1]['out_image']
            show_json = item_dict_list[-1]['show_json']
            annotation_list = item_dict_list[-1]['annotation_list']
            poly_done = item_dict_list[-1]['poly_done']
            vertex_list = item_dict_list[-1]['vertex_list']
            del item_dict_list[-1]
            #print('get:',poly_done,vertex_list)

            return output_img,show_json

    def save_to_file():
        global annotation_list
        global file_name
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{file_name}.txt", "w") as file:
            file.write(str(annotation_list))
        annotation_list = []
        global click_evt_list
        click_evt_list = []
        global vertex_list
        vertex_list = []
        global poly_done
        poly_done = False
        global item_dict_list
        item_dict_list = [init_item_dict]
        return None, None, None, None, None

    annotate_btn = gr.Button("Annotate")
    clear_btn = gr.Button("Clear")
    undo_btn = gr.Button("Undo")
    save_btn = gr.Button("Save to file")

    with gr.Row():
        object_postion_img = gr.Image(label="Object Position")
        detail_show_img = gr.Image(label="Posed Image")

    show_json = gr.JSON(label="Annotate History")
    input_file.change(
        get_file_path, inputs=[input_file], outputs=[input_img, object_postion_img]
    )
    input_img.select(draw_dot, [input_img,output_img,total_vertex_num], output_img)
    object_postion_img.select(
        new_draw_dot, [input_img], [object_postion_img, detail_show_img]
    )
    clear_btn.click(fn=clear, inputs=[output_img, show_json], outputs=[output_img, show_json])
    undo_btn.click(fn=undo, inputs=[output_img, show_json], outputs=[output_img, show_json])
    annotate_btn.click(
        fn=annotate, inputs=[label,output_img,show_json], outputs=[label, output_img, show_json]
    )
    save_btn.click(
        fn=save_to_file,
        inputs=[],
        outputs=[
            input_file,
            output_img,
            show_json,
            object_postion_img,
            detail_show_img,
        ],
    )

demo.queue()
if __name__ == "__main__":

    import os

    # only load scene 0

    if os.path.exists("example_data/example.npy"):

        anno = np.load("example_data/example.npy", allow_pickle=True).item()
    else:

        anno = read_annotation_pickle("example_data/embodiedscan_infos_train_full.pkl")[
            "scene0000_00"
        ]
        np.save("example_data/example.npy", anno)

    bboxes = anno["bboxes"]
    object_ids = anno["object_ids"]
    object_types = anno["object_types"]
    visible_view_object_dict = anno["visible_view_object_dict"]
    point_cloud_path = "example_data/lidar/main.pcd"
    painted_img_dir = "example_data/anno_lang/painted_images"  # where is the color data
    output_dir = "example_data/anno_lang/region_anno"
    exclude_type = ["wall"]

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(point_cloud.points)
    sorted_indices = np.argsort(points[:, 2])
    points = points[sorted_indices]
    min_x, min_y, _ = np.min(points, axis=0)
    max_x, max_y, _ = np.max(points, axis=0)

    useful_object = {}
    for img_file in os.listdir(painted_img_dir):
        useful_object[int(img_file[:3])] = painted_img_dir + "/" + img_file

    demo.launch(show_error=True)