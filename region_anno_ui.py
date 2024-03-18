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
# -*- coding:utf-8 –*-
import os

import cv2
import gradio as gr  # Now: gradio==3.50.2 Deprecated: gradio==3.44.0
import numpy as np
from math import ceil
import json
import open3d as o3d
from utils_read import read_annotation_pickles
from render_bev import load_mesh, _render_2d_bev, take_bev_screenshot, process_mesh
from region_matching import get_data
import matplotlib
import copy
global init_item_dict
init_item_dict = {}

init_item_dict["out_image"] = None
init_item_dict["poly_done"] = False

init_item_dict["vertex_list"] = []
init_item_dict["annotation_list"] = []

RENDER_IMAGE_PATH = "data"


scene_list = os.listdir(RENDER_IMAGE_PATH)
scene_list.sort()

REGIONS = {"起居室/客厅区": "living region", 
           "书房/工作学习区": "study region", 
           "卧室/休息区": "sleeping region", 
           "饭厅/进食区": "dinning region", 
           "厨房/烹饪区": "cooking region", 
           "浴室/洗澡区": "bathing region", 
           "储藏区": "storage region", 
           "厕所": "toliet region", 
           "走廊/通道": "corridor region", 
           "空地": "open area region", 
           "其它": "other region"}
REGIONS_COLOR = {
    "living region": (0, 0, 255), 
    "study region": (0, 255, 0), 
    "sleeping region": (255, 0, 0), 
    "dinning region": (0, 255, 255), 
    "cooking region": (255, 255, 0), 
    "bathing region": (255, 20, 147), 
    "storage region": (122, 255, 122), 
    "toliet region": (69, 139, 0), 
    "corridor region": (139, 105, 20), 
    "open area region": (255, 0, 255), 
    "other region": (122, 122, 0)
}
REGIONS_COLOR = {k: np.array(v) for k, v in REGIONS_COLOR.items()}


def lang_translation(region_name):
    return REGIONS.get(region_name, None)


with gr.Blocks() as demo:
    click_evt_list = gr.State([])
    vertex_list = gr.State([])
    annotation_list = gr.State([])
    poly_done = gr.State(False)
    poly_image = gr.State()
    item_dict_list = gr.State([])
    enable_undo= gr.State(False)
    anno_result = gr.State([])
    store_vertex_list = gr.State([])
    to_rotate_clockwise_90 = gr.State(False)
    to_show_areas = gr.State(False)
    user_name_is_valid = gr.State(False)

    user_name = gr.Textbox(label='user name', value='', placeholder='在此输入用户名，首位必须为字母，不要带空格。')
    scene = gr.Dropdown(scene_list, label="在此选择待标注的场景")
    scene_anno_info = gr.Textbox('', visible=True, interactive=False)
    anno_img = gr.Image(label="result of annotation", interactive=True)
    show_label_box = gr.Textbox(label='label from annotation')

    total_vertex_num = gr.Slider(
        label="Vertex Number", 
        info="拖动滑块选择多边形的顶点个数", 
        minimum=3, 
        maximum=20, 
        step=1, 
        value=4, 
    )

    def check_user_name_validity(user_name):
        if len(user_name) == 0 or ' ' in user_name or not user_name[0].isalpha():
            gr.Warning("用户名不合法。请首位必须为字母，并不要带空格。请重新输入。")
            return False
        return True
    user_name.blur(check_user_name_validity, inputs=[user_name], outputs=[user_name_is_valid])

    def get_file(scene_id, user_name):
        annotation_list = []
        click_evt_list = []
        vertex_list = []
        poly_done = False
        item_dict_list = [init_item_dict]
        anno_result = []
        to_show_areas = False

        if not check_user_name_validity(user_name):
            return None, None, None, None, None, '', None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, anno_result, to_show_areas

        if scene_id == None or scene_id == 'None':
            return None, None, None, None, None, '', None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, anno_result, to_show_areas

        file_path_to_save = os.path.join(scene_info[scene_id]["output_dir"], f'region_segmentation_{user_name}.txt')
        anno_img_path = gr.update(value=f'{render_image_path}/{scene_id}/render.png')
        if os.path.exists(file_path_to_save):
            scene_anno_state = scene_id + ' 已经被标注过 ! ! ! ! !'
            gr.Info("该场景已经被标注过，若非必要请不要重复标注")
            anno_result = get_data(file_path_to_save)
            if os.path.exists(f'{render_image_path}/{scene_id}/render_anno.png'):
                anno_img_path = f'{render_image_path}/{scene_id}/render_anno.png'
            else:
                to_show_areas = True
        else:
            scene_anno_state = scene_id + ' 需要标注'
            anno_result = []

        return f'{render_image_path}/{scene_id}/render.png', f'{render_image_path}/{scene_id}/render.png', \
               None, None, None, scene_anno_state, anno_img_path, \
               annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, anno_result, to_show_areas




    with gr.Row():
        input_img = gr.Image(label="Image")

        output_img = gr.Image(label="Selected Polygon")

    label = gr.Radio(REGIONS.keys(), 
                     label="label", 
                     info="在此选择该区域发挥的功能", 
                     )


    def is_in_poly(ps, poly):
        """
            ps: a numpy array of shape (N, 2)
            poly: a polygon represented as a list of (x, y) tuples
        """
        if isinstance(ps, tuple):
            ps = np.array([ps])
        if len(ps.shape) == 1:
            ps = np.expand_dims(ps, axis=0)
        assert ps.shape[1] == 2
        assert len(ps.shape) == 2
        path = matplotlib.path.Path(poly)
        return path.contains_points(ps)

    def get_coverage_mask(h, w, poly):
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        pixel_points = np.vstack((y, x)).T
        p = matplotlib.path.Path(poly)
        mask = p.contains_points(pixel_points).reshape((h, w))
        return mask

    def draw_polygon(img, vertex_list):
        img = img.copy()
        coverage_mask = get_coverage_mask(img.shape[0], img.shape[1], vertex_list)
        img[coverage_mask] = img[coverage_mask] * 0.6 + np.array([255, 0, 0]) * 0.4
        return img


    def draw_dot(scene_id, img, out_img, vertex_num, 
                 click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, item_dict_list, store_vertex_list, 
                 evt: gr.SelectData
                 ):
        w, h, c = img.shape
        size = ceil(max([w, h]) * 0.01)

        out = img.copy() * 0.6

        if not poly_done:
            # print(item_dict_list[-1]["vertex_list"], item_dict_list[-1]["poly_done"], vertex_list, poly_done)
            new_item_dict = copy.deepcopy(item_dict_list[-1])
            new_item_dict["poly_done"] = poly_done
            new_item_dict["out_image"] = out_img
            new_item_dict["annotation_list"] = copy.deepcopy(annotation_list)

            new_item_dict["vertex_list"] = copy.deepcopy(vertex_list)
            item_dict_list.append(new_item_dict)
            vertex_list.append([evt.index[1], evt.index[0]]) # evt index is in h, w order

            if len(vertex_list) == vertex_num:

                for vertex in vertex_list:
                    out[
                    max(vertex[0] - size, 0): min(
                        vertex[0] + size, out.shape[0] - 1
                    ), 
                    max(vertex[1] - size, 0): min(
                        vertex[1] + size, out.shape[1] - 1
                    ), 
                    ] = np.array([255, 0, 0]).astype(np.uint8)

                new_out = draw_polygon(out.copy(), vertex_list)
                poly_done = True
                new_out = new_out.astype(np.uint8)
                ks = np.arange(len(scene_info[scene_id]["object_ids"]))
                xys_in_world = scene_info[scene_id]["bboxes"][:, :2]
                draw_size_of_object = 5
                ys, xs = get_position_in_mesh_render_image(xys_in_world, scene_info[scene_id]["center_x"], 
                                                         scene_info[scene_id]["center_y"], 
                                                         scene_info[scene_id]["num_pixels_per_meter"], 
                                                         img.shape[:2])
                is_in = is_in_poly(np.array([ys, xs]).T, vertex_list)
                for k, is_in_k in enumerate(is_in):
                    if is_in_k and scene_info[scene_id]["object_types"][k] not in exclude_type:
                        new_out[
                        max(ys[k] - draw_size_of_object, 0): min(ys[k] + draw_size_of_object, out.shape[0] - 1), 
                        max(xs[k] - draw_size_of_object, 0): min(xs[k] + draw_size_of_object, out.shape[1] - 1), 
                        ] = np.array([0, 0, 255]).astype(np.uint8)

                store_vertex_list = copy.deepcopy(vertex_list)
                # print(store_vertex_list)
                vertex_list = []
                # print('before return', poly_done, vertex_list)

                return new_out, click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, item_dict_list, store_vertex_list, 

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
            # print('before return', poly_done, vertex_list)
            return out, click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, item_dict_list, store_vertex_list, 



        else:
            return poly_image, click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, item_dict_list, store_vertex_list, 


    def get_position_in_mesh_render_image(points, center_x, center_y, num_pixels_per_meter, photo_pixel):
        """
            Args:
            points: a numpy array of shape (N, 2) IN WORLD COORDINATE
            phote_pixel: a tuple of (h, w)
            return: a numpy array of shape (N, 2) IN RENDER IMAGE COORDINATE, in y, x order
        """
        if isinstance(points, tuple):
            points = np.array([points])
        if len(points.shape) == 1:
            points = np.expand_dims(points, axis=0)
        assert points.shape[1] == 2
        assert len(points.shape) == 2
        dxs = points[:, 0] - center_x
        dys = points[:, 1] - center_y
        xs = (dxs * num_pixels_per_meter + photo_pixel[1] // 2).astype(int)
        ys = (photo_pixel[0] // 2 - dys * num_pixels_per_meter).astype(int)
        if len(xs) == 1:
            xs = xs[0]
            ys = ys[0]
        return ys, xs


    def new_draw_dot(scene_id, img, to_rotate_clockwise_90, evt: gr.SelectData):
        # import pdb; pdb.set_trace()
        out = img.copy()
        min_distance = np.inf
        min_object_id = 0
        m_x = evt.index[1]
        m_y = evt.index[0]
        if m_x is None or m_y is None:
            return out, None
        w, h, c = img.shape
        size = ceil(max([w, h]) * 0.01)
        useful_object_ids = list(scene_info[scene_id]["useful_object"].keys())
        useful_object_index = []
        for object_id in useful_object_ids:
            index = np.where(scene_info[scene_id]["object_ids"] == object_id)[0]
            useful_object_index.extend(index)
        useful_object_index = np.array(useful_object_index)
        xys_in_world = scene_info[scene_id]["bboxes"][useful_object_index, :2]
        xs, ys = get_position_in_mesh_render_image(xys_in_world, scene_info[scene_id]["center_x"], 
                                                 scene_info[scene_id]["center_y"], 
                                                 scene_info[scene_id]["num_pixels_per_meter"], 
                                                 img.shape[:2])
        distances = (xs - m_x) ** 2 + (ys - m_y) ** 2
        min_distance_index = np.argmin(distances)
        min_object_id = useful_object_ids[min_distance_index]
        p = (xs[min_distance_index], ys[min_distance_index])

        view_id = scene_info[scene_id]['useful_object_view_id'][min_object_id]
        view_id_idx = list(scene_info[scene_id]['view_ids']).index(view_id)
        extrinsics_c2w = scene_info[scene_id]['camera_extrinsics_c2w'][view_id_idx]
        front = extrinsics_c2w[:3, 2]
        pos = extrinsics_c2w[:3, 3]
        sx_in_world, sy_in_world = pos[0], pos[1]
        ex_in_world, ey_in_world = pos[0] + front[0], pos[1] + front[1]
        s_x, s_y = get_position_in_mesh_render_image((sx_in_world, sy_in_world), scene_info[scene_id]["center_x"], 
                                                     scene_info[scene_id]["center_y"], 
                                                     scene_info[scene_id]["num_pixels_per_meter"], 
                                                     img.shape[:2])
        e_x, e_y = get_position_in_mesh_render_image((ex_in_world, ey_in_world), scene_info[scene_id]["center_x"], 
                                                     scene_info[scene_id]["center_y"], 
                                                     scene_info[scene_id]["num_pixels_per_meter"], 
                                                     img.shape[:2])
        out = cv2.arrowedLine(out, (s_y, s_x), (e_y, e_x), (255, 0, 0), 2)
        out[
        max(p[0] - size, 0): min(p[0] + size, out.shape[0] - 1), 
        max(p[1] - size, 0): min(p[1] + size, out.shape[1] - 1), 
        ] = np.array([0, 0, 255]).astype(np.uint8)
        detail_img = gr.update(value=scene_info[scene_id]["useful_object"][min_object_id])

        if scene_id[:6] == '3rscan':
            to_rotate_clockwise_90 = True

        return out, detail_img, to_rotate_clockwise_90


    def rotate_clockwise_90(detail_img, to_rotate_clockwise_90):

        if to_rotate_clockwise_90:
            to_rotate_clockwise_90 = False
            detail_img = cv2.rotate(detail_img, cv2.ROTATE_90_CLOCKWISE)
        return detail_img, to_rotate_clockwise_90




    def show_areas(anno_img, to_show_areas, anno_result):

        if to_show_areas:
            to_show_areas = False
            anno_img = 0.6 * anno_img.copy()
            for anno_ in anno_result:
                print(anno_)
                label_ = anno_["label"]
                poly_ = anno_["vertex"]
                color_ = REGIONS_COLOR[label_]
                coverage_mask = get_coverage_mask(anno_img.shape[0], anno_img.shape[1], poly_)
                anno_img[coverage_mask] = anno_img[coverage_mask] + 0.4 * color_
            return anno_img.astype(np.uint8), to_show_areas, anno_result
        return anno_img, to_show_areas, anno_result


    def show_labels(anno_img, anno_result, evt: gr.SelectData):

        m_x = evt.index[1]
        m_y = evt.index[0]
        for anno_ in anno_result:
            label_ = anno_["label"]
            poly_ = anno_["vertex"]
            if is_in_poly((m_x, m_y), poly_):
                return label_, anno_result
        return "no annotation", anno_result


    def annotate(scene_id, label, output_img, poly_done, vertex_list, click_evt_list, 
                 annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list):


        if scene_id == None or scene_id == 'None':
            return None, None, None

        if poly_done and label != None:
            new_item_dict = copy.deepcopy(item_dict_list[-1])
            new_item_dict["poly_done"] = poly_done
            new_item_dict["out_image"] = output_img
            new_item_dict["annotation_list"] = copy.deepcopy(annotation_list)

            new_item_dict["vertex_list"] = copy.deepcopy(vertex_list)
            item_dict_list.append(new_item_dict)

            annotation = {}
            annotation["id"] = len(annotation_list)
            annotation["label"] = lang_translation(label)

            annotation["vertex"] = copy.deepcopy(store_vertex_list)
            annotation_list.append(annotation)
            vertex_list = []

            poly_done = False

            to_show_areas = True

            anno_result = copy.deepcopy(annotation_list)

            return None, None, annotation_list, poly_done, vertex_list, click_evt_list, \
                 annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list
        else:

            gr.Info('多边形顶点数不匹配或没有标签！请重新（继续）标注。')

            return label, output_img, annotation_list, poly_done, vertex_list, click_evt_list, \
                 annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list


    def clear(scene_id, output_img, annotation_list, poly_done, click_evt_list, vertex_list):

        if scene_id == None or scene_id == 'None':
            return [None]*6

        new_item_dict = copy.deepcopy(item_dict_list[-1])
        new_item_dict["poly_done"] = poly_done
        new_item_dict["out_image"] = output_img
        new_item_dict["annotation_list"] = copy.deepcopy(annotation_list)
        new_item_dict["vertex_list"] = copy.deepcopy(vertex_list)
        item_dict_list.append(new_item_dict)
        annotation_list = []
        vertex_list = []

        poly_done = False
        return None, annotation_list, annotation_list, poly_done, click_evt_list, vertex_list


    def undo(output_img, annotation_list, poly_done, vertex_list, click_evt_list, item_dict_list):
        # print('undo!!!!')

        # print(len(item_dict_list), item_dict_list[-1]["annotation_list"], item_dict_list[-1]['show_json'])
        if len(item_dict_list) == 1:
            return output_img, annotation_list, poly_done, vertex_list, click_evt_list
        else:


            output_img = item_dict_list[-1]["out_image"]
            annotation_list = copy.deepcopy(item_dict_list[-1]["annotation_list"])
            poly_done = item_dict_list[-1]["poly_done"]
            vertex_list = item_dict_list[-1]["vertex_list"]
            del item_dict_list[-1]

            return output_img, annotation_list, annotation_list, poly_done, vertex_list, click_evt_list


    def save_to_file(scene_id, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, user_name):
        if scene_id == None or scene_id == 'None':
            return None, None, None, None, None, None, None, None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list


        os.makedirs(scene_info[scene_id]['output_dir'], exist_ok=True)
        with open(f"{scene_info[scene_id]['output_dir']}/region_segmentation_{user_name}.txt", "w") as file:
            file.write(str(annotation_list))
        annotation_list = []
        click_evt_list = []
        vertex_list = []
        poly_done = False
        item_dict_list = [init_item_dict]

        return None, None, None, None, None, None, None, None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list


    with gr.Row():
        annotate_btn = gr.Button("标注")
        undo_btn = gr.Button("回退一步")
        save_btn = gr.Button("所有区域都已经标注完成，保存")
        clear_btn = gr.Button("清空当前场景所有标注（谨慎操作）")

    with gr.Row():
        object_postion_img = gr.Image(label="辅助查看区", interactive=True)
        detail_show_img = gr.Image(label="Posed Image")

    show_json = gr.JSON(label="Annotate History")
    scene.change(
        get_file, inputs=[scene, user_name], outputs=[input_img, object_postion_img, output_img, 
                                           show_json, detail_show_img, scene_anno_info, anno_img, 
                                           annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, 
                                           anno_result, to_show_areas
                                           ]
    )

    input_img.select(draw_dot, [scene, input_img, output_img, total_vertex_num, 
                                click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, 
                                item_dict_list, store_vertex_list, 
                                ], [output_img, click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, item_dict_list, store_vertex_list])
    object_postion_img.select(
        new_draw_dot, [scene, input_img, to_rotate_clockwise_90], [object_postion_img, detail_show_img, to_rotate_clockwise_90]
    )
    anno_img.change(
        show_areas, [anno_img, to_show_areas, anno_result], [anno_img, to_show_areas, anno_result]
    )
    anno_img.select(
        show_labels, [anno_img, anno_result], [show_label_box, anno_result]
    )
    detail_show_img.change(
        rotate_clockwise_90, [detail_show_img, to_rotate_clockwise_90], [detail_show_img, to_rotate_clockwise_90]
    )
    clear_btn.click(fn=clear, inputs=[scene, output_img, annotation_list, poly_done, click_evt_list, vertex_list], 
                    outputs=[output_img, show_json, annotation_list, poly_done, click_evt_list, vertex_list])
    undo_btn.click(fn=undo, inputs=[output_img, annotation_list, poly_done, vertex_list, click_evt_list, item_dict_list], 
                   outputs=[output_img, show_json, annotation_list, poly_done, vertex_list, click_evt_list])
    annotate_btn.click(
        fn=annotate, inputs=[scene, label, output_img, poly_done, vertex_list, click_evt_list, 
                 annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list], 
        outputs=[label, output_img, show_json, poly_done, vertex_list, click_evt_list, 
                 annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list]
    )
    annotate_btn.click(
        show_areas, [anno_img, to_show_areas, anno_result], [anno_img, to_show_areas, anno_result]
    )

    save_btn.click(
        fn=save_to_file, 
        inputs=[scene, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, user_name], 
        outputs=[
            scene, 
            output_img, 
            show_json, 
            object_postion_img, 
            detail_show_img, 
            scene_anno_info, 
            anno_img, 
            show_label_box, 
            annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list

        ], 
    )
demo.queue(concurrency_count=20)

if __name__ == "__main__":

    import os

    # 存储所有render images的文件夹目录
    render_image_path = RENDER_IMAGE_PATH
    # 不用显示的类型（但会记录）
    exclude_type = ["wall", "ceiling", "floor"]
    # 存储所有场景标注信息的列表
    anno_full = read_annotation_pickles(['embodiedscan_infos_train_full.pkl', 'embodiedscan_infos_val_full.pkl'])
    # 存储渲染参数的文件（坐标变换要用到）
    all_scene_info = np.load('all_render_param.npy', allow_pickle=True).item()
    # 输出文件夹
    output_dir = './region_annos/'
    # 辅助查看图文件夹
    painted_dir = RENDER_IMAGE_PATH

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    scene_info = {}
    from tqdm import tqdm

    for scene_id in tqdm(scene_list):
        anno = anno_full.get(scene_id, None)
        if anno is None:
            print(f"No annotation for {scene_id}")
            continue

        scene_info[scene_id] = {}

        scene_info[scene_id]["bboxes"] = anno["bboxes"]
        scene_info[scene_id]["object_ids"] = anno["object_ids"]
        scene_info[scene_id]["object_types"] = anno["object_types"]
        scene_info[scene_id]["visible_view_object_dict"] = anno["visible_view_object_dict"]
        scene_info[scene_id]["output_dir"] = f"{output_dir}/{scene_id}"
        painted_img_dir = f"{painted_dir}/{scene_id}/painted_objects"
        scene_info[scene_id]["useful_object"] = {}
        scene_info[scene_id]["useful_object_view_id"] = {}
        for img_file in os.listdir(painted_img_dir):
            if img_file.endswith(".png") or img_file.endswith(".jpg"):
                scene_info[scene_id]["useful_object"][int(img_file.split("_")[0])] = painted_img_dir + "/" + img_file
                scene_info[scene_id]["useful_object_view_id"][int(img_file.split("_")[0])] = "_".join(
                    img_file.split(".")[0].split("_")[2:])
        scene_info[scene_id]["view_ids"] = [path.split("/")[-1].split(".")[0] for path in anno["image_paths"]]
        scene_info[scene_id]["camera_extrinsics_c2w"] = [(anno["axis_align_matrix"] @ extrinsic) for extrinsic in
                                                         anno["extrinsics_c2w"]]
        scene_info[scene_id]["center_x"], scene_info[scene_id]["center_y"], scene_info[scene_id]["num_pixels_per_meter"] \
            = all_scene_info[scene_id]["center_x"], all_scene_info[scene_id]["center_y"], all_scene_info[scene_id][
            "num_pixels_per_meter"], 

    demo.launch(show_error=True, allowed_paths=[painted_dir], server_port=7859)

