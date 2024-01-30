import numpy as np
import open3d as o3d
import json
import os
import cv2


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

def get_position_in_render_image(point, ratio):
    x, y= point
    pixel_x = int((x - min_x) * ratio)
    pixel_y = int((y - min_y) * ratio)
    return pixel_y, pixel_x

def get_data(output_annotation_path, visibility_json_path, object_json_path, point_cloud_path):
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    region_with_label_file = open(output_annotation_path, "r", encoding="UTF-8")
    visible_objects_file = open(visibility_json_path, "r")
    object_json_file = open(object_json_path, "r")
    lines = region_with_label_file.readlines()
    s = ''
    for l in lines[0]:
        if l == "'":
            s += '"'
        else:
            s += l
    region_with_label = json.loads(s)

    content = visible_objects_file.read()
    a = json.loads(content)
    content = object_json_file.read()
    object_data = json.loads(content)
    visible_objects_id = []
    for camera_id in a.keys():
        for object_id in a[camera_id]:
            if object_id not in visible_objects_id:
                visible_objects_id.append(object_id)

    points = np.asarray(point_cloud.points)
    sorted_indices = np.argsort(points[:, 2])
    points = points[sorted_indices]

    min_x, min_y, _ = np.min(points, axis=0)
    max_x, max_y, _ = np.max(points, axis=0)

    return visible_objects_id, min_x, min_y, region_with_label, object_data

def process_data(region_with_label, object_data):
    for region in region_with_label:

        poly = region['vertex']
        region['object_ids']=[]

        for object in object_data:
            object_x = object["psr"]["position"]["x"]
            object_y = object["psr"]["position"]["y"]
            #print(get_position_in_render_image((object_x, object_y)))
            if is_in_poly(get_position_in_render_image((object_x, object_y), ratio), poly) and int(object["obj_id"]) in visible_objects_id:
                region['object_ids'].append(int(object["obj_id"]))
    return region_with_label


if __name__ == "__main__":
    file_name = 'point_cloud_top_view.png'
    input_annotation_path = f"C:/Users/86186/Desktop/3D_scene_data/CorporaGen-master/CorporaGen-master/annotation/{file_name}.txt" #The result of manual annotation
    visibility_json_path = "C:/Users/86186/Desktop/3D_scene_data/anno_lang/visible_objects.json"
    object_json_path = "C:/Users/86186/Desktop/3D_scene_data/example_data/label/main_MDJH13.json"
    point_cloud_path = "C:/Users/86186/Desktop/3D_scene_data/example_data/lidar/main.pcd"
    otuput_annotation_dir = "C:/Users/86186/Desktop/3D_scene_data/CorporaGen-master/CorporaGen-master/process_annotation"

    ratio = 20 #ratio of point to render image

    visible_objects_id, min_x, min_y, region_with_label, object_data = get_data(input_annotation_path, visibility_json_path, object_json_path, point_cloud_path)

    region_with_label = process_data(region_with_label, object_data)

    #print(region_with_label)

    os.makedirs(otuput_annotation_dir, exist_ok=True)
    with open(f"{otuput_annotation_dir}/{file_name}.txt", 'w') as file:
        file.write(str(region_with_label))

    #render_image = cv2.imread("../example_data/point_cloud_top_view.png")
    # print(shape_x, shape_y)
    # print(render_image.shape)


    # print(ratio)
    # object_min_x=100
    # object_min_y=100
    # object_max_x=0
    # object_max_y=0
    # for object in b:
    #     if int(object["obj_id"]) not in visible_objects_id:
    #         continue
    #     object_x = object["psr"]["position"]["x"]
    #     object_y = object["psr"]["position"]["y"]
    #     if object_x>object_max_x:
    #         object_max_x = object_x
    #     if object_y>object_max_y:
    #         object_max_y = object_y
    #     if object_x<object_min_x:
    #         object_min_x = object_x
    #     if object_y<object_min_y:
    #         object_min_y = object_y
    #     object_x, object_y=get_position_in_render_image((object_x, object_y), ratio)
    #     size=1
    #     render_image[max(object_x - size, 0):min(object_x + size, render_image.shape[0] - 1), max(object_y - size, 0):min(object_y + size, render_image.shape[1] - 1)] = np.array(
    #         [255, 0, 0]).astype(np.uint8)
    # cv2.imwrite('render_image.png', render_image)
    # print(object_max_x, object_max_y)
    # print(object_min_x, object_min_y)










