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

def get_position_in_render_image(point,min_x, min_y,ratio=20):
    x, y= point
    pixel_x = int((x - min_x) * ratio)
    pixel_y = int((y - min_y) * ratio)
    return pixel_y,pixel_x

def get_data(input_annotation_path,point_cloud_path):
    '''
    extracts the min_x, min_y, and region_with_label from input data
    Returns:
        min_x,min_y: the minimum x and y coordinates of the point cloud
        region_with_label: a list of dict [{id:int, label:str, vertex:[(x1,y1),(x2,y2),...,(xn,yn)]]
    '''

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    region_with_label_file = open(input_annotation_path, "r", encoding="UTF-8")

    lines = region_with_label_file.readlines()
    s = ''
    for l in lines[0]:
        if l == "'":
            s += '"'
        else:
            s += l
    region_with_label = json.loads(s)



    points = np.asarray(point_cloud.points)
    sorted_indices = np.argsort(points[:, 2])
    points = points[sorted_indices]

    min_x, min_y, _ = np.min(points, axis=0)
    max_x, max_y, _ = np.max(points, axis=0)

    return min_x,min_y,region_with_label

def update_objects(region_with_label,object_ids,bboxes,min_x, min_y):
    '''
    update the object_ids for each region in the region_with_label 
    Returns:
        region_with_label (list(dict)): updated dict with object_ids for each region
    '''

    for region in region_with_label:

        poly = region['vertex']
        region['object_ids']=[]

        for _index in range(len(object_ids)):
            object_x = bboxes[_index][0]
            object_y = bboxes[_index][1]
            #print(get_position_in_render_image((object_x,object_y)))
            if is_in_poly(get_position_in_render_image((object_x,object_y),min_x, min_y),poly):
                region['object_ids'].append(object_ids[_index])
    return region_with_label


if __name__ == "__main__":
    file_name = 'point_cloud_top_view.png'
    input_annotation_path = f"./example_data/anno_lang/region_anno/{file_name}.txt" #The result of manual annotation
    point_cloud_path = "./example_data/lidar/main.pcd"
    output_annotation_dir = "./example_data/anno_lang/region_anno"

    ratio = 20 #ratio of point to render image

    min_x, min_y, region_with_label = get_data(input_annotation_path,point_cloud_path)

    object_data = np.load("example_data/example.npy",allow_pickle=True).item()
    bboxes = object_data['bboxes']
    object_ids = object_data['object_ids']

    region_with_label = update_objects(region_with_label,object_ids,bboxes,min_x, min_y)

    os.makedirs(output_annotation_dir, exist_ok=True)
    with open(f"{output_annotation_dir}/{file_name}_updated.txt", 'w') as file:
        file.write(str(region_with_label))











