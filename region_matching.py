import numpy as np
import json
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
def get_position_in_mesh_render_image(point, center_x, center_y, num_pixels_per_meter, photo_pixel):

    dx = point[0] - center_x
    dy = point[1] - center_y

    ox = (int(num_pixels_per_meter * dx) + photo_pixel[1] // 2)
    oy = photo_pixel[0] // 2 - (int(num_pixels_per_meter * dy))

    return oy, ox


def get_data(input_annotation_path):
    '''extract data from input files'''


    region_with_label_file = open(input_annotation_path, "r", encoding="UTF-8")

    lines = region_with_label_file.readlines()
    s = ''
    for l in lines[0]:
        if l == "'":
            s += '"'
        else:
            s += l
    region_with_label = json.loads(s)

    return region_with_label

def process_data(region_with_label, scene_id, object_ids, bboxes, center_x, center_y, num_pixels_per_meter):
    '''process the annotation data'''

    img = cv2.imread(f'data/{scene_id}/render.png')
    print(img.shape)

    for region in region_with_label:

        poly = region['vertex']
        region['object_ids']=[]

        for _index in range(len(object_ids)):
            object_x = bboxes[_index][0]
            object_y = bboxes[_index][1]
            x, y = get_position_in_mesh_render_image((object_x, object_y), center_x, center_y, num_pixels_per_meter, img.shape[:2])
            img[x-3:x+3, y-3:y+3] = (255, 0, 0)
            #print(get_position_in_render_image((object_x, object_y)))
            if is_in_poly((x, y), poly):
                region['object_ids'].append(object_ids[_index])
                print(object_data['object_types'][_index])
    
    show_objects_in_bev = True
    if show_objects_in_bev:
        cv2.imwrite('testing.png', img)
    return region_with_label


if __name__ == "__main__":

    scene_id = '1mp3d_0000_region1'

    all_scene_info = np.load('all_render_param.npy', allow_pickle=True).item()
    region_with_label = get_data(f'data/{scene_id}/region_segmentation.txt')
    scene_info = all_scene_info[scene_id]

    from utils_read import read_annotation_pickles
    annotation_data = read_annotation_pickles(["embodiedscan_infos_train_full.pkl", "embodiedscan_infos_val_full.pkl"])
    object_data = annotation_data[scene_id]
    bboxes = object_data['bboxes']
    object_ids = object_data['object_ids']

    region_with_label = process_data(region_with_label, scene_id, object_ids, bboxes, scene_info['center_x'], scene_info['center_y'], scene_info['num_pixels_per_meter'])
    print(region_with_label)

