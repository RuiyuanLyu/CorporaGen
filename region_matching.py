import numpy as np
import json
import cv2
import matplotlib


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


def get_data(input_annotation_path):
    '''extract data from input files'''
    with open(input_annotation_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
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
    show_objects_in_bev = True
    if show_objects_in_bev:
        cv2.imwrite('testing.png', img)
    return region_with_label


if __name__ == "__main__":

    scene_id = 'scene0000_00'

    all_scene_info = np.load('all_render_param.npy', allow_pickle=True).item()
    region_with_label = get_data(f'data/{scene_id}/region_segmentation_shujutang_0，shujutang_1.txt')
    scene_info = all_scene_info[scene_id]

    from utils_read import read_annotation_pickles
    annotation_data = read_annotation_pickles(["embodiedscan_infos_train_full.pkl", "embodiedscan_infos_val_full.pkl"])
    object_data = annotation_data[scene_id]
    bboxes = object_data['bboxes']
    object_ids = object_data['object_ids']

    region_with_label = process_data(region_with_label, scene_id, object_ids, bboxes, scene_info['center_x'], scene_info['center_y'], scene_info['num_pixels_per_meter'])
    print(region_with_label)

