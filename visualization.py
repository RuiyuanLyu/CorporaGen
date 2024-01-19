import open3d as o3d
import numpy as np
import cv2
import matplotlib
import json
from linemesh import LineMesh
from utils_read import read_bboxes_json, read_intrinsic, read_extrinsic
EPS = 1e-4
ALPHA = 0.15


def annotate_image_with_single_3dbbox_path_mode(img_path, object_json_path, intrinsic_path, extrinsic_path, axis_align_matrix_path, out_img_path, object_id):
    """
        Annotate an image with a single 3D bounding box, and also object type and id.
        Args:
            img_path: path to the image to be annotated
            object_json_path: path to the json file containing all the 3D bounding boxes in the scene
            intrinsic_path: path to the intrinsic matrix
            extrinsic_path: path to the extrinsic matrix, camera to world'
            axis_align_matrix_path: path to the extra extrinsic, world' to world
            out_img_path: path to save the annotated image
        Returns:
            None
    """
    img = cv2.imread(img_path)
    intrinsic = read_intrinsic(intrinsic_path)
    extrinsic = read_extrinsic(extrinsic_path) # camera to world'
    axis_align_matrix = read_extrinsic(axis_align_matrix_path) # world' to world
    extrinsic = axis_align_matrix @ extrinsic # camera to world
    bboxes, object_ids, object_types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    index = np.where(object_ids == object_id)[0][0]
    color_dict = get_color_map('color_map.txt')
    color = color_dict.get(object_types[index], (0, 0, 192))
    bboxes = get_9dof_boxes(bboxes, 'xyz', color)
    label = str(object_ids[index]) + ' ' + object_types[index]
    img, occupency_map = draw_box3d_on_img(img, bboxes[index], color, label, extrinsic, intrinsic, occupency_map=None)
    cv2.imwrite(out_img_path, img)
    print("Annotated image saved to  %s" % out_img_path)


def annotate_image_with_3dbboxes_path_mode(img_path, object_json_path, intrinsic_path, extrinsic_path, axis_align_matrix_path, out_img_path):
    """
        Annotate an image with 3D bounding boxes, and also object types and ids.
        Args:
            img_path: path to the image to be annotated
            object_json_path: path to the json file containing all the 3D bounding boxes in the scene
            intrinsic_path: path to the intrinsic matrix
            extrinsic_path: path to the extrinsic matrix, camera to world'
            axis_align_matrix_path: path to the extra extrinsic, world' to world
            out_img_path: path to save the annotated image
        Returns:
            None
    """
    img = cv2.imread(img_path)
    intrinsic = read_intrinsic(intrinsic_path)
    extrinsic = read_extrinsic(extrinsic_path) # camera to world'
    axis_align_matrix = read_extrinsic(axis_align_matrix_path) # world' to world
    extrinsic = axis_align_matrix @ extrinsic # camera to world
    bboxes, object_ids, object_types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    indices, distances = sort_objects_by_projection_distance(bboxes, extrinsic)
    bboxes = bboxes[indices]
    object_ids = [object_ids[i] for i in indices]
    object_types = [object_types[i] for i in indices]
    color_dict = get_color_map('color_map.txt')
    bboxes = get_9dof_boxes(bboxes, 'xyz', (0, 0, 192))
    occupency_map = np.zeros_like(img[:, :, 0], dtype=bool)
    for i, bbox in enumerate(bboxes):
        if distances[i] > 100 or distances[i] < 0:
            continue
        color = color_dict.get(object_types[i], (0, 0, 192))
        label = str(object_ids[i]) + ' ' + object_types[i]
        img, occupency_map = draw_box3d_on_img(img, bbox, color, label, extrinsic, intrinsic, occupency_map=occupency_map)
    cv2.imwrite(out_img_path, img)
    print("Annotated image saved to  %s" % out_img_path)


def get_color_map(path='color_map.txt'):
    """
        Data structure of the input file:
        item1 [r, g, b]
        item2 [r, g, b]
       ...
       Returns:
            item_colors (dict): a dictionary of item to color mapping
    """
    with open(path, 'r') as f:
        txt_content = f.read()
    item_colors = {}
    for line in txt_content.strip().split('\n'):
        item, color = line.split('[')
        item = item.strip()
        color = color.strip('] ').split(',')
        color = tuple(int(c) for c in color)
        item_colors[item] = color
    return item_colors


def sort_objects_by_projection_distance(bboxes, extrinsic):
    """
        Sort objects by distance from the camera.
        Args:
            bboxes (numpy.ndarray): Nx9 bboxes of the N objects.
            extrinsic (numpy.ndarray): 4x4 extrinsic, camera to world.
        Returns:
            sorted_indices (numpy.ndarray): a permutation of the indices of the input boxxes
    """
    centers = bboxes[:, :3]
    centers = np.concatenate([centers, np.ones((centers.shape[0], 1))], axis=1) # shape (N, 4)
    centers_in_camera = extrinsic @ centers.transpose() # shape (4, N)
    distance = centers_in_camera[2, :] # shape (N,)
    sorted_indices = np.argsort(-distance)
    return sorted_indices, distance[sorted_indices]
    

def visualize_object_types_on_sam_image(sam_img_path, sam_json_path, object_json_path, id_mapping, out_sam_img_path):
    """
        Render object types on SAM image, and save the result to a new SAM image.
        Args:
            sam_img_path: path to SAM image
            sam_json_path: path to SAM json
            object_json_path: path to object json, containing bboxes, object ids, and object types
            id_mapping: dict, mapping from SAM ids to object ids
            out_sam_img_path: path to save the new SAM image with object types
        Returns:
            None
    """
    sam_img = cv2.imread(sam_img_path) # shape 480, 640, 3
    with open(sam_json_path, 'r') as f:
        sam_id_json = json.load(f)
    sam_id_array = np.array(sam_id_json, dtype=int)
    bboxes, object_ids, object_types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    for sam_id, object_id in id_mapping.items():
        list_idx = np.where(object_ids == object_id)[0][0]
        object_type = object_types[list_idx]
        text_to_show = str(object_id) + object_type
        us, vs = np.where(sam_id_array == sam_id)
        maxu, maxv, minu, minv = np.max(us), np.max(vs), np.min(us), np.min(vs)
        caption_center = (int((maxv + minv) / 2), int((maxu + minu) / 2)) # u coresponding to y, v coresponding to x
        text_scale = min((maxu - minu), (maxv - minv)) / 200
        text_scale = max(text_scale, 0.3)
        cv2.putText(sam_img, text_to_show, caption_center, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 2)
        if text_scale > 1:
            cv2.rectangle(sam_img, (minv, minu), (maxv, maxu), (0, 0, 255), 2)
    cv2.imwrite(out_sam_img_path, sam_img)
    print("Texted image saved to  %s" % out_sam_img_path)


def draw_box3d_on_img(img, box, color, label, extrinsic, intrinsic, alpha=None, occupency_map=None, ignore_outside=True):
    """
        Draw a 3D box on an image.
        Args:
            img (numpy.ndarray): shape (h, w, 3)
            box (open3d.geometry.OrientedBoundingBox): A 3D box.
            color (tuple): RGB color of the box.
            label (str): Label of the box.
            extrinsic (numpy.ndarray): 4x4 extrinsic, camera to world.
            intrinsic (numpy.ndarray): 4x4 (extended) intrinsic.
            alpha (float): Alpha value of the drawn faces.
            occupency_map (numpy.ndarray): boolean array, occupency map of the image.
        Returns:
            img (numpy.ndarray): Updated image with the box drawn on it.
            occupency_map (numpy.ndarray): updated occupency map 
    """
    extrinsic_w2c = np.linalg.inv(extrinsic)
    h, w, _ = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T

    camera_pos_in_world = (extrinsic @ np.array([0, 0, 0, 1]).reshape(4,1)).transpose()
    if is_inside_box(box, camera_pos_in_world):
        return img, occupency_map
    
    if ignore_outside:
        center = box.get_center()
        center_2d = intrinsic @ extrinsic_w2c @ np.array([center[0], center[1], center[2], 1]).reshape(4,1)
        center_2d = center_2d[:2] / center_2d[2]
        if (center_2d[0] < 0 or center_2d[0] > w or center_2d[1] < 0 or center_2d[1] > h):
            return img, occupency_map

    corners = np.asarray(box.get_box_points()) # shape (8, 3)
    corners = corners[[0,1,7,2,3,6,4,5]] # shape (8, 3)
    corners = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1) # shape (8, 4)
    corners_img = intrinsic @ extrinsic_w2c @ corners.transpose() # shape (4, 8)
    corners_img = corners_img.transpose() # shape (8, 4)
    if (corners_img[:, 2] < EPS).any():
        return img, occupency_map
    corners_pixel = np.zeros((corners_img.shape[0], 2)) # shape (8, 2)
    for i in range(corners_img.shape[0]):
        corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])
    lines = [[0,1], [1,2],[2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
    faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [3,2,6,7], [0,3,7,4], [1,2,6,5]]
    for line in lines:
        if (corners_img[line][:, 2] < EPS).any():
            continue
        px = corners_pixel[line[0]].astype(np.int32)
        py = corners_pixel[line[1]].astype(np.int32)
        cv2.line(img, (px[0], px[1]), (py[0], py[1]), color, 2)
    
    all_mask = np.zeros((h,w), dtype=bool)
    for face in faces:
        if (corners_img[face][:, 2] < EPS).any():
            continue
        pts = corners_pixel[face]
        p = matplotlib.path.Path(pts[:, :2])
        mask = p.contains_points(pixel_points).reshape((h, w))
        all_mask = np.logical_or(all_mask, mask)
    if alpha is None:
        alpha = ALPHA
    img[all_mask] = img[all_mask] * (1 - alpha) + alpha * np.array(color)

    if (all_mask.any()):
        textpos = np.min(corners_pixel, axis=0).astype(np.int32)
        textpos[0] = np.clip(textpos[0], a_min=0, a_max=w)
        textpos[1] = np.clip(textpos[1], a_min=0, a_max=h)
        occupency_map = draw_text(img, label, pos=textpos, bound = (w, h), text_color=(255, 255, 255), text_color_bg=color, occupency_map=occupency_map)
    return img, occupency_map


def is_inside_box(box, point):
    """
        Check if a point is inside a box.
        Args:
            box (open3d.geometry.OrientedBoundingBox): A box.
            point (numpy.ndarray): A point.
        Returns:
            bool: True if the point is inside the box, False otherwise.
    """
    point_vec = o3d.utility.Vector3dVector(point[:, :3])
    inside_idx = box.get_point_indices_within_bounding_box(point_vec)
    return len(inside_idx) > 0


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          bound = (0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0),
          occupency_map=None
          ):
    """
        Draw text on an image.
        Args:
            img (numpy.ndarray): Image to be drawn on.
            text (str): Text to be drawn.
            font (int): Font type.
            pos (tuple): Position of the text.
            bound (tuple): Bound of the text.
            font_scale (float): Font scale.
            font_thickness (int): Font thickness.
            text_color (tuple): RGB color of the text.
            text_color_bg (tuple): RGB color of the background.
            occupency_map (numpy.ndarray): boolean array, occupency map of the image.
        Returns:
            occupency_map (numpy.ndarray): updated occupency map 
    """
    if occupency_map is None:
        occupency_map = np.zeros_like(img[..., 0], dtype=bool)
    x, y = pos
    w, h = bound
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if y * 2 > h:
        dy = -10
    else:
        dy = 10
    
    try:
        while occupency_map[y, x] or occupency_map[y, x+text_w] or occupency_map[y+text_h, x] or occupency_map[y + text_h, x + text_w]:
            y += dy
    except:
        pass
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    
    occupency_map[y:y+text_h, x:x+text_w] = True
    
    return occupency_map

def get_boxes_with_thickness(boxes, line_width=0.02):
    """
        Convert OrientedBoundingBox objects to LineSet objects with thickness.
        Args:
            boxes (list): A list of open3d.geometry.OrientedBoundingBox objects.
            line_width (float): Thickness of the lines.
        Returns:
            list: A list of open3d.geometry.LineSet objects with thickness.
    """
    results = []
    for box in boxes:
        bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        bbox_lines_width = LineMesh(points=bbox_lines.points, lines=bbox_lines.lines, colors=box.color, radius = line_width)
        results += bbox_lines_width.cylinder_segments
    return results

def get_9dof_boxes(bbox, mode, colors):
    """
        Get a list of open3d.geometry.OrientedBoundingBox objects from a (N, 9) array of bounding boxes.
        Args:
            bbox (numpy.ndarray): (N, 9) array of bounding boxes.
            mode (str): 'xyz' or 'zxy' for the rotation mode.
            colors (numpy.ndarray): (N, 3) array of RGB colors, or a single RGB color for all boxes.
        Returns:
            list: A list of open3d.geometry.OrientedBoundingBox objects.
    """
    n = bbox.shape[0]
    if isinstance(colors, tuple):
        colors = np.tile(colors, (n, 1))
    elif len(colors.shape) == 1:
        colors = np.tile(colors.reshape(1, 3), (n, 1))
    assert colors.shape[0] == n and colors.shape[1] == 3
    geo_list = []
    for i in range(n):
        center = bbox[i][:3].reshape(3,1)
        scale = bbox[i][3:6].reshape(3,1)
        rot = bbox[i][6:].reshape(3,1)
        if mode == 'xyz':
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(rot)
        elif mode == 'zxy':
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
        else:
            raise NotImplementedError
        geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
        color = colors[i]
        geo.color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        geo_list.append(geo)
    return geo_list

def visualize_distribution_hist(data, num_bins=20):
    import matplotlib.pyplot as plt
    plt.hist(data, bins=num_bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    img_id = "02300"
    img_path = f"./example_data/posed_images/{img_id}.jpg"
    object_json_path = f"./example_data/label/main_MDJH01.json"
    intrinsic_path = f"./example_data/posed_images/intrinsic.txt"
    extrinsic_path = f"./example_data/posed_images/{img_id}.txt"
    axis_align_matrix_path = "./example_data/label/rot_matrix.npy"
    out_img_path = f"./{img_id}_annotated.jpg"
    annotate_image_with_3dbboxes_path_mode(img_path, object_json_path, intrinsic_path, extrinsic_path, axis_align_matrix_path, out_img_path)
    # annotate_image_with_single_3dbbox_path_mode(img_path, object_json_path, intrinsic_path, extrinsic_path, axis_align_matrix_path, out_img_path, object_id=50)