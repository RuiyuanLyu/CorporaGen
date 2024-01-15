import open3d as o3d
import numpy as np
import cv2
import matplotlib
from .linemesh import LineMesh

EPS = 1e-4
ALPHA = 0.25


def draw_box3d_on_img(img, box, color, label, extrinsic, intrinsic, alpha=None):
    """
        Draw a 3D box on an image.
        Args:
            img (numpy.ndarray): An image.
            box (open3d.geometry.OrientedBoundingBox): A box.
            color (tuple): RGB color of the box.
            label (str): Label of the box.
            extrinsic (numpy.ndarray): 4x4 matrix camera to world.
            intrinsic (numpy.ndarray): Intrinsic matrix of the camera.
            alpha (float): Alpha value of the drawn faces.
        Returns:
            numpy.ndarray: An image with the box drawn on it.
    """
    global occupied
    extrinsic_w2c = np.linalg.inv(extrinsic)
    h, w, _ = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T

    camera_pos_in_world = (extrinsic @ np.array([0, 0, 0, 1]).reshape(4,1)).transpose()
    if is_inside_box(box, camera_pos_in_world):
        return

    corners = np.asarray(box.get_box_points())
    corners = corners[[0,1,7,2,3,6,4,5]]
    corners = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    corners_img = intrinsic @ extrinsic_w2c @ corners.transpose()
    corners_img = corners_img.transpose()
    corners_pixel = np.zeros((corners_img.shape[0], 2))
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
    img[all_mask] = img[all_mask] * alpha + (1 - alpha) * np.array(color)

    if (all_mask.any()):
        textpos = np.min(corners_pixel, axis=0).astype(np.int32)
        textpos[0] = np.clip(textpos[0], a_min=0, a_max=w)
        textpos[1] = np.clip(textpos[1], a_min=0, a_max=h)
        draw_text(img, label, pos=textpos, bound = (w, h), text_color=(255, 255, 255), text_color_bg=color)
    return img




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
          text_color_bg=(0, 0, 0)
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
        Returns:
            tuple: Size of the text.
    """

    global occupied
    x, y = pos
    w, h = bound
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if y * 2 > h:
        dy = -10
    else:
        dy = 10
    
    try:
        while occupied[y, x] or occupied[y, x+text_w] or occupied[y+text_h, x] or occupied[y + text_h, x + text_w]:
            y += dy
    except:
        pass
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    
    occupied[y:y+text_h, x:x+text_w] = True
    
    return text_size


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
            colors (numpy.ndarray): (N, 3) array of RGB colors.
        Returns:
            list: A list of open3d.geometry.OrientedBoundingBox objects.
    """
    n = bbox.shape[0]
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