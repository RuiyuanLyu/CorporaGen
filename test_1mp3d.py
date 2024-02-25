import open3d as o3d  # version 0.16.0. NEVER EVER use version 0.17.0 or later, or you will get fucked by vis control.
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils_read import read_axis_align_matrix, load_json, reverse_121_mapping

from utils_read import read_annotation_pickle
def compute_extrinsic_matrix(lookat_point, camera_coords):
    """
        lookat_point: 3D point in world coordinate
        camera_coords: 3D point in world coordinate
        NOTE: the camera convention is xyz:RDF
        NOTE: works differently if the camera is looking straight up or down
        return: 4 x 4 extrinsic matrix, world coord -> camera coord
    """
    camera_direction = lookat_point - camera_coords
    camera_direction_normalized = camera_direction / np.linalg.norm(camera_direction)
    up_vector = np.array([0, 0, -1])
    if np.allclose(camera_direction_normalized, up_vector) or np.allclose(camera_direction_normalized, -up_vector):
        up_vector = np.array([0, -1, 0])
    right_vector = np.cross(up_vector, camera_direction_normalized)
    right_vector_normalized = right_vector / np.linalg.norm(right_vector)
    true_up_vector = np.cross(camera_direction_normalized, right_vector_normalized)
    view_direction_matrix = np.vstack((right_vector_normalized, true_up_vector, camera_direction_normalized))
    extrinsic = np.zeros((4, 4))
    extrinsic[:3, :3] = view_direction_matrix
    extrinsic[:3, 3] = - view_direction_matrix @ camera_coords
    extrinsic[3, 3] = 1
    return extrinsic


# testing_scannet_file = "./data/scannet/scene0000_00/lidar/main.pcd"
# testing_mp3d_file = "./data/mp3d/1mp3d_0000_region0/lidar/main.pcd"
# testing_3rscan_dir = "./data/3rscan/3rscan0041/lidar/" # contains mesh.refined_0.png, mesh.refined.mtl, mesh.refined.v2.obj, axis_align_matrix.npy

# Load the testing point cloud file and visualize it
def load_point_cloud(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    return pcd

def _render_2d_bev(point_cloud_path, output_path, axis_align_matrix=None, resolution=20, roof_percentage=0.0,get_data=False):
    if axis_align_matrix is None:
        axis_align_matrix = np.eye(4)
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    points = np.asarray(point_cloud.points)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)  # shape (n, 4)
    points = axis_align_matrix @ points.T  # shape (4, n)
    points = (points.T)[:, :3]
    colors = np.asarray(point_cloud.colors)
    colors = colors[:, [2, 1, 0]]
    sorted_indices = np.argsort(points[:, 2])
    if roof_percentage > 0:
        num_points = len(sorted_indices)
        num_to_remove = int(num_points * roof_percentage)
        sorted_indices = sorted_indices[:-num_to_remove]
    points = points[sorted_indices]
    colors = colors[sorted_indices]

    min_x, min_y, _ = np.min(points, axis=0)
    max_x, max_y, _ = np.max(points, axis=0)

    image_width = int((max_x - min_x) * resolution) + 1
    image_height = int((max_y - min_y) * resolution) + 1



    projection_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    print(f"num points considered:{len(points)}")
    x, y, z = points.T
    pixel_x = ((x - min_x) * resolution).astype(int)
    pixel_y = ((y - min_y) * resolution).astype(int)
    sorted_indices = np.lexsort((-z, y, x))
    points = np.column_stack((pixel_x, pixel_y))
    points = points[sorted_indices]
    colors = colors[sorted_indices]
    cur_x, cur_y = None, None
    for i in range(len(points)):
        if cur_x == points[i, 0]:
            if cur_y == points[i, 1]:
                continue
            cur_y = points[i, 1]
            projection_image[cur_y, cur_x, :] = (255 * colors[i]).astype(np.uint8)
            continue
        cur_x, cur_y = points[i]
        projection_image[cur_y, cur_x, :] = (255 * colors[i]).astype(np.uint8)

    plt.imshow(projection_image, origin='lower')
    plt.axis('off')  # hide axis ticks and labels
    cv2.imwrite(output_path, projection_image)
    print("Successfully saved to", output_path)
    plt.show()

    for kkk in range(len(bboxes)):
        o_x = bboxes[kkk][0]
        o_y = bboxes[kkk][1]
        o_x, o_y = get_position_in_render_image((o_x, o_y),min_x,min_y)
        projection_image[o_x-1:o_x+1,o_y-1:o_y+1] = (0,0,255)
    cv2.imwrite('check.png',projection_image)

def get_position_in_render_image(point,min_x,min_y, ratio=20):

    x, y = point
    pixel_x = int((x - min_x) * ratio)
    pixel_y = int((y - min_y) * ratio)
    return pixel_y, pixel_x
if __name__ == "__main__":

    scene_id = '1mp3d_0000_region0'
    #scene_id ='scene0000_00'


    pcd_path = f'./{scene_id}/lidar/main.pcd'
    anno = read_annotation_pickle("example_data/embodiedscan_infos_train_full.pkl")['1mp3d_0000_region0']
    bboxes = anno["bboxes"]
    object_ids = anno["object_ids"]
    object_types = anno["object_types"]
    matrix = anno['axis_align_matrix']

    _render_2d_bev(pcd_path,f"./{scene_id}/anno_lang/render.png")




