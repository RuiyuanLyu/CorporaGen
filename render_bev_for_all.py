import open3d as o3d  # version 0.16.0. NEVER EVER use version 0.17.0 or later, or you will get fucked by vis control.
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.utils_read import read_axis_align_matrix, load_json, reverse_121_mapping,read_annotation_pickle


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


def load_mesh(mesh_dir):
    mesh = o3d.io.read_triangle_mesh(mesh_dir + "mesh.refined.v2.obj", enable_post_processing=True)
    axis_align_matrix = np.load(mesh_dir + "axis_align_matrix.npy")
    mesh.transform(axis_align_matrix)
    return mesh
def process_mesh(mesh,axis_align_matrix):
    return mesh.transform(axis_align_matrix)


def take_bev_screenshot(o3d_obj, filename, scene_info, get_data=False):

    min_x, min_y, min_z = np.min(o3d_obj.vertices, axis=0)
    max_x, max_y, max_z = np.max(o3d_obj.vertices, axis=0)

    if isinstance (scene_info,int):
        Z_HEIGHT = scene_info

        width = max_x - min_x
        height = max_y - min_y
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        num_pixels_per_meter = 100
        width_in_pixels = int(width * num_pixels_per_meter)
        height_in_pixels = int(height * num_pixels_per_meter)
        if width_in_pixels<=256 or height_in_pixels<=256:
            num_pixels_per_meter = 150
        if width_in_pixels>=1000 or height_in_pixels>=1000:
            num_pixels_per_meter = 50
        width_in_pixels = int(width * num_pixels_per_meter)
        height_in_pixels = int(height * num_pixels_per_meter)
    else:
        Z_HEIGHT = scene_info['zheight']
        max_x = max_x + scene_info['dx']
        min_x = min_x - scene_info['dx']
        if 'dy' in scene_info.keys():
            min_y =min_y-scene_info['dy']
            max_y =max_y+scene_info['dy']
        width = max_x - min_x
        height = max_y - min_y
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        num_pixels_per_meter = scene_info['num_pixels_per_meter']
        width_in_pixels = int(width * num_pixels_per_meter)
        height_in_pixels = int(height * num_pixels_per_meter)

    if get_data:
        return center_x,center_y,num_pixels_per_meter


    #print("pixels: WxH ", width_in_pixels, height_in_pixels)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width_in_pixels, height=height_in_pixels)
    # if isinstance(o3d_obj, o3d.geometry.TriangleMesh):
    #     o3d_obj.compute_vertex_normals()
    ctr = vis.get_view_control()
    camera_param = ctr.convert_to_pinhole_camera_parameters()
    # print(camera_param.intrinsic.intrinsic_matrix)
    # print(camera_param.extrinsic)

    vis.add_geometry(o3d_obj)
    ctr.set_zoom(0.5)

    f = Z_HEIGHT * num_pixels_per_meter
    new_intrinsic = np.array([[f, 0, (width_in_pixels/2.0)-0.5], [0, f, (height_in_pixels/2.0)-0.5], [0, 0, 1]])
    camera_param.intrinsic.intrinsic_matrix = new_intrinsic

    camera_param.extrinsic = compute_extrinsic_matrix(lookat_point=np.array([center_x, center_y, 0]),
                                                      camera_coords=np.array([center_x, center_y, Z_HEIGHT]))

    ctr.convert_from_pinhole_camera_parameters(camera_param)
    #camera_param = ctr.convert_to_pinhole_camera_parameters()


    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(filename)
    vis.destroy_window()
    return center_x,center_y,num_pixels_per_meter




def _render_3d_bev(ply_path, output_path, axis_align_matrix=None, resolution=100):
    mesh = o3d.io.read_triangle_mesh(ply_path)

    if axis_align_matrix is None:
        axis_align_matrix = np.eye(4)
    points = np.asarray(mesh.vertices)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)  # shape (n, 4)
    points = axis_align_matrix @ points.T  # shape (4, n)
    points = (points.T)[:, :3]
    mesh.vertices = o3d.utility.Vector3dVector(points)
    # Create a visualizer object
    visualizer = o3d.visualization.Visualizer()

    # Add the point cloud to the visualizer
    visualizer.create_window(visible=False, width=1024, height=768)
    visualizer.add_geometry(mesh)

    # Set the camera view point
    ctr = visualizer.get_view_control()
    pinhole_camera_param = ctr.convert_to_pinhole_camera_parameters()
    camera_position = np.array([0, 0, 1000])
    lookat_point = np.array([0, 0, 0])
    extrinsic_matrix = compute_extrinsic_matrix(lookat_point, camera_position)
    pinhole_camera_param.extrinsic = extrinsic_matrix
    ctr.convert_from_pinhole_camera_parameters(pinhole_camera_param)
    ctr.set_field_of_view(1)
    visualizer.poll_events()
    visualizer.update_renderer()

    visualizer.capture_screen_image(output_path)
    print("Image saved!")
    visualizer.destroy_window()


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

    if get_data:
        return  min_x, min_y, max_x, max_y, resolution

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


def render_bev_scannet(scene_id):
    """
    Render the bird's eye view for a given scene, specified by scene id
    scend-id_example: scene0000_00
    """
    point_cloud_path = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/ScanNet_v2/scans/{scene_id}/{scene_id}_vh_clean.ply"
    output_path = "testing_scannet.png"
    axis_align_matrix = read_axis_align_matrix(
        f"/mnt/hwfile/OpenRobotLab/maoxiaohan/ScanNet_v2/scans/{scene_id}/{scene_id}.txt", mode="scannet")
    _render_3d_bev(point_cloud_path, output_path, axis_align_matrix, resolution=80)


def render_bev_mp3d(scene_id):
    """
    Render the bird's eye view for a given scene, specified by scene id
    scene_id example: 1mp3d_0000_region0
    """
    point_cloud_path = '1mp3d_0000_region0/lidar/region0.ply'
    output_path = "testing_mp3d.png"
    axis_align_matrix = np.load(
        '1mp3d_0000_region0/lidar/rot_matrix.npy')
    print(axis_align_matrix)
    _render_3d_bev(point_cloud_path, output_path, axis_align_matrix, resolution=20)


def render_bev_3rscan(scene_id):
    """
    Render the bird's eye view for a given scene, specified by scene id
    scene_id example: 3rscan_0000
    """
    point_cloud_path = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/3rscan/data/{scene_id}/lidar/mesh.refined.v2.obj"
    output_path = "testing_3rscan.png"
    # axis_align_matrix = np.load(f"/mnt/hwfile/OpenRobotLab/maoxiaohan/matterport3d/matterport3d/data/{scene_id}/label/rot_matrix.npy")
    _render_3d_bev(point_cloud_path, output_path, resolution=80)


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import numpy as np
    import numpy as np

    mp3d_normal_dict = np.load('./mp3d_filter/normal.npy', allow_pickle=True).item()
    mp3d_abnormal_dict = np.load('./mp3d_filter/abnormal.npy', allow_pickle=True).item()
    _3rscan_normal_dict = np.load('./3rscan_filter/normal.npy', allow_pickle=True).item()
    _3rscan_abnormal_dict = np.load('./3rscan_filter/abnormal.npy', allow_pickle=True).item()
    scannet_normal_dict = np.load('./scannet_filter/normal.npy', allow_pickle=True).item()
    scannet_abnormal_dict = np.load('./scannet_filter/abnormal.npy', allow_pickle=True).item()

    anno_train = read_annotation_pickle('example_data/embodiedscan_infos_train_full.pkl')
    anno_val = read_annotation_pickle('example_data/embodiedscan_infos_val_full.pkl')
    dir_name = 'all_final'
    save_name = 'mp3d_ply1'
    #os.mkdir(dir_name)
    scene_id_list = []
    render_param_dict = {}

    for scene_id in anno_train.keys():
        scene_id_list.append(scene_id)
    for scene_id in anno_val.keys():
        scene_id_list.append(scene_id)


    for scene_id in tqdm(scene_id_list):
        if scene_id in anno_train:
            anno = anno_train[scene_id]
        elif scene_id in anno_val:
            anno = anno_val[scene_id]

        if scene_id[:6] == '3rscan':
            abnormal_dict = _3rscan_abnormal_dict
            normal_dict = _3rscan_normal_dict
        elif scene_id[:5] == '1mp3d':
            abnormal_dict = mp3d_abnormal_dict
            normal_dict = mp3d_normal_dict
        elif scene_id[:5] == 'scene':
            abnormal_dict = scannet_abnormal_dict
            normal_dict = scannet_normal_dict

        if scene_id in abnormal_dict.keys():
            scene_info = abnormal_dict[scene_id]
        elif scene_id in normal_dict.keys():
            scene_info = normal_dict[scene_id]
        else:
            print("wrong!!!")


        if scene_id[:6] == '3rscan':
            mesh = o3d.io.read_triangle_mesh(f'./3rscan_obj/{scene_id}/mesh.refined.v2.obj',enable_post_processing=True)
            matrix = np.asarray(anno['axis_align_matrix'])
            mesh = process_mesh(mesh, matrix)
        elif scene_id[:5] == '1mp3d':
            ply_name = scene_id.split('_')[2] + '.ply'
            mesh = o3d.io.read_triangle_mesh(f'./mp3d_ply1/{scene_id}/{ply_name}',enable_post_processing=True)
            matrix = np.asarray(anno['axis_align_matrix'])
            mesh = process_mesh(mesh, matrix)
        elif scene_id[:5] == 'scene':
            ply_name = scene_id + '_vh_clean_2.ply'
            mesh = o3d.io.read_triangle_mesh(f'./scene_ply/{scene_id}/{ply_name}',enable_post_processing=True)
            matrix = np.asarray(anno['axis_align_matrix'])
            mesh = process_mesh(mesh, matrix)
        if not os.path.exists(f'./{dir_name}/{scene_id}'):
            os.mkdir(f'./{dir_name}/{scene_id}')

        center_x,center_y,num_pixels_per_meter = take_bev_screenshot(mesh, f"./{dir_name}/{scene_id}/render.png",scene_info)
        render_param_dict[scene_id]= {}
        render_param_dict[scene_id]['center_x']=center_x
        render_param_dict[scene_id]['center_y'] = center_y
        render_param_dict[scene_id]['num_pixels_per_meter'] = num_pixels_per_meter

    np.save('all_render_param.npy',render_param_dict)
