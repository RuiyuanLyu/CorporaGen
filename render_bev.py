import open3d as o3d # version 0.16.0. NEVER EVER use version 0.17.0 or later, or you will get fucked by vis control.
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils_read import read_axis_align_matrix, load_json, reverse_121_mapping

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

def take_bev_screenshot(o3d_obj, filename):
    min_x, min_y, min_z = np.min(o3d_obj.vertices, axis=0)
    max_x, max_y, max_z = np.max(o3d_obj.vertices, axis=0)
    print("x range: ", min_x, max_x)
    print("y range: ", min_y, max_y)
    print("z range: ", min_z, max_z)
    width = max_x - min_x
    height = max_y - min_y
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    num_pixels_per_meter = 200
    width_in_pixels = int(width * num_pixels_per_meter)
    height_in_pixels = int(height * num_pixels_per_meter)
    print("pixels: WxH ", width_in_pixels, height_in_pixels)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width_in_pixels, height=height_in_pixels)
    # if isinstance(o3d_obj, o3d.geometry.TriangleMesh):
    #     o3d_obj.compute_vertex_normals()
    ctr = vis.get_view_control()
    camera_param = ctr.convert_to_pinhole_camera_parameters()
    print(camera_param.intrinsic.intrinsic_matrix)
    print(camera_param.extrinsic)

    vis.add_geometry(o3d_obj)
    ctr.set_zoom(0.5)
    Z_HEIGHT = 50
    f = Z_HEIGHT * num_pixels_per_meter
    new_intrinsic = np.array([[f, 0, (width_in_pixels-1)/2], [0, f, (height_in_pixels-1)/2], [0, 0, 1]])
    camera_param.intrinsic.intrinsic_matrix = new_intrinsic
    
    camera_param.extrinsic = compute_extrinsic_matrix(lookat_point=np.array([center_x, center_y, 0]), camera_coords=np.array([center_x, center_y, Z_HEIGHT]))
    print(camera_param.intrinsic.intrinsic_matrix)
    print(camera_param.extrinsic)
    ctr.convert_from_pinhole_camera_parameters(camera_param)
    camera_param = ctr.convert_to_pinhole_camera_parameters()
    print(camera_param.intrinsic.intrinsic_matrix)
    print(camera_param.extrinsic)

    vis.poll_events()    
    vis.update_renderer()
    import time; time.sleep(2)
    vis.capture_screen_image(filename)
    vis.destroy_window()

def _render_3d_bev(ply_path, output_path, axis_align_matrix=None, resolution=100):
    mesh = o3d.io.read_triangle_mesh(ply_path)

    if axis_align_matrix is None:
        axis_align_matrix = np.eye(4)
    points = np.asarray(mesh.vertices)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)  # shape (n, 4)
    points = axis_align_matrix @ points.T # shape (4, n)
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
    

def _render_2d_bev(point_cloud_path, output_path, axis_align_matrix=None, resolution=80, roof_percentage=0.0):
    if axis_align_matrix is None:
        axis_align_matrix = np.eye(4)
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    points = np.asarray(point_cloud.points)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)  # shape (n, 4)
    points = axis_align_matrix @ points.T # shape (4, n)
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
    cv2.imwrite(output_path,projection_image)
    print("Successfully saved to", output_path)
    # plt.show()

def render_bev_scannet(scene_id):
    """
    Render the bird's eye view for a given scene, specified by scene id
    scend-id_example: scene0000_00
    """
    point_cloud_path = f"/mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2/scans/{scene_id}/{scene_id}_vh_clean.ply"
    output_path = "testing_scannet.png"
    axis_align_matrix = read_axis_align_matrix(f"/mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2/scans/{scene_id}/{scene_id}.txt", mode="scannet")
    _render_3d_bev(point_cloud_path, output_path, axis_align_matrix, resolution=80)

def render_bev_mp3d(scene_id):
    """
    Render the bird's eye view for a given scene, specified by scene id
    scene_id example: 1mp3d_0000_region0
    """
    prefix_str, house_idx, reg_idx = scene_id.split("_") # prefix_str is always "1mp3d"
    scene_mapping_dict = load_json("/mnt/petrelfs/share_data/maoxiaohan/matterport3d/meta_data/scene_mapping.json")
    scene_mapping_dict = reverse_121_mapping(scene_mapping_dict)
    key = prefix_str + "_" + house_idx
    scene_hash = scene_mapping_dict[key]
    point_cloud_path = f"/mnt/petrelfs/share_data/maoxiaohan/matterport3d/scans/{scene_hash}/region_segmentations/{reg_idx}.ply"
    output_path = "testing_mp3d.png"
    axis_align_matrix = np.load(f"/mnt/petrelfs/share_data/maoxiaohan/matterport3d/matterport3d/data/{scene_id}/label/rot_matrix.npy")
    print(axis_align_matrix)
    _render_3d_bev(point_cloud_path, output_path, axis_align_matrix, resolution=80, roof_percentage=0.2)

def render_bev_3rscan(scene_id):
    """
    Render the bird's eye view for a given scene, specified by scene id
    scene_id example: 3rscan_0000
    """
    point_cloud_path = f"/mnt/petrelfs/share_data/maoxiaohan/3rscan/data/{scene_id}/lidar/mesh.refined.v2.obj"
    output_path = "testing_3rscan.png"
    # axis_align_matrix = np.load(f"/mnt/petrelfs/share_data/maoxiaohan/matterport3d/matterport3d/data/{scene_id}/label/rot_matrix.npy")
    _render_3d_bev(point_cloud_path, output_path, resolution=80)


if __name__ == "__main__":
    # test_load_point_cloud(testing_scannet_file)
    # test_load_point_cloud(testing_mp3d_file)
    mesh = test_load_mesh(testing_3rscan_dir)
    take_bev_screenshot(mesh, "testing.png")