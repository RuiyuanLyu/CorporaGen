import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

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


def render_3d_top_down_view(point_cloud_path="example_data/lidar/main.pcd"):
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    # Create a visualizer object
    visualizer = o3d.visualization.Visualizer()

    # Add the point cloud to the visualizer
    visualizer.create_window(visible=False, width=1024, height=768)
    visualizer.add_geometry(point_cloud)

    # Set the camera view point
    view_control = visualizer.get_view_control()
    pinhole_camera_param = view_control.convert_to_pinhole_camera_parameters()
    camera_position = np.array([0, 0, 8])
    lookat_point = np.array([0, 0, 0])
    extrinsic_matrix = compute_extrinsic_matrix(lookat_point, camera_position)
    pinhole_camera_param.extrinsic = extrinsic_matrix
    view_control.convert_from_pinhole_camera_parameters(pinhole_camera_param)
    visualizer.poll_events()
    visualizer.update_renderer()

    visualizer.capture_screen_image("rendered.png")
    print("Image saved!")
    visualizer.destroy_window()
    

def render_bev(point_cloud_path="example_data/lidar/main.pcd", output_path='point_cloud_top_view.png'):
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    sorted_indices = np.argsort(points[:, 2])
    points = points[sorted_indices]
    colors = colors[sorted_indices]

    min_x, min_y, _ = np.min(points, axis=0)
    max_x, max_y, _ = np.max(points, axis=0)

    image_width = int((max_x - min_x) * 20) + 1
    image_height = int((max_y - min_y) * 20) + 1

    projection_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for point, color in zip(points, colors):
        x, y, _ = point
        pixel_x = int((x - min_x) * 20)
        pixel_y = int((y - min_y) * 20)
        projection_image[pixel_y, pixel_x] = (color * 255).astype(np.uint8)
    
    plt.imshow(projection_image, origin='lower')
    plt.axis('off')  # hide axis ticks and labels
    cv2.imwrite(output_path,projection_image)
    print("Successfully saved to", output_path)
    plt.show()
    
    
if __name__ == '__main__':
    # render_3d_top_down_view()
    point_cloud_path="example_data/lidar/main.pcd"
    output_path='example_data/anno_lang/point_cloud_top_view.png'
    render_bev(point_cloud_path, output_path)