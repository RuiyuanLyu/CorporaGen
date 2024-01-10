import open3d as o3d
import numpy as np

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


# Load the point cloud from the .pcd file
point_cloud = o3d.io.read_point_cloud("example_data/lidar/main.pcd")

# Create a visualizer object
visualizer = o3d.visualization.Visualizer()

# Add the point cloud to the visualizer
visualizer.create_window(visible=False, width=1024, height=768)
visualizer.add_geometry(point_cloud)

# View the point cloud
# visualizer.run()

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

# Render the image
visualizer.capture_screen_image("rendered.png")
print("Image saved!")
# Close the visualizer window
visualizer.destroy_window()
