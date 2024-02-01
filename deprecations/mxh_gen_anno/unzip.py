import zipfile
import os
import mmengine

todo_files = ['house_segmentations.zip',
              'matterport_camera_intrinsics.zip',
              'matterport_camera_poses.zip',
              'matterport_color_images.zip',
              'matterport_depth_images.zip',
              'region_segmentations.zip'
              ]
root = './raw_data'
output = './scans'

def unzip_scene(scene):
    for file in todo_files:
        filepath = os.path.join(root, scene, file)
        with zipfile.ZipFile(filepath, 'r') as zp:
            zp.extractall(output)

scenes = os.listdir(root)
mmengine.utils.track_parallel_progress(unzip_scene, scenes, 8)