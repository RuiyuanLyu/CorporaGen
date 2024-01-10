import os
import numpy as np
import json
from tqdm import tqdm
import cv2
import mmengine


def draw_depth(depth_img_path, extrinsic, depth_instrincs, shift=1000):
    extrinsics = np.linalg.inv(extrinsic)
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    depth_img = depth_img.astype(np.float32) / shift
    depth_point = []
    us, vs = np.where(depth_img > 0)
    ds = depth_img[us, vs]
    points = np.stack([vs * ds, us * ds, ds, np.ones((us.shape[0],))], axis=1)
    # print(points.shape)
    xyzs = np.linalg.inv(extrinsics) @ np.linalg.inv(depth_instrincs) @ points.transpose()
    xyzs = xyzs.transpose()
    depth_point = xyzs[:,:3]
    return depth_point

def splitnames(name):
    assert name[-6] == '_'
    assert name.find('_') == 32
    cam_name = name[:32]
    cam_pose = name[-7:-4]
    return cam_name, cam_pose

def read_region(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 11 regions = []
    region_xyxy = []
    for line in lines:
        if line[0] == 'R':
            a = line.split()
            index = int(a[1])
            assert index == len(region_xyxy)
            x1, y1, z1, x2, y2, z2 = float(a[9]), float(a[10]), float(a[11]), float(a[12]), float(a[13]), float(a[14])
            region_xyxy.append([x1, y1, z1, x2, y2, z2])
            # 11 regions.append([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2, x2-x1, y2-y1, z2-z1, 0, 0, 0])
    
    return np.array(region_xyxy)

def read_intrinsic(path):
    a = np.loadtxt(path)
    intrinsic = np.identity(4, dtype=float)
    intrinsic[0][0] = a[2]  # fx
    intrinsic[1][1] = a[3]  # fy
    intrinsic[0][2] = a[4]  # cx
    intrinsic[1][2] = a[5]  # cy
    return intrinsic

def handle_one_scene(scene):
    root = '/mnt/petrelfs/share_data/lipeisen/matterport3d/scans'
    out_dir = './select_cam'
    xyxy = read_region(os.path.join(root, scene, 'house_segmentations', scene, 'house_segmentations', f'{scene}.house'))
    num_regions = xyxy.shape[0]
    results = [{'name' : f'region{x}', 'cams' : []} for x in range(num_regions)]
    
    # o3d.visualization.draw_geometries([pcd, frame] + regions)
    camera_paths = os.listdir(os.path.join(root, scene, 'matterport_camera_poses', scene, 'matterport_camera_poses'))
    
    cameras = []
    for camera in camera_paths:
        try:
            camera_path = os.path.join(root, scene, 'matterport_camera_poses', scene, 'matterport_camera_poses', camera)
            camera_name, cam_pose_id = splitnames(camera)
            depth_path = os.path.join(root, scene, 'matterport_depth_images', scene, 'matterport_depth_images', f'{camera_name}_d{cam_pose_id}.png')
            intrinsic = read_intrinsic(os.path.join(root, scene, 'matterport_camera_intrinsics', scene, 'matterport_camera_intrinsics', f'{camera_name}_intrinsics_{cam_pose_id[0]}.txt'))
            depth_points = draw_depth(depth_path, np.loadtxt(camera_path), intrinsic, shift=4000)
            n = depth_points.shape[0]
            insided_region = []
            for i in range(num_regions):
                inside = (depth_points[:,0] > xyxy[i][0]) & (depth_points[:,1] > xyxy[i][1]) & (depth_points[:,2] > xyxy[i][2]) & (depth_points[:,0] < xyxy[i][3]) & (depth_points[:,1] < xyxy[i][4]) & (depth_points[:,2] < xyxy[i][5])
                count = np.sum(inside)
                if count > 0.1 * n:
                    results[i]['cams'].append(camera)
                    # insided_region.append(i)
        except Exception as e:
            print(e)
            continue
        
    with open(os.path.join(out_dir, f'{scene}.json'), 'w') as f:
        json.dump(results, f)
        # _, camera_geo = draw_camera(np.loadtxt(camera_path))
        # cameras.append(camera_geo)
        # o3d.visualization.draw_geometries([pcd, frame, depth, camera_geo] + insided_region)
    
    # o3d.visualization.draw_geometries([pcd, frame] + cameras + bbox_geo)
    
def main():
    root = '/mnt/petrelfs/share_data/lipeisen/matterport3d/scans'
    scans = os.listdir(root)
    mmengine.utils.track_parallel_progress(handle_one_scene, scans, 8)

main()
