import os
import json
import open3d as o3d
import numpy as np
import shutil
import argparse

import mmengine
from tqdm import tqdm
from utils import splitnames, read_intrinsic, read_region, depthimg_points, cp, get_axis_align, json_move
from sam_utils import generate_sam_box, is_in_cone_by_camera_params, coverage_solver

def prepare_sam_mapping():
    mapping = dict()
    root1 = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/matterport3d_sam'
    dirs = os.listdir(root1)
    for d in dirs:
        files = os.listdir(os.path.join(root1, d, 'sam_2dmask'))
        for file in files:
            mapping[file] = os.path.join(root1, d, 'sam_2dmask', file)
    
    root2 = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/sam_new'
    dirs = os.listdir(root2)
    for d in dirs:
        files = os.listdir(os.path.join(root2, d))
        for file in files:
            mapping[file] = os.path.join(root2, d, file)
    return mapping

ROOT = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/scans'
OUTROOT = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/regions'
BUILDROOT = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/data2'
sam_path_mapping = prepare_sam_mapping()
OUT_PLY = True
OUT_OBJ = True
OUT_IMG = True
OUT_SAM = True
OUT_BUILD = True

def generate_one_scene(params):
    print(params)
    scene, mapped_scene = params
    
    if os.path.exists(f'rename/{scene}.json'):
        with open(f'rename/{scene}.json', 'r') as f:
            camera_mapping = json.load(f)
    else:
        camera_mapping = dict()
        
    root = ROOT
    extrinsic_dir = os.path.join(root, scene, 'matterport_camera_poses')
    intrinsic_dir = os.path.join(root, scene, 'matterport_camera_intrinsics')
    colorimg_dir = os.path.join(root, scene, 'matterport_color_images')
    depthimg_dir = os.path.join(root, scene, 'matterport_depth_images')
    house_info_dir = os.path.join(root, scene, 'house_segmentations')
    region_dir = os.path.join(root, scene, 'region_segmentations')
    
    # calc regions
    regions = os.listdir(region_dir)
    regions = [x[:-4] for x in regions if x[-4:] == '.ply']
    regions.sort()
    
    if OUT_PLY:
        for region in regions:
            mapped_region = mapped_scene + '_' + region
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'lidar'))
            ply = o3d.io.read_point_cloud(os.path.join(region_dir, region + '.ply'), format='ply')
            o3d.io.write_point_cloud(os.path.join(OUTROOT, mapped_region, 'lidar', 'main.pcd'), ply)
        print('PLY complete')
    # split object region
    region_bound, objects, belongs, labels = read_region(os.path.join(house_info_dir, f'{scene}.house'))
    num_regions = region_bound.shape[0]
    
    if OUT_OBJ:
        obj_json = []
        for _ in range(num_regions):
            obj_json.append([])
        
        for i in range(len(belongs)):
            belong = belongs[i]
            label = labels[i]
            obj_idx = len(obj_json[belong]) + 1
            obj_json[belong].append({"obj_id": str(obj_idx), "obj_type": label,
                                    "psr" : {"position" : {"x" : objects[i][0], "y" : objects[i][1], "z": objects[i][2]},
                                            "scale" : {"x" : objects[i][3], "y" : objects[i][4], "z": objects[i][5]},
                                            "rotation": {"x" : objects[i][6], "y" : objects[i][7], "z": objects[i][8]}
                                            }})
        
        for i in range(num_regions):
            region = f'region{i}'
            # 1 region = regions[i]
            if region not in regions:
                assert len(obj_json[i]) == 0
                continue
            mapped_region = mapped_scene + '_' + region
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'label'))
            with open(os.path.join(OUTROOT, mapped_region, 'label', 'main.json'), 'w') as f:
                json.dump(obj_json[i], f)
        
        print('OBJ complete')
    
    if OUT_IMG:
        for i in range(num_regions):
            region = f'region{i}'
            if region not in regions:
                continue
            mapped_region = mapped_scene + '_' + region
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'color_images'))
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'depth_images'))
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'camera_poses'))
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'camera_intrinsics'))
        
        region_cameras = [[] for i in range(num_regions)]
        
        camera_paths = os.listdir(extrinsic_dir)
        camera_paths.sort()
        for camera_single_path in camera_paths:
            camera_path = os.path.join(extrinsic_dir, camera_single_path)
            cam_id, cam_pose_id = splitnames(camera_single_path)
            if cam_id in camera_mapping:
                mapped_cam_id = camera_mapping[cam_id]
            else:
                mapped_cam_id = len(list(camera_mapping.keys()))
                mapped_cam_id = str(mapped_cam_id).zfill(4)
                camera_mapping[cam_id] = mapped_cam_id
            
            extrinsic = np.loadtxt(camera_path)
            # camera_location = [extrinsic[0][3], extrinsic[1][3], extrinsic[2][3]]
            cam_intrin_id = cam_pose_id.split('_')[0]
            depth_path = os.path.join(depthimg_dir, f'{cam_id}_d{cam_pose_id}.png')
            width, height, intrinsic = read_intrinsic(os.path.join(intrinsic_dir, f'{cam_id}_intrinsics_{cam_pose_id[0]}.txt'))
            depth_points = depthimg_points(depth_path, extrinsic, intrinsic, shift=4000)
            n = depth_points.shape[0]
            for i in range(num_regions):
                region = f'region{i}'
                if region not in regions:
                    continue
                mapped_region = mapped_scene + '_' + region
                inside = ((depth_points[:,0] > region_bound[i][0]) & (depth_points[:,1] > region_bound[i][1]) & (depth_points[:,2] > region_bound[i][2]) &
                          (depth_points[:,0] < region_bound[i][3]) & (depth_points[:,1] < region_bound[i][4]) & (depth_points[:,2] < region_bound[i][5]))
                count = np.sum(inside)
                if count > 0.1 * n:
                    region_cameras[i].append((cam_id, cam_pose_id))
                    cp(os.path.join(colorimg_dir, f'{cam_id}_i{cam_pose_id}.jpg'), os.path.join(OUTROOT, mapped_region, 'color_images', f'{cam_id}_i{cam_pose_id}.jpg'))
                    cp(os.path.join(depthimg_dir, f'{cam_id}_d{cam_pose_id}.png'), os.path.join(OUTROOT, mapped_region, 'depth_images', f'{cam_id}_d{cam_pose_id}.png'))
                    cp(os.path.join(extrinsic_dir, f'{cam_id}_pose_{cam_pose_id}.txt'), os.path.join(OUTROOT, mapped_region, 'camera_poses', f'{cam_id}_pose_{cam_pose_id}.txt'))
                    cp(os.path.join(intrinsic_dir, f'{cam_id}_intrinsics_{cam_intrin_id}.txt'), os.path.join(OUTROOT, mapped_region, 'camera_intrinsics', f'{cam_id}_intrinsics_{cam_intrin_id}.txt'))
        
        with open(os.path.join(root, scene, 'region_cameras.json'), 'w') as f:
            json.dump(region_cameras, f)
        
        with open(f'rename/{scene}.json', 'w') as f:
            json.dump(camera_mapping, f)
        print('IMG complete')
    
    if OUT_SAM:
        assert os.path.exists(os.path.join(root, scene, 'region_cameras.json'))
        with open(os.path.join(root, scene, 'region_cameras.json'), 'r') as f:
            region_cameras = json.load(f)
        
        for i in range(num_regions):
            region = f'region{i}'
            if region not in regions:
                continue
            mapped_region = mapped_scene + '_' + region
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'sam'))
            os.makedirs(os.path.join(OUTROOT, mapped_region, 'sam_2dmask'))
            ply = o3d.io.read_point_cloud(os.path.join(region_dir, region + '.ply'), format='ply')
            low_pcd = ply.voxel_down_sample(voxel_size=0.05)
            points = np.asarray(low_pcd.points)
            boolean_vecs = []
            for cam_id, cam_pose_id in region_cameras[i]:
                extrinsic = np.loadtxt(os.path.join(extrinsic_dir, f'{cam_id}_pose_{cam_pose_id}.txt'))
                cam_intrin_id = cam_pose_id.split('_')[0]
                width, height, intrinsic = read_intrinsic(os.path.join(intrinsic_dir, f'{cam_id}_intrinsics_{cam_intrin_id}.txt'))
                boolean_vec = is_in_cone_by_camera_params(points, intrinsic, np.linalg.inv(extrinsic), width, height)
                boolean_vecs.append(boolean_vec)
            
            boolean_vecs = np.array(boolean_vecs)
            selected_camera_indices = coverage_solver(boolean_vecs, 0.95)
            
            for selected in selected_camera_indices:
                cam_id, cam_pose_id = region_cameras[i][selected]
                if not (f'{cam_id}_{cam_pose_id}.png' in sam_path_mapping and f'{cam_id}_{cam_pose_id}.json' in sam_path_mapping):
                    error_color_img_path = os.path.join(ROOT, scene, 'matterport_color_images', f'{cam_id}_i{cam_pose_id}.jpg')
                    print(error_color_img_path, f'{cam_id}_{cam_pose_id}.png not exists')
                    with open('error.txt','a') as f:
                        print(error_color_img_path, file=f)
                    continue
                generate_sam_box(
                    sam_img_path=sam_path_mapping[f'{cam_id}_{cam_pose_id}.png'],
                    depth_img_path=os.path.join(depthimg_dir, f'{cam_id}_d{cam_pose_id}.png'),
                    sam_json_path=sam_path_mapping[f'{cam_id}_{cam_pose_id}.json'],
                    intrinsic_path=os.path.join(intrinsic_dir, f'{cam_id}_intrinsics_{cam_intrin_id}.txt'),
                    extrinsic_path=os.path.join(extrinsic_dir, f'{cam_id}_pose_{cam_pose_id}.txt'),
                    out_sam_box_path=os.path.join(OUTROOT, mapped_region, 'sam', f'{cam_id}_{cam_pose_id}.json'),
                    out_sam_img_path=os.path.join(OUTROOT, mapped_region, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.jpg'),
                    out_sam_json_path=os.path.join(OUTROOT, mapped_region, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.json')
                )
        print('SAM complete')
    
    if OUT_BUILD:
        for i in range(num_regions):
            region = f'region{i}'
            if region not in regions:
                continue
            mapped_region = mapped_scene + '_' + region
            os.makedirs(os.path.join(BUILDROOT, mapped_region, 'label'))
            os.makedirs(os.path.join(BUILDROOT, mapped_region, 'lidar'))
            os.makedirs(os.path.join(BUILDROOT, mapped_region, 'sam'))
            os.makedirs(os.path.join(BUILDROOT, mapped_region, 'sam_2dmask'))
            os.makedirs(os.path.join(BUILDROOT, 'posed_images', mapped_region))
            
            ply = o3d.io.read_point_cloud(os.path.join(region_dir, region + '.ply'), format='ply')
            axis_align_mat = get_axis_align(ply)
            ply.transform(axis_align_mat)
            o3d.io.write_point_cloud(os.path.join(BUILDROOT, mapped_region, 'lidar', 'main.pcd'), ply)
            
            json_move(os.path.join(OUTROOT, mapped_region, 'label', 'main.json'), os.path.join(BUILDROOT, mapped_region, 'label', 'main.json'), axis_align_mat)
            np.save(os.path.join(BUILDROOT, mapped_region, 'label', 'rot_matrix.npy'), axis_align_mat)
            
            camera_paths = os.listdir(os.path.join(OUTROOT, mapped_region, 'camera_poses'))
            camera_paths.sort()
            
            for camera_single_path in camera_paths:
                cam_id, cam_pose_id = splitnames(camera_single_path)
                assert cam_id in camera_mapping
                mapped_cam_id = camera_mapping[cam_id]
                cam_intrin_id = cam_pose_id.split('_')[0]
                
                cp(os.path.join(OUTROOT, mapped_region, 'camera_poses', f'{cam_id}_pose_{cam_pose_id}.txt'), os.path.join(BUILDROOT, 'posed_images', mapped_region, f'{mapped_cam_id}_{cam_pose_id}.txt'))
                cp(os.path.join(OUTROOT, mapped_region, 'color_images', f'{cam_id}_i{cam_pose_id}.jpg'), os.path.join(BUILDROOT, 'posed_images', mapped_region, f'{mapped_cam_id}_{cam_pose_id}.jpg'))
                cp(os.path.join(OUTROOT, mapped_region, 'depth_images', f'{cam_id}_d{cam_pose_id}.png'), os.path.join(BUILDROOT, 'posed_images', mapped_region, f'{mapped_cam_id}_{cam_pose_id}.png'))
                _, __, intrinsic = read_intrinsic(os.path.join(OUTROOT, mapped_region, 'camera_intrinsics', f'{cam_id}_intrinsics_{cam_intrin_id}.txt'))
                np.savetxt(os.path.join(BUILDROOT, 'posed_images', mapped_region, f'intrinsic_{mapped_cam_id}_{cam_intrin_id}.txt'), intrinsic, fmt='%.3f')
            
            sam_paths = os.listdir(os.path.join(OUTROOT, mapped_region, 'sam'))
            sam_paths.sort()
            
            for sam_single_path in sam_paths:
                cam_id, cam_pose_id = splitnames(sam_single_path)
                assert cam_id in camera_mapping
                mapped_cam_id = camera_mapping[cam_id]
                
                json_move(os.path.join(OUTROOT, mapped_region, 'sam', f'{cam_id}_{cam_pose_id}.json'), os.path.join(BUILDROOT, mapped_region, 'sam', f'{mapped_cam_id}_{cam_pose_id}.json'), axis_align_mat)
                cp(os.path.join(OUTROOT, mapped_region, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.json'), os.path.join(BUILDROOT, mapped_region, 'sam_2dmask', f'{mapped_cam_id}_{cam_pose_id}.json'))
                cp(os.path.join(OUTROOT, mapped_region, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.jpg'), os.path.join(BUILDROOT, mapped_region, 'sam_2dmask', f'{mapped_cam_id}_{cam_pose_id}.jpg'))
        
        print('BUILD complete')
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', required=True, type=int)
args = parser.parse_args()

with open('scene_mapping.json', 'r') as f:
    scene_mapping = json.load(f)
# scenes = os.listdir(ROOT)
scenes_ori = ["Vvot9Ly1tCj", "cV4RVeZvu5T", "aayBHfsNo7d", "D7G3Y4RVNrH", "ULsKaCPVFJR", "YFuZgdQ5vWj", "UwV83HsGsw3", "VFuaQ6m2Qom", "2n8kARJN3HM", "pRbA3pwrgk9", "ac26ZMwG7aT", "pa4otMbVnkk", "rPc6DW4iMge", "5LpN3gDmAk7", "yqstnuAEVhm", "Z6MFQCViBuw", "gTV8FGcVJC9", "5ZKStnWn8Zo", "EDJbREhghzL", "kEZ7cmS4wCh", "jh4fc5c5qoQ", "vyrNrziPKCB", "fzynW3qQPVF", "S9hNv5qa7GM", "YVUC4YcDtcY", "8WUmhLawc2A", "gZ6f7yhEvPG", "gxdoqLR6rwA", "E9uDoFAP3SH", "RPmz2sHmrrY", "x8F5xyUWy9e", "7y3sRwLe3Va", "rqfALeAoiTq", "2t7WUuJeko7", "SN83YJsR3w2", "oLBMNvg9in8", "1pXnuDYAj8r", "q9vSo1VnCiC", "Pm6F8kyY3z2", "ARNzJeq3xxb", "YmJkqBEsHnH", "VVfe2KiqLaN", "2azQ1b91cZZ", "JeFG25nYj2p", "B6ByNegPMKs", "pLe4wQe7qrG", "i5noydFURQK", "17DRP5sb8fy", "ur6pFq6Qu1A", "wc2JMjhGNzB", "JF19kD82Mey", "r1Q1Z4BcV1o", "VLzqgDo317F", "r47D5H71a5s", "VzqfbhrpDEA", "qoiz87JEwZ2", "JmbYfDe2QKZ", "b8cTxDM8gDG", "sKLMLpTHeUy", "759xd9YjKW5", "5q7pvUzZiYa", "X7HyMhZNoso", "dhjEzFoUFzH", "s8pcmisQ38h", "gYvKGZ5eRqb", "WYY7iVyf5p8", "jtcxE69GiFV", "uNb9QFRL6hY", "mJXqzFtmKg4", "TbHJrupSAjP", "Uxmj2M2itWa", "EU6Fwq7SyZv", "QUCTc6BB5sX", "82sE5b5pLXE", "HxpKQynjfin", "Vt2qJdWjCF2", "V2XKFyX4ASd", "GdvgFV5R1Z5", "ZMojNkEp431", "XcA2TqTSSAj", "sT4fr6TAbpF", "29hnd4uzFmX", "PX4nDJXEHrG", "1LXtFkjw3qL", "8194nk5LbLH", "e9zR4mvMWw7", "zsNo4HB9uLZ", "p5wJjkQkbXX", "D7N2EKCX4Sj", "PuKPg4mmafe"]
scene_l = args.num * 16
scene_r = (args.num + 1) * 16
scenes = scenes_ori[scene_l : scene_r]
print(scenes)

for scene in scenes:
    assert scene in scene_mapping
params = [(scene, scene_mapping[scene]) for scene in scenes]

# for param in tqdm(params):
#     generate_one_scene(param)
#     break
mmengine.utils.track_parallel_progress(generate_one_scene, params, len(scenes))