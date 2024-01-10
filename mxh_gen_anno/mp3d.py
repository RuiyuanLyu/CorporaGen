import json
import numpy as np
import os
import shutil
import open3d as o3d
import cv2
import mmengine
from tqdm import tqdm

def splitnames(name):
    name = name.split('.')[0]
    assert name[-2] == '_'
    assert name.find('_') == 32
    cam_name = name[:32]
    if '_intrinsics_' in name:
        cam_pose = name.split('_')[-1]
    else:
        cam_pose = name[-3:]
    return cam_name, cam_pose

def read_intrinsic(path):
    a = np.loadtxt(path)
    intrinsic = np.identity(4, dtype=float)
    intrinsic[0][0] = a[2]  # fx
    intrinsic[1][1] = a[3]  # fy
    intrinsic[0][2] = a[4]  # cx
    intrinsic[1][2] = a[5]  # cy
    return a[0], a[1], intrinsic

def world_coord_to_camera_coord(points, extrinsic):
    """
        points: n x 3 in world coordinate
        extrinsic: 4 x 4
        returns: n x 3 in camera coordinate
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    points_in_camera = rotation @ points.T + translation.reshape([-1, 1])
    return points_in_camera.T

def camera_coord_to_2d_image(points, intrinsic):
    """
        points: n x 3 in camera coordinate
        intrinsic: dict
        returns: n x 2 in 2d image
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    new_x = points[:, 0] * fx / points[:, 2] + cx
    new_y = points[:, 1] * fy / points[:, 2] + cy
    points_2d = np.hstack([new_x.reshape([-1, 1]), new_y.reshape([-1, 1])])
    return points_2d

def is_in_cone_by_camera_params(points, intrinsic, extrinsic, width, height):
    """
        points: n x 3 in world coordinate
        intrinsic: dict
        extrinsic: 4 x 4    w2c
        returns: n x 1 bool array
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    points_in_camera = world_coord_to_camera_coord(points, extrinsic)
    # eliminate those points behind the camera
    output_indices = points_in_camera[:, 2] > 0
    points_2d = camera_coord_to_2d_image(points_in_camera, intrinsic)
    # eliminate those points outside the image
    xmax, ymax = width, height
    output_indices = np.logical_and(output_indices, points_2d[:, 0] >= 0)
    output_indices = np.logical_and(output_indices, points_2d[:, 0] < xmax)
    output_indices = np.logical_and(output_indices, points_2d[:, 1] >= 0)
    output_indices = np.logical_and(output_indices, points_2d[:, 1] < ymax)
    return output_indices

def coverage_solver(boolean_vecs, target_coverage):
    """
        boolean_vecs: n boolean vectors of length m, shape (n, m)
        target_coverage: float
    """
    if boolean_vecs.shape[0] == 0:
        return []
    num_vecs, vec_len = boolean_vecs.shape
    can_be_covered = np.sum(boolean_vecs, axis=0) > 0
    already_covered = np.zeros(vec_len, dtype=bool)
    num_can_be_covered = np.sum(can_be_covered)
    # print("can be covered: {}/{}".format(num_can_be_covered, vec_len))
    sol = []
    while np.sum(already_covered) < target_coverage * num_can_be_covered:
        # find the camera that can cover the most points
        num_covered = np.sum(boolean_vecs[:, can_be_covered], axis=1)
        best_cam_idx = np.argmax(num_covered)
        sol.append(best_cam_idx)
        already_covered = np.logical_or(already_covered, boolean_vecs[best_cam_idx])
        can_be_covered = np.logical_and(can_be_covered, np.logical_not(already_covered))
        # print("already covered: {}/{}".format(np.sum(already_covered), num_can_be_covered))
    # print("solution: {}".format(sol))
    return sol

def generate_sam_box(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, extrinsic_path, out_sam_box_path, out_sam_json_path, out_sam_img_path, axis_matrix):
    intrinsic = np.loadtxt(intrinsic_path)
    extrinsic = np.loadtxt(extrinsic_path)
    img = cv2.imread(sam_img_path)
    depth = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 4000.0
    with open(sam_json_path, 'r') as f:
        sam_id_json = json.load(f)
    sam_id = np.array(sam_id_json, dtype=int)
    
    group_id = np.unique(sam_id)
    boxes = []
    count = 0
    mapping = dict()
    mapping[-1] = -1
    for id in group_id:
        if id == -1:
            continue
        mapping[id] = count
        points = []
        us, vs = np.where(sam_id == id)
        us = (us / img.shape[0] * depth.shape[0]).astype(int)
        vs = (vs / img.shape[1] * depth.shape[1]).astype(int)
        ds = depth[us, vs]
        greater_than_0 = np.where(ds > 0)
        points = np.stack([vs * ds, us * ds, ds, np.ones((us.shape[0],))], axis=1)
        points = points[greater_than_0]
        
        # for _ in range(us.shape[0]):
        #     u = us[_]
        #     v = vs[_]
        #     d = depth[u][v] / 1000.0
        #     if d > 0:
        #         u = u * d
        #         v = v * d
        #         points.append([v,u,d,1])
        # points = np.array(points)
        if points.shape[0] == 0:
            mapping[id] = -1
            continue
        xyzs = (extrinsic) @ np.linalg.inv(intrinsic) @ points.transpose()
        xyzs = xyzs.transpose()
        xyzs = xyzs[:,:3]
        
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(xyzs)
        # o3d.visualization.draw_geometries([textured_mesh, pcd2])
        # pcd2.paint_uniform_color((1, 0, 0))
        
        xmin, xmax = np.min(xyzs[:,0]), np.max(xyzs[:,0])
        ymin, ymax = np.min(xyzs[:,1]), np.max(xyzs[:,1])
        zmin, zmax = np.min(xyzs[:,2]), np.max(xyzs[:,2])
        
        box = {'position': {'x' : (xmin+xmax) / 2.0 + axis_matrix[0][3], 
                            'y' : (ymin+ymax) / 2.0 + axis_matrix[1][3],
                            'z' : (zmin+zmax) / 2.0 + axis_matrix[2][3]},
                'rotation': {'x' : 0, 
                            'y' : 0,
                            'z' : 0},
                'scale': {'x' : xmax - xmin, 
                            'y' : ymax - ymin,
                            'z' : zmax - zmin}
                }
        boxes.append(box)
        count += 1
    
    with open(out_sam_box_path, 'w') as f:
        json.dump(boxes, f)
    
    sam_id_new = np.zeros_like(sam_id)
    for i in range(sam_id.shape[0]):
        for j in range(sam_id.shape[1]):
            sam_id_new[i][j] = mapping[sam_id[i][j]]
    
    with open(out_sam_json_path, 'w') as f:
        json.dump(sam_id_new.tolist(), f)
    
    cv2.imwrite(out_sam_img_path, img)

def get_axis_align(pcd, info_path):
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return np.zeros((4,4))
    x = (np.max(points[:,0])+np.min(points[:,0]))/2
    y = (np.max(points[:,1])+np.min(points[:,1]))/2
    z = np.min(points[:,2])
    mat = np.identity(4)
    mat[0][3] = -x
    mat[1][3] = -y
    mat[2][3] = -z
    return mat

def generate_one_region(packed_data):
    in_root, in_sam_dir, out_root, posed_image_dir = packed_data
    if not os.path.exists(posed_image_dir):
        os.makedirs(posed_image_dir)
    if not os.path.exists(os.path.join(out_root, 'label')):
        os.makedirs(os.path.join(out_root, 'label'))
    if not os.path.exists(os.path.join(out_root, 'lidar')):
        os.makedirs(os.path.join(out_root, 'lidar'))
    if not os.path.exists(os.path.join(out_root, 'sam')):
        os.makedirs(os.path.join(out_root, 'sam'))
    if not os.path.exists(os.path.join(out_root, 'sam_2dmask')):
        os.makedirs(os.path.join(out_root, 'sam_2dmask'))
    
    if os.path.exists(os.path.join(in_root, 'camera_mapping.json')):
        with open(os.path.join(in_root, 'camera_mapping.json'),'r') as f:
            camera_mapping = json.load(f)
    else:
        camera_mapping = dict()
        
    camera_names = os.listdir(os.path.join(in_root, 'camera_poses'))
    boolean_vecs = []
    pcd = o3d.io.read_point_cloud(os.path.join(in_root, 'lidar', 'main.pcd'))
    axis_align_mat = get_axis_align(pcd, in_root)
    if axis_align_mat[3,3] < 0.5:
        with open('error.txt','a') as f:
            print('Important bug!', file=f)
            print('pcd no point', in_root, file=f)
        print('Important bug!')
        print('pcd no point', in_root)
        return
    low_pcd = pcd.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(low_pcd.points)
    
    for camera_name in camera_names:
        cam_id, cam_pose_id = splitnames(camera_name)
        if cam_id not in camera_mapping:
            mapped_cam_id = len(list(camera_mapping.keys()))
            mapped_cam_id = str(mapped_cam_id).zfill(4)
            camera_mapping[cam_id] = mapped_cam_id
        else:
            mapped_cam_id = camera_mapping[cam_id]
        
        try:
            shutil.copyfile(os.path.join(in_root, 'color_images', f'{cam_id}_i{cam_pose_id}.jpg'), os.path.join(posed_image_dir, f'{mapped_cam_id}_{cam_pose_id}.jpg'))
            shutil.copyfile(os.path.join(in_root, 'depth_images', f'{cam_id}_d{cam_pose_id}.png'), os.path.join(posed_image_dir, f'{mapped_cam_id}_{cam_pose_id}.png'))
            shutil.copyfile(os.path.join(in_root, 'camera_poses', f'{cam_id}_pose_{cam_pose_id}.txt'), os.path.join(posed_image_dir, f'{mapped_cam_id}_{cam_pose_id}.txt'))
        except Exception as e:
            with open('error.txt','a') as f:
                print(e, file=f)
            boolean_vec = np.zeros((points.shape[0],), dtype=bool)
            boolean_vecs.append(boolean_vec)
            continue
        
        extrinsic = np.loadtxt(os.path.join(in_root, 'camera_poses', f'{cam_id}_pose_{cam_pose_id}.txt'))
        cam_intrin_id = cam_pose_id.split('_')[0]
        width, height, intrinsic = read_intrinsic(os.path.join(in_root, 'camera_intrinsics', f'{cam_id}_intrinsics_{cam_intrin_id}.txt'))
        boolean_vec = is_in_cone_by_camera_params(points, intrinsic, np.linalg.inv(extrinsic), width, height)
        boolean_vecs.append(boolean_vec)
    
    intrinsic_names = os.listdir(os.path.join(in_root, 'camera_intrinsics'))
    for intrinsic_name in intrinsic_names:
        cam_id, cam_pose_id = splitnames(intrinsic_name)
        cam_intrin_id = cam_pose_id.split('_')[0]
        assert cam_id in camera_mapping
        mapped_cam_id = camera_mapping[cam_id]
        width, height, intrinsic = read_intrinsic(os.path.join(in_root, 'camera_intrinsics', f'{cam_id}_intrinsics_{cam_intrin_id}.txt'))
        np.savetxt(os.path.join(posed_image_dir, f'intrinsic_{mapped_cam_id}_{cam_intrin_id}.txt'), intrinsic, fmt='%.3f')
    
    boolean_vecs = np.array(boolean_vecs)
    selected_camera_indices = coverage_solver(boolean_vecs, 0.95)
    
    for selected in selected_camera_indices:
        selected_camera = camera_names[selected]
        cam_id, cam_pose_id = splitnames(selected_camera)
        assert cam_id in camera_mapping
        mapped_cam_id = camera_mapping[cam_id]
        try:
            generate_sam_box(
                sam_img_path=os.path.join(in_sam_dir, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.png'),        # TODO
                depth_img_path=os.path.join(in_root, 'depth_images', f'{cam_id}_d{cam_pose_id}.png'),
                sam_json_path=os.path.join(in_sam_dir, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.json'),      # TODO
                intrinsic_path=os.path.join(posed_image_dir, f'intrinsic_{mapped_cam_id}_{cam_intrin_id}.txt'),
                extrinsic_path=os.path.join(in_root, 'camera_poses', f'{cam_id}_pose_{cam_pose_id}.txt'),
                out_sam_box_path=os.path.join(out_root, 'sam', f'{mapped_cam_id}_{cam_pose_id}.json'),
                out_sam_img_path=os.path.join(out_root, 'sam_2dmask', f'{mapped_cam_id}_{cam_pose_id}.jpg'),
                out_sam_json_path=os.path.join(out_root, 'sam_2dmask', f'{mapped_cam_id}_{cam_pose_id}.json'),
                axis_matrix= axis_align_mat
            )
        except Exception as e:
            with open('error.txt','a') as f:
                print(e, file=f)
                print('sam box error', os.path.join(in_sam_dir, 'sam_2dmask', f'{cam_id}_{cam_pose_id}.png'), file=f)
    
    with open(os.path.join(in_root, 'label', 'main.json'), 'r') as f:
        annos = json.load(f)
    
    for anno in annos:
        x, y, z = anno['psr']['scale']['x'], anno['psr']['scale']['y'], anno['psr']['scale']['z']
        anno['psr']['scale']['x'], anno['psr']['scale']['y'], anno['psr']['scale']['z'] = x * 2, y * 2, z * 2
        x, y, z = anno['psr']['position']['x'], anno['psr']['position']['y'], anno['psr']['position']['z']
        x += axis_align_mat[0][3]
        y += axis_align_mat[1][3]
        z += axis_align_mat[2][3]
        anno['psr']['position']['x'], anno['psr']['position']['y'], anno['psr']['position']['z'] = x, y, z
    
    with open(os.path.join(out_root, 'label', 'main.json'), 'w') as f:
        json.dump(annos, f)
    
    pcd.transform(axis_align_mat)
    o3d.io.write_point_cloud(os.path.join(out_root, 'lidar', 'main.pcd'), pcd)
    np.save(os.path.join(out_root, 'label', 'rot_matrix.npy'), axis_align_mat)
    try:
        with open(os.path.join(in_root, 'camera_mapping.json'),'w') as f:
            json.dump(camera_mapping, f)
    except Exception as e:
        with open('error.txt','a') as f:
            print(e, file=f)
            print('write error', file=f)
    
    with open(os.path.join(out_root, 'camera_mapping.json'),'w') as f:
        json.dump(camera_mapping, f)
            
    # shutil.copyfile(os.path.join(in_root, 'lidar', 'main.pcd'), os.path.join(out_root, 'lidar', 'main.pcd'))


def main(root = './'):
    if os.path.exists(os.path.join(root, 'scene_mapping.json')):
        with open(os.path.join(root, 'scene_mapping.json'), 'r') as f:
            scene_mapping = json.load(f)
    else:
        scene_mapping = dict()
        
    in_root = '/mnt/petrelfs/share_data/lipeisen/matterport3d/regions'
    sam_root = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/matterport3d_sam'
    out_root = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/data'
    posed_image_dir = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/data/posed_images'
    regions = os.listdir(in_root)
    todo_datas = []
    for region in regions:
        scene, reg_id = region.split('_')
        if scene in scene_mapping:
            mapped_scene = scene_mapping[scene]
        else:
            mapped_scene = len(list(scene_mapping.keys()))
            mapped_scene = '1mp3d_' + str(mapped_scene).zfill(4)
            scene_mapping[scene] = mapped_scene
        mapped_region = mapped_scene + '_' + reg_id
        if not os.path.exists(os.path.join(out_root, mapped_region)):
            os.makedirs(os.path.join(out_root, mapped_region))
        if not os.path.exists(os.path.join(posed_image_dir, mapped_region)):
            os.makedirs(os.path.join(posed_image_dir, mapped_region))
        
        todo_datas.append((os.path.join(in_root, region), os.path.join(sam_root, region), os.path.join(out_root, mapped_region), os.path.join(posed_image_dir, mapped_region)))
    
    # for data in tqdm(todo_datas):
    #     generate_one_region(data)
    mmengine.utils.track_parallel_progress(generate_one_region, todo_datas, 8)
    
    with open(os.path.join(root, 'scene_mapping.json'), 'w') as f:
        json.dump(scene_mapping, f)
        
    
main()