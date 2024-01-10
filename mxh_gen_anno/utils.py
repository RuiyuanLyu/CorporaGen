import json
import numpy as np
import os
import shutil
import open3d as o3d
import cv2
import copy
from scipy.spatial.transform import Rotation

def cp(src, dst):
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)

def json_move(src, dst, axis_align_mat=None):
    with open(src, 'r') as f:
        annos = json.load(f)
    
    if axis_align_mat is not None:
        for anno in annos:
            if 'psr' in anno:
                x, y, z = anno['psr']['position']['x'], anno['psr']['position']['y'], anno['psr']['position']['z']
                x += axis_align_mat[0][3]
                y += axis_align_mat[1][3]
                z += axis_align_mat[2][3]
                anno['psr']['position']['x'], anno['psr']['position']['y'], anno['psr']['position']['z'] = x, y, z
            else:
                assert 'position' in anno
                x, y, z = anno['position']['x'], anno['position']['y'], anno['position']['z']
                x += axis_align_mat[0][3]
                y += axis_align_mat[1][3]
                z += axis_align_mat[2][3]
                anno['position']['x'], anno['position']['y'], anno['position']['z'] = x, y, z
    
    with open(dst, 'w') as f:
        json.dump(annos, f)

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

def depthimg_points(depth_img_path, extrinsic, depth_instrincs, shift=1000):
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

def get_axis_align(pcd):
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

def calc_rot_mat(a, b):
    v0 = copy.deepcopy(a)
    v1 = copy.deepcopy(b)
    v0 = v0 / np.linalg.norm(v0, 2)
    v1 = v1 / np.linalg.norm(v1, 2)
    v1 = np.cross(np.cross(v0, v1), v0)
    v1 = v1 / np.linalg.norm(v1, 2)
    v2 = np.cross(v0, v1)
    v2 = v2 / np.linalg.norm(v2, 2)
    rot_matrix = np.concatenate([v0.reshape(3,1), v1.reshape(3,1), v2.reshape(3,1)], axis=1)
    return rot_matrix

def read_region(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    regions = []
    objects = []
    belongs = []
    cates = []
    labels = []
    # rot_mats = []
    for line in lines:
        if line[0] == 'R':
            a = line.split()
            index = int(a[1])
            assert index == len(regions)
            x1, y1, z1, x2, y2, z2 = float(a[9]), float(a[10]), float(a[11]), float(a[12]), float(a[13]), float(a[14])
            regions.append([x1, y1, z1, x2, y2, z2])
        elif line[0] == 'C':
            a = line.split()
            cate_index = int(a[1])
            assert cate_index == len(cates)
            cate_name = a[3].replace('#', ' ')
            cates.append(cate_name)
        elif line[0] == 'O':
            a = line.split()
            obj_index = int(a[1])
            assert obj_index == len(objects)
            belong_region = int(a[2])
            belongs.append(belong_region)
            label = int(a[3])
            labels.append(cates[label])
            pos = np.array([float(a[4]), float(a[5]), float(a[6])], dtype=float)
            a0 = np.array([float(a[7]), float(a[8]), float(a[9])], dtype=float)
            a1 = np.array([float(a[10]), float(a[11]), float(a[12])], dtype=float)
            rot_mat = calc_rot_mat(a0, a1)
            # rot_mats.append(rot_mat)
            scale = np.array([float(a[13]), float(a[14]), float(a[15])], dtype=float) * 2
            euler_angle = Rotation.from_matrix(np.linalg.inv(rot_mat)).as_euler('xyz', degrees=False)
            objects.append(np.concatenate([pos, scale, euler_angle]))
    
    return np.array(regions), np.array(objects), belongs, labels