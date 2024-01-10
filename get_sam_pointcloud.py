import cv2
import numpy as np
import json
import open3d as o3d

def read_intrinsic(path):
    a = np.loadtxt(path)
    intrinsic = np.identity(4, dtype=float)
    intrinsic[0][0] = a[2]  # fx
    intrinsic[1][1] = a[3]  # fy
    intrinsic[0][2] = a[4]  # cx
    intrinsic[1][2] = a[5]  # cy
    return a[0], a[1], intrinsic


def generate_sam_box(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, extrinsic_path, out_sam_box_path, out_sam_json_path, out_sam_img_path):
    _, __, intrinsic = read_intrinsic(intrinsic_path)
    extrinsic = np.loadtxt(extrinsic_path)
    img = cv2.imread(sam_img_path)
    depth = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    with open(sam_json_path, 'r') as f:
        sam_id_json = json.load(f)
    sam_id = np.array(sam_id_json, dtype=int)
    
    group_id = np.unique(sam_id)
    group_of_points = []
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
        
        if points.shape[0] == 0:
            mapping[id] = -1
            continue
        xyzs = (extrinsic) @ np.linalg.inv(intrinsic) @ points.transpose()
        xyzs = xyzs.transpose()
        xyzs = xyzs[:,:3]
        group_of_points.append(xyzs)
                
        count += 1
    return group_of_points
    
