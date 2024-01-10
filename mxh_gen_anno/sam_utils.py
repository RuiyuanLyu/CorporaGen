import numpy as np
import cv2
import json
from utils import read_intrinsic

def generate_sam_box(sam_img_path, depth_img_path, sam_json_path, intrinsic_path, extrinsic_path, out_sam_box_path, out_sam_json_path, out_sam_img_path):
    _, __, intrinsic = read_intrinsic(intrinsic_path)
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
        
        box = {'position': {'x' : (xmin+xmax) / 2.0,
                            'y' : (ymin+ymax) / 2.0,
                            'z' : (zmin+zmax) / 2.0},
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