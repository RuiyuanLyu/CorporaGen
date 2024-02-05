import json
import numpy as np
import os
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
from pytorch3d.io import load_obj
from tqdm import tqdm
import torch

torch.set_printoptions(precision=5, sci_mode= False)
np.set_printoptions(suppress=True)

def get_xy(rot_mat):
    tmp = []
    for j in range(3):
        if abs(rot_mat[j][2] / np.linalg.norm(rot_mat[j])) > 0.5:
            continue
        tmp.append(rot_mat[j])
    assert len(tmp) == 2
    x, y = tmp
    x = x[:2]
    y = y[:2]
    norm = np.linalg.norm(x)
    x = x / norm
    norm = np.linalg.norm(y)
    y = y / norm
    return x, y

def calc(rot_mats):
    max_count = 0.0
    ans = None
    for i in range(len(rot_mats)):
        count = 0.0
        rot_mat = rot_mats[i].copy().transpose()
        x, y = get_xy(rot_mat)
        for j in range(len(rot_mats)):
            xx, yy = get_xy(rot_mats[j].copy().transpose())
            count += max(np.abs(np.dot(x, xx)), np.abs(np.dot(x, yy)))
        
        if count > max_count:
            max_count = count
            if np.sum(np.cross(x,y)) < 0:
                ans = (y,x)
            else:
                ans = (x,y)
    
    result = np.zeros((3,3))
    result[0][0] = ans[0][0]
    result[1][0] = ans[0][1]
    result[0][1] = ans[1][0]
    result[1][1] = ans[1][1]
    result[2][2] = 1
    return result

def xyz_zxy(ori_box):
    if ori_box.shape[0] == 0:
        return ori_box
    box = torch.tensor(ori_box)
    ori_matrix = euler_angles_to_matrix(box[:,6:], 'XYZ')
    # print('xyz to zxy')
    # print(ori_matrix)
    # print('--------------------------------')
    zxy_angle = matrix_to_euler_angles(ori_matrix, 'ZXY')
    box[:, 6:] = zxy_angle
    return box.numpy()

def zxy_xyz(ori_box):
    if ori_box.shape[0] == 0:
        return ori_box
    box = torch.tensor(ori_box)
    ori_matrix = euler_angles_to_matrix(box[:,6:], 'ZXY')
    zxy_angle = matrix_to_euler_angles(ori_matrix, 'XYZ')
    box[:, 6:] = zxy_angle
    return box.numpy()

def box_to_rot(bbox):
    tmp = torch.tensor(bbox)
    ori_matrix = euler_angles_to_matrix(tmp[:,6:], 'ZXY')
    # print(ori_matrix)
    # print('end')
    return ori_matrix.numpy()

def transform(matrix, ori_box):
    box = torch.tensor(ori_box)
    if box.shape[0] == 0:
        return ori_box
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    points = box[:,:3]
    # print(points)
    constant = points.new_ones(points.shape[0], 1)
    points_extend = torch.concat([points, constant], dim=-1)
    points_trans = torch.matmul(points_extend, matrix.transpose(-2,-1))[:,:3]
    
    size = box[:,3:6]
    # print(size)

    ori_matrix = euler_angles_to_matrix(box[:,6:], 'ZXY')

    # print(ori_matrix)
    # print('-------------------------------')
    rot_matrix = matrix[:3,:3].expand_as(ori_matrix)
    final = torch.bmm(rot_matrix, ori_matrix)
    angle = matrix_to_euler_angles(final, 'ZXY')
    
    # print(angle)
    
    # print(points_trans)
    # print(size)
    # print(final)
    # print('=============================')
    box2 = torch.cat([points_trans, size, angle], dim=-1)
    # print(box2)
    return box2.numpy()

def anno_9dof(ins):
    box = []
    box.append(ins['psr']['position']['x'])
    box.append(ins['psr']['position']['y'])
    box.append(ins['psr']['position']['z'])
    box.append(ins['psr']['scale']['x'])
    box.append(ins['psr']['scale']['y'])
    box.append(ins['psr']['scale']['z'])
    box.append(ins['psr']['rotation']['x'])
    box.append(ins['psr']['rotation']['y'])
    box.append(ins['psr']['rotation']['z'])
    return box

def dof_anno(box, anno):
    n = box.shape[0]
    assert n == len(anno)
    res = []
    for i in range(n):
        ins = dict()
        ins['obj_id'] = anno[i]['obj_id']
        ins['obj_type'] = anno[i]['obj_type']
        ins['psr'] = {'position': dict(), 'scale': dict(), 'rotation': dict()}
        ins['psr']['position']['x'] = box[i][0]
        ins['psr']['position']['y'] = box[i][1]
        ins['psr']['position']['z'] = box[i][2]
        ins['psr']['scale']['x'] = box[i][3]
        ins['psr']['scale']['y'] = box[i][4]
        ins['psr']['scale']['z'] = box[i][5]
        ins['psr']['rotation']['x'] = box[i][6]
        ins['psr']['rotation']['y'] = box[i][7]
        ins['psr']['rotation']['z'] = box[i][8]
        res.append(ins)
    return res

def main(root, outroot, scene, json_names):
    with open(os.path.join(root, scene, 'label', 'main.json'), 'r') as f:
        anno = json.load(f)
    bbox = []
    for ins in anno:
        if ins['obj_type'] == 'wall':
            bbox.append(anno_9dof(ins))

    bbox = np.array(bbox)
    bbox = xyz_zxy(bbox)
    if bbox.shape[0] == 0:
        with open('align_log.txt','a') as f:
            print('no wall', f'{scene}/{json_names}', file=f)
        matrix = np.identity(3)
        # axis_align_matrix = np.identity(4)
    else:
        rot_mats = box_to_rot(bbox)
        # print(rot_mats)
        matrix = calc(rot_mats)
    
    floor_z = []
    for ins in anno:
        if ins['obj_type'] == 'floor':
            floor_z.append(ins['psr']['position']['z'])
    # print(floor_z)
    if len(floor_z) > 0:
        final_z = 0 # floor_z[len(floor_z) // 2]
    else:
        with open('align_log.txt','a') as f:
            print('no floor', f'{scene}/{json_names}', file=f)
        
        final_z = 0
        
    # points = np.asarray(mesh.vertices) @ np.linalg.inv(matrix)
    # final_x = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
    # final_y = (np.min(points[:, 1]) + np.max(points[:, 1])) / 2
    new_matrix = np.identity(4)
    new_matrix[:3,:3] = matrix
    new_matrix[2, 3] = final_z
    axis_align_matrix = np.linalg.inv(new_matrix)
    
    if not os.path.exists(os.path.join(outroot, scene)):
        os.makedirs(os.path.join(outroot, scene))
    
    with open(os.path.join(root, scene, 'label', json_names), 'r') as f:
        human_anno = json.load(f)
        
    all_box = []
    for ins in human_anno:
        all_box.append(anno_9dof(ins))
    
    all_box = np.array(all_box)
    all_box = xyz_zxy(all_box)
    all_box2 = transform(axis_align_matrix, all_box)
    
    all_box2 = zxy_xyz(all_box2)
    with open(os.path.join(outroot, scene, 'aligned_' + json_names), 'w') as f:
        json.dump(dof_anno(all_box2, human_anno), f)

    np.save(os.path.join(outroot, scene, 'axis_align_matrix.npy'), axis_align_matrix)

if __name__ == '__main__':
    root = '/mnt/petrelfs/share_data/maoxiaohan/3rscan_data/data_1023/matterport3d'
    outroot = '/mnt/petrelfs/share_data/maoxiaohan/3rscan_data/data_1023/aligned_matterport3d'
    scenes = os.listdir(root)
    with open('align_log.txt','w') as f:
        print('begin', file=f)
    for scene in tqdm(scenes):
        files = os.listdir(os.path.join(root, scene, 'label'))
        json_files = [file for file in files if file.split('.')[-1]=='json']
        for json_file in json_files:
            main(root, outroot, scene, json_file)