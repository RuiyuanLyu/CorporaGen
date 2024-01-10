import os
import cv2
from tqdm import tqdm
import json
from utils import splitnames

sam_mapping = dict()
def prepare_sam_mapping(root):
    dirs = os.listdir(root)
    for d in dirs:
        if os.path.isfile(os.path.join(root, d)):
            sam_mapping[d] = os.path.join(root, d)
        else:
            prepare_sam_mapping(os.path.join(root, d))

def main():
    root1 = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/scans'
    root2 = '/mnt/petrelfs/share_data/maoxiaohan/matterport3d/matterport3d_sam'
    prepare_sam_mapping('/mnt/petrelfs/share_data/maoxiaohan/matterport3d/matterport3d_sam')
    prepare_sam_mapping('/mnt/petrelfs/share_data/maoxiaohan/matterport3d/sam_new')
    
    count = 0
    lost = []
    scans = os.listdir(root1)
    for scan in tqdm(scans):
        now_path = os.path.join(root1, scan, 'matterport_camera_poses')
        images = os.listdir(now_path)
        for image in images:
            count += 1
            assert image[-4:] == '.txt'
            cam_id, cam_pose_id = splitnames(image)
            color_image = os.path.join(root1, scan, 'matterport_color_images', f'{cam_id}_i{cam_pose_id}.jpg')
            assert os.path.exists(color_image)
            json_in = (f'{cam_id}_{cam_pose_id}.json' not in sam_mapping)
            png_in = (f'{cam_id}_{cam_pose_id}.png' not in sam_mapping)
            if (not json_in) and (not png_in):
                lost.append(color_image)
    
    print(len(list(sam_mapping.keys())))
    print(count)
    
    with open('check_sam.txt', 'w') as f:
        for l in lost:
            print(l, file=f)
    
    with open('check_sam.json','w') as f:
        json.dump(lost, f)
    
    # ? regions = os.listdir(root2)
    # sam_images = []
    # for region in tqdm(regions):
    #     now_path = os.path.join(root2, region, 'sam_2dmask')
    #     images = os.listdir(now_path)
    #     for image in images:
    #         if image[-4:] == '.png':
    #             sam_images.append(image[:-4])
    
    # print(len(ori_images))
    # print(len(sam_images))
    # print(ori_images[:10])
    # print(sam_images[:10])
    # count = 0
    # for image in tqdm(ori_images):
    #     if image not in sam_images:
    #         count += 1
    # print(count)

main()
    