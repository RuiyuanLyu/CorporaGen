import numpy as np
import pickle
from tqdm import tqdm
import os

out_put_dir = "./splitted_infos"
os.makedirs(out_put_dir, exist_ok=True)
# input_files = 3rscan_infos_test_MDJH_aligned_full_10_visible.pkl embodiedscan_infos_train_full.pkl embodiedscan_infos_val_full.pkl matterport3d_infos_test_full_10_visible.pkl
input_files = ["embodiedscan_infos_train_full.pkl",
               "embodiedscan_infos_val_full.pkl",
               "matterport3d_infos_test_full_10_visible.pkl",
               "3rscan_infos_test_MDJH_aligned_full_10_visible.pkl"]

for input_file in input_files:
    print(f"Processing {input_file}...")
    infos = np.load(input_file, allow_pickle=True)
    metainfo = infos['metainfo']
    data_list = infos['data_list']
    for i, data in tqdm(enumerate(data_list)):
        scene_id = data["images"][0]["img_path"].split("/")[-2]  # str
        out_file = f"{out_put_dir}/{scene_id}.pkl"
        out_dict = {"metainfo": metainfo, "data_list": [data]}
        with open(out_file, 'wb') as f:
            pickle.dump(out_dict, f)
        # if i == 0:
        #     # test if the output file can be loaded
        #     from utils.utils_read import read_annotation_pickles
        #     anno = read_annotation_pickles([out_file])
        #     anno2 = np.load(input_file, allow_pickle=True)
        #     import pdb; pdb.set_trace()