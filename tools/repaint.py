from utils_vis import *
from utils_read import *
import numpy as np
import os
import shutil

pickle_files = ["embodiedscan_infos_train_full.pkl",
                "embodiedscan_infos_val_full.pkl",
                "3rscan_infos_test_MDJH_aligned_full_10_visible.pkl",
                "matterport3d_infos_test_full_10_visible.pkl"][-2:-1]

full_annotation = read_annotation_pickles(pickle_files)
# scenes = [sub_dir for sub_dir in os.listdir("data/3rscan") if sub_dir.startswith("3rscan")]
scenes = ['3rscan0000', '3rscan0040', '3rscan0063', '3rscan0078', '3rscan0102', '3rscan0131', '3rscan0151', '3rscan0182', '3rscan0210', '3rscan0261', '3rscan0284', '3rscan0312', '3rscan0339', '3rscan0368', '3rscan0389', '3rscan0409', '3rscan0501', '3rscan0530', '3rscan0544', '3rscan0575', '3rscan0599', '3rscan0602', '3rscan0637', '3rscan0672', '3rscan0698', '3rscan0746', '3rscan0777', '3rscan0803', '3rscan0854', '3rscan0886', '3rscan0921', '3rscan0943', '3rscan0970', '3rscan0999', '3rscan1016', '3rscan1049', '3rscan1077', '3rscan1098', '3rscan1127', '3rscan1169', '3rscan1192', '3rscan1231', '3rscan1260', '3rscan1298', '3rscan1326', '3rscan1359']
# scene_id = '3rscan0001'

with open("scene_mappings/3rscan_mapping.json", "r") as f:
    scene_id_mapping = json.load(f)
scene_id_mapping = reverse_121_mapping(scene_id_mapping)
def transform_img_path(img_path):
    # example input: 3rscan/posed_images/3rscan0001/000000.jpg
    # example input: posed_images/3rscan0001/000000.jpg
    # exapnd the path to the real path, like /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/3rscan/raw_data/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/sequence/frame-000000.color.jpg
    _, scene_id, img_name = img_path.split(".")[0].split("/")
    scene_id = scene_id_mapping.get(scene_id, scene_id)
    real_path = os.path.join("/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/3rscan/raw_data", scene_id, "sequence", f"frame-{img_name}.color.jpg")
    return real_path

def repaint_scene(scene_id):
    scene_id_info = full_annotation[scene_id]
    # draw_box3d_on_img()
    input_dir = os.path.join("data", "3rscan", scene_id, "cropped_objects")
    output_dir = os.path.join("data", "3rscan", scene_id, "repainted_objects")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    color_dict = get_color_map("color_map.txt")

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".jpg"):
            # if os.path.exists(os.path.join(output_dir, file_name)):
            #     continue
            object_id = int(file_name.split("_")[0])
            view_id = file_name.split(".")[0].split("_")[2].split("-")[-1]
            # file_name: 002_couch_000200.jpg, or 16_bicycle_frame-000334.color.jpg
            # print(object_id)
            index = list(scene_id_info["object_ids"]).index(object_id)
            extrinsic_c2w = scene_id_info["extrinsics_c2w"][int(view_id)]
            axis_align_matrix = scene_id_info["axis_align_matrix"]
            extrinsic_c2w = axis_align_matrix @ extrinsic_c2w
            intrinsic = scene_id_info["intrinsics"][int(view_id)]
            object_type = scene_id_info["object_types"][index]
            img_path = scene_id_info["image_paths"][int(view_id)]
            real_img_path = transform_img_path(img_path)
            cy = intrinsic[1, 2]
            intrinsic_converter = np.array([[0, -1, 2*cy, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            intrinsic = np.dot(intrinsic_converter, intrinsic)
            img = cv2.imread(real_img_path)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            box = scene_id_info["bboxes"][index]
            box = np.array(box).reshape(1, 9)
            box = get_9dof_boxes(box, mode="zxy", colors=(0, 255, 0))[0]
            color = color_dict[object_type]
            label = str(object_id) + " " + object_type
            new_img, _ = draw_box3d_on_img(img=img, box=box, color=color, label=label, extrinsic_c2w=extrinsic_c2w, intrinsic=intrinsic, ignore_outside=False)
            cv2.imwrite(os.path.join(output_dir, file_name), new_img)
            print("written to {}".format(os.path.join(output_dir, file_name)))
        elif file_name.endswith(".txt"): # copy the txt file
            shutil.copyfile(os.path.join(input_dir, file_name), os.path.join(output_dir, file_name))

if __name__ == "__main__":
    ####################################################################
    ## usage: on scluster only.
    ####################################################################
    import mmengine
    mmengine.track_parallel_progress(repaint_scene, scenes, nproc=1)