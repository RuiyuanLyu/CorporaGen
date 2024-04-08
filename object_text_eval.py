from utils_read import load_json
from object_text_anno import DATA_ROOT, map_text_to_bool
import os

scene_ids = ["1mp3d_0001_region8", "1mp3d_0002_region23", "3rscan0138", "3rscan0036", "scene0026_00", "scene0094_00", "scene0147_00"]
# scene_ids = ["scene0000_00"]
model_names = ["cogvlm_crop", "XComposer2_crop", "gpt4v_crop", "gpt4v_paint_highdetail", "InternVL-Chat-V1-2-Plus_crop"][:]

for model_name in model_names:
    acc_dict_list = []
    for scene_id in scene_ids:
        reference_dir = os.path.join(DATA_ROOT, scene_id, f"corpora_object_{model_names[0]}")
        reference_file_names = os.listdir(reference_dir)
        for file_name in reference_file_names:
            if not file_name.endswith(".json"):
                reference_file_names.remove(file_name)
        reference_file_names.sort()
        dir_to_check = os.path.join(DATA_ROOT, scene_id, f"corpora_object_{model_name}/user_test")
        for file_name in reference_file_names:
            file_name_to_check = os.path.join(dir_to_check, file_name)
            acc_dict = {}
            acc_dict["scene_id"] = scene_id
            acc_dict["object_id"] = ' '.join(file_name.split("_")[:2])
            if not os.path.exists(file_name_to_check):
                acc_dict_list.append(acc_dict)
                continue
            annotation = load_json(os.path.join(dir_to_check, file_name))
            updation = annotation.get("accuracy_dict", {})
            for key, value in updation.items():
                if value is None:
                    print(key, value)
                    continue
                acc_dict[key] = map_text_to_bool(value)
            acc_dict_list.append(acc_dict)
    import pandas as pd
    df = pd.DataFrame(acc_dict_list)
    df.to_csv(f"accuracy_dict_{model_name}.csv", index=True)
    print(f"Accuracy of {model_name} on {len(scene_ids)} scenes has been saved.")
    # print(df)
