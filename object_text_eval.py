from utils.utils_read import load_json
from object_text_anno import DATA_ROOT, map_text_to_bool, remove_specific_expressions
import os
import json
from tqdm import tqdm

def check_actual_users(scene_id):
    dir_to_check = os.path.join(DATA_ROOT, scene_id, f"corpora_object") 
    if not os.path.exists(dir_to_check):
        return set()
    user_names = os.listdir(dir_to_check)
    user_names = [user_name for user_name in user_names if user_name.startswith("user_")]
    user_names = [user_name.replace("user_", "") for user_name in user_names]
    return set(user_names)

def check_all_users(scene_ids):
    from collections import Counter
    all_users = Counter()
    for scene_id in scene_ids:
        user_names = check_actual_users(scene_id)
        for user_name in user_names:
            all_users[user_name] += 1
    return all_users

import difflib
def compute_delta(original, modified):
    d = difflib.Differ()
    delta = [(token[2:], token[0] if token[0] != " " else None)for token in d.compare(original, modified)]
    token_added = sum(len(token[0]) for token in delta if token[1] == "+")
    token_removed = sum(len(token[0]) for token in delta if token[1] == "-")
    return token_added, token_removed

def compute_min_delta(originals, modified):
    idx_min, idx_max = 0, 0
    if isinstance(originals, str):
        originals = [originals]
    min_added, min_removed = float('inf'), float('inf')
    for i, original in enumerate(originals):
        token_added, token_removed = compute_delta(original, modified)
        if token_added < min_added:
            idx_min = i
            min_added = token_added
        if token_removed < min_removed:
            idx_max = i
            min_removed = token_removed
    return min_added, min_removed, idx_min, idx_max

def eval_for_scene(scene_id, user, model_names, acc_dict_list):
    reference_dir = os.path.join(DATA_ROOT, scene_id, f"corpora_object_{model_names[0]}")
    if not os.path.exists(reference_dir):
        return
    reference_file_names = os.listdir(reference_dir)
    for file_name in reference_file_names:
        if not file_name.endswith(".json"):
            reference_file_names.remove(file_name)
    reference_file_names.sort()
    # dir_to_check = os.path.join(DATA_ROOT, scene_id, f"corpora_object/user_test/user_{user}")
    dir_to_check = os.path.join(DATA_ROOT, scene_id, f"corpora_object/user_{user}")
    for file_name in reference_file_names:
        file_name_to_check = os.path.join(dir_to_check, file_name)
        acc_dict = {}
        acc_dict["scene_id"] = scene_id
        acc_dict["object_id"] = ' '.join(file_name.split("_")[:2])
        if not os.path.exists(file_name_to_check):
            acc_dict_list.append(acc_dict)
            continue
        annotation = load_json(file_name_to_check)
        ######################################################################
        ## previsouly, we used the following code to load the accuracy_dict
        updation = annotation.get("accuracy_dict", {})
        for key, value in updation.items():
            if value is None:
                # print(key, value)
                continue
            acc_dict[key] = map_text_to_bool(value)
        from object_text_ui import KEYS
        keys_to_check = KEYS
        score = sum(acc_dict[key] for key in keys_to_check) / len(keys_to_check)
        acc_dict["error_detected"] = 1 - score
        corpora_source = annotation.get("corpora_source", "")
        acc_dict["corpora_source"] = corpora_source.replace("corpora_object_", "")
        # acc_dict_list.append(acc_dict)
        ######################################################################
        # now we simply compute the length of the delta between the reference and the user's annotation
        # original_annotation = annotation.get("translated_description", "")
        
        original_annotations = []
        loaded_models = []
        for model_name in model_names:
            source_to_load = os.path.join(DATA_ROOT, scene_id, f"corpora_object_{model_name}", file_name)
            if not os.path.exists(source_to_load):
                continue
            loaded_models.append(f"corpora_object_{model_name}")
            original_annotation = load_json(source_to_load).get("translated_description", "")
            original_annotation = remove_specific_expressions(original_annotation)
            original_annotations.append(original_annotation)
        modified_annotation = annotation.get("modified_description", "")
        if not modified_annotation:
            print("Warning: modified annotation is empty.")
            continue
        modified_annotation = remove_specific_expressions(modified_annotation)
        token_added, token_removed, idx_min, idx_max = compute_min_delta(original_annotations, modified_annotation)
        acc_dict["token_added"] = token_added
        acc_dict["token_removed"] = token_removed
        if idx_min == idx_max:
            if annotation["corpora_source"] == loaded_models[idx_min]:
                continue
            annotation["corpora_source"] = loaded_models[idx_min] 
            annotation["translated_description"] = original_annotations[idx_min]
            with open(file_name_to_check, "w") as f:
                json.dump(annotation, f, indent=4, ensure_ascii=False)
            print("fixed:" + file_name_to_check)
            # exit()
        acc_dict_list.append(acc_dict)


if __name__ == "__main__":
    scene_ids = os.listdir(DATA_ROOT)
    # scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith("scene") or scene_id.startswith("1mp3d_0000_region")]
    scene_ids = [scene_id for scene_id in scene_ids if scene_id.startswith("scene") or scene_id.startswith("1mp3d") or scene_id.startswith("3rscan")]
    # all_users = check_all_users(scene_ids)
    # # all_users = sorted(all_users)
    # print(f"All users: {all_users}")
    # exit()

    # scene_ids = ["1mp3d_0001_region8", "1mp3d_0002_region23", "3rscan0138", "3rscan0036", "scene0026_00", "scene0094_00", "scene0147_00"][3:4]
    # scene_ids = ["scene0000_00"]
    model_names = ["cogvlm_crop", "XComposer2_crop", "gpt4v_crop", "gpt4v_paint_highdetail", "InternVL-Chat-V1-2-Plus_crop"][:]
    # users_to_check = ["boyue", "QS_wangjing", "QS_001", "用户名用QS_002", "shujutang_01"]
    users_to_check = ["shujutang_czc"]

    user_eval_dict = {}
    for user in users_to_check:
        acc_dict_list = []
        for scene_id in tqdm(scene_ids):
            eval_for_scene(scene_id, user, model_names, acc_dict_list)
        ######################################################################
        ## now we save the accuracy_dict_list to a csv file
        import pandas as pd
        df = pd.DataFrame(acc_dict_list)
        df.to_csv(f"evaluation_results/accuracy_dict_{user}.csv", index=True)
        # print(f"Accuracy of {user} on {len(scene_ids)} scenes has been saved.")
        # print(df)
        #####################################################################
        ## also compute the average score
        # scene_id = "3rscan0036"
        # acc_dict_list = [acc_dict for acc_dict in acc_dict_list if acc_dict["scene_id"] == scene_id]
        acc_dict_list = [acc_dict for acc_dict in acc_dict_list if "meta" in acc_dict]
        ## filter out the annotations without meta or without visual_info_sufficient
        FILTER_WO_META = False
        if FILTER_WO_META:
            acc_dict_list = [acc_dict for acc_dict in acc_dict_list if acc_dict.get("meta", False)]
        corpora_source_list = [acc_dict.get("corpora_source", "") for acc_dict in acc_dict_list]
        from collections import Counter
        c = Counter(corpora_source_list)
        print(f"Corpora source distribution for {user}: {c}")
        score_list = [acc_dict["error_detected"] for acc_dict in acc_dict_list]
        avg_score = sum(score_list) / len(score_list)
        token_added_list = [acc_dict["token_added"] for acc_dict in acc_dict_list if acc_dict["token_added"] < 100]
        token_removed_list = [acc_dict["token_removed"] for acc_dict in acc_dict_list if acc_dict["token_removed"] < 100]
        import numpy as np
        avg_token_added = np.mean(token_added_list)
        avg_token_removed = np.mean(token_removed_list)
        std_token_added = np.std(token_added_list)
        std_token_removed = np.std(token_removed_list)
        # NOTE: lower score is better (more errors detected by user)
        user_eval_dict[user] = {"num_objects": len(acc_dict_list), "avg_detect_score": avg_score, "avg_token_added": avg_token_added, "avg_token_removed": avg_token_removed, "std_token_added": std_token_added, "std_token_removed": std_token_removed}
        # print(f"Average score eval by {user} is {sum(score_list):.2f}/{len(score_list)}={avg_score:.2f}:")
        # print(f"Average token added/removed by {user} is {avg_token_added:.2f}/{avg_token_removed:.2f}:")
        # print(f"Std token added/removed by {user} is {std_token_added:.2f}/{std_token_removed:.2f}:")
    df = pd.DataFrame(user_eval_dict).T
    df.to_csv("user_eval_dict.csv", index=True)
    print("User evaluation results:")
    print("Filter the annotations without meta or without visual_info_sufficient: ", FILTER_WO_META)
    print(df)