from utils_read import load_json
from object_text_anno import DATA_ROOT, map_text_to_bool
import os

scene_ids = ["1mp3d_0001_region8", "1mp3d_0002_region23", "3rscan0138", "3rscan0036", "scene0026_00", "scene0094_00", "scene0147_00"][3:5]
# scene_ids = ["scene0000_00"]
model_names = ["cogvlm_crop", "XComposer2_crop", "gpt4v_crop", "gpt4v_paint_highdetail", "InternVL-Chat-V1-2-Plus_crop"][:]
users_to_check = ["boyue", "QS_wangjing", "shujutang_01"][1:]

STRINGS_TO_NEGLECT = ["似乎", "好像", "大概", "看起来", "可能",
                      "图片中的", "图片", "照片中的", "照片"]
def filter_text(text: str) -> str:
    """
    Remove some strings from the text.
    """
    prev_len = len(text)
    for string in STRINGS_TO_NEGLECT:
        text = text.replace(string, "")
    new_len = len(text)
    # if prev_len != new_len:
    #     print(f"Infomation: {prev_len-new_len} chars of {text} are removed from the text.")   
    return text.strip()


for user in users_to_check:
    acc_dict_list = []
    for scene_id in scene_ids:
        reference_dir = os.path.join(DATA_ROOT, scene_id, f"corpora_object_{model_names[0]}")
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
            annotation = load_json(os.path.join(dir_to_check, file_name))
            ######################################################################
            ## previsouly, we used the following code to load the accuracy_dict
            updation = annotation.get("accuracy_dict", {})
            for key, value in updation.items():
                if value is None:
                    print(key, value)
                    continue
                acc_dict[key] = map_text_to_bool(value)
            from object_text_ui import KEYS
            keys_to_check = KEYS
            score = sum(acc_dict[key] for key in keys_to_check) / len(keys_to_check)
            acc_dict["score"] = score
            # acc_dict_list.append(acc_dict)
            ######################################################################
            # now we simply compute the length of the delta between the reference and the user's annotation
            import difflib
            original_annotation = annotation.get("translated_description", "")
            modified_annotation = annotation.get("modified_description", "")
            if not original_annotation or not modified_annotation:
                print("Warning: original or modified annotation is empty.")
            original_annotation = filter_text(original_annotation)
            modified_annotation = filter_text(modified_annotation)
            d = difflib.Differ()
            delta = [(token[2:], token[0] if token[0] != " " else None)for token in d.compare(original_annotation, modified_annotation)]
            token_added = sum(len(token[0]) for token in delta if token[1] == "+")
            token_removed = sum(len(token[0]) for token in delta if token[1] == "-")
            acc_dict["token_added"] = token_added
            acc_dict["token_removed"] = token_removed
            acc_dict_list.append(acc_dict)
    ######################################################################
    ## now we save the accuracy_dict_list to a csv file
    import pandas as pd
    df = pd.DataFrame(acc_dict_list)
    df.to_csv(f"accuracy_dict_{user}.csv", index=True)
    print(f"Accuracy of {user} on {len(scene_ids)} scenes has been saved.")
    # print(df)
    #####################################################################
    ## also compute the average score
    # scene_id = "3rscan0036"
    # acc_dict_list = [acc_dict for acc_dict in acc_dict_list if acc_dict["scene_id"] == scene_id]
    acc_dict_list = [acc_dict for acc_dict in acc_dict_list if "meta" in acc_dict]
    ## filter out the annotations without meta or without visual_info_sufficient
    # acc_dict_list = [acc_dict for acc_dict in acc_dict_list if acc_dict["meta"] and acc_dict["visual_info_sufficient"]]
    score_list = [acc_dict["score"] for acc_dict in acc_dict_list]
    avg_score = sum(score_list) / len(score_list)
    print(f"Note:lower is better: lower score means more errors detected by user")
    print(f"Average score eval by {user} is {sum(score_list):.2f}/{len(score_list)}={avg_score:.2f}:")
    token_added_list = [acc_dict["token_added"] for acc_dict in acc_dict_list]
    token_removed_list = [acc_dict["token_removed"] for acc_dict in acc_dict_list]
    avg_token_added = sum(token_added_list) / len(token_added_list)
    avg_token_removed = sum(token_removed_list) / len(token_removed_list)
    print(f"Average token added/removed by {user} is {avg_token_added:.2f}/{avg_token_removed:.2f}:")

