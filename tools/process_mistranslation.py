# The code is used to process the translation that do not strictly follow the pattern.
import json
import os
import re
import numpy as np

with open("object_type_translation.json", "r") as f:
    OBJ_EN2CN = json.load(f)
OBJ_CN2EN = {v: k for k, v in OBJ_EN2CN.items()}

def check_object_with_ids(text):
    """
    Check if the text contains objects with ids.
    """
    pattern = r"<\w+_\d+>"
    objects_with_ids = re.findall(pattern, text)
    if len(objects_with_ids) != 1:
        return False
    return True

def process_string(string:str, objects_with_ids:list[str]) -> str:
    """
    Process the string to match the pattern.
    """
    for obj in objects_with_ids:
        # obj must be in the format of "<typeEN_id>""
        assert check_object_with_ids(obj)
        obj_type_en, obj_id = obj.strip("<>").split("_")
        obj_type_cn = OBJ_EN2CN[obj_type_en]
        pattern = rf'<[^<>]+_{obj_id}>'
        string = re.sub(pattern, obj, string)
        string = string.replace(f"{obj_type_cn}_{obj_id}", obj)
        string = string.replace(f"{obj_type_cn}{obj_id}", obj)
        string = string.replace(f"{obj_type_cn}", obj)
        string = re.sub(pattern, obj, string)
    return string

def process_mistranslation(region_info):
    """
        Process the file to match the pattern.
    """
    # loc_relation_dict = region_info[0] 
    logic_relation_dict = region_info[1]
    for k, v in logic_relation_dict.items():
        objects_with_ids = [x for x in k]
        v = process_string(v, objects_with_ids)
        logic_relation_dict[k] = v
    # wall_floor_addition = region_info[2]
    object_function_dict = region_info[3]
    for k, v in object_function_dict.items():
        v = process_string(v, [k])
        object_function_dict[k] = v
    # large_class_dict = region_info[4]
    # region_feature_dict = region_info[5]
    return region_info

if __name__ == "__main__":
    string = "这本<书_31>被放在<书架_28>上，方便阅读。"
    objects_with_ids = ["<book_31>", "<shelf_28>"]
    x = process_string(string, objects_with_ids)
    import pdb; pdb.set_trace()
    test_file = "data/scene0548_01/region_views/0_storage region/struction_trans.npy"
    region_info = np.load(test_file, allow_pickle=True)
    # loc_relation_dict = region_info[0] 
    logic_relation_dict = region_info[1]
    for k, v in logic_relation_dict.items():
        objects_with_ids = [x for x in k]
        v = process_string(v, objects_with_ids)
        logic_relation_dict[k] = v
    # wall_floor_addition = region_info[2]
    object_function_dict = region_info[3]
    for k, v in object_function_dict.items():
        v = process_string(v, [k])
        object_function_dict[k] = v
    # large_class_dict = region_info[4]
    # region_feature_dict = region_info[5]
    print(logic_relation_dict)
    import pdb; pdb.set_trace()