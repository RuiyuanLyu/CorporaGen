import json
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def test1():
    example_json = load_json('example.json')["scene0000_00"]
    # import pdb; pdb.set_trace()
    for vg_obj in example_json:
        sentence = vg_obj['sentence']
        positive = vg_obj['positive']
        for obj_id, positive_list in positive.items():
            for bgn_end in positive_list:
                bgn, end = bgn_end
                print(sentence[bgn:end+1])

def test2():
    example_json = load_json('example2.json')
    # import pdb; pdb.set_trace()
    with open('example2.json', 'w') as f:
        json.dump(example_json, f, indent=4)

def test3():
    file = 'D:\Projects\corpora_local\data\\0_demo\\region_views\\1_dinning region\struction_trans_mada_1.npy'
    region_info = np.load(file, allow_pickle=True)
    # Here a object_id example is '<stool_29>'
    loc_relation_dict = region_info[0]
    # key: object_id pair, value: string, location relation (closed set of relations)
    logic_relation_dict = region_info[1]
    # key: object_id pair, value: string, logic relation (open set of relations)
    wall_floor_addition = region_info[2]
    # key: 'wall' or 'floor', value: list of object_id
    object_function_dict = region_info[3]
    # key: object_id, value: string, object function (open set of functions)
    large_class_dict = region_info[4]
    # key: object_id tuple of undetermined length, value: string, large class, describes the function of the object set
    region_feature_dict = region_info[5]
    # keys: total 8, example: 'location and function description'. value: string, region feature (open set of features)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    # test1()
    # test2()
    test3()