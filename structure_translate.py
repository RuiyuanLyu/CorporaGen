from strict_translate import strict_list_translate
import numpy as np
def anno_translation(region_info,src_lang,tgt_lang,need_translate_index = [1,3,4,5]):

    total_list = []
    for _index in need_translate_index:
        total_list+= [region_info[_index][k] for k in region_info[_index].keys()]

    output_list,try_ = strict_list_translate(total_list,src_lang,tgt_lang)
    print(try_)

    start = 0
    for _index in need_translate_index:
        for k in range(len(region_info[_index].keys())):
            _key = list(region_info[_index].keys())[k]
            region_info[_index][_key] = output_list[k+start]
        start+=len(region_info[_index].keys())

    return region_info
if __name__ == "__main__":

    scene_id = 'scene0000_00'
    data_dir = f'data/{scene_id}/region_view_test/4_toliet region'
    region_info = np.load(data_dir+'/struction.npy',allow_pickle=True)
    print(region_info)
    out = anno_translation(region_info,"English","Chinese")
    print(out)
    np.save(data_dir+'/struction_trans.npy',out)