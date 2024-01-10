import json
import os
import shutil

def collect(root, out_root):
    scenes = os.listdir(root)
    for scene in scenes:
        if scene[:6] == '3rscan':
            if not os.path.exists(os.path.join(out_root, scene)):
                os.makedirs(os.path.join(out_root, scene))
            shutil.copytree(os.path.join(root, scene, 'label'), os.path.join(out_root, scene, 'label'))
    

collect('./data', './collected_data')