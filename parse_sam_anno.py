import json
import os

data_root = './example_data/'
frame_id = '02300'

sam_anno_file = os.path.join(data_root, 'sam',  frame_id + '.json')
mask_anno_file = os.path.join(data_root, 'sam_2dmask', frame_id + '.json')

