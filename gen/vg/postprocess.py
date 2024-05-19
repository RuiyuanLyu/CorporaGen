import json

input_file = "gen\example_outs\VG_raw.json"
output_file = "gen\example_outs\VG.json"

with open(input_file, 'r') as f:
    data = json.load(f)

def to_list_of_int(x):
    # x may be a single int, or a list of str(int), or a list of int
    if isinstance(x, int):
        return [x]
    elif isinstance(x, str):
        return [int(x)]
    elif isinstance(x, list):
        return [int(i) for i in x]
    else:
        raise ValueError("Invalid input type")

for d in data:
    # {"scan_id": "scene0000_00", "target_id": [7], "distractor_ids": [], "text": "choose the curtain that is above the desk", "target": ["curtain"], "anchors": ["desk"], "anchor_ids": [8], "tokens_positive": [[11, 18], [37, 41]]}
    try:
        d["target_id"] = to_list_of_int(d["target_id"])
        d["distractor_ids"] = to_list_of_int(d["distractor_ids"])
        d["anchor_ids"] = to_list_of_int(d["anchor_ids"])
    except ValueError:
        del d

with open(output_file, 'w') as f:
    json.dump(data, f)