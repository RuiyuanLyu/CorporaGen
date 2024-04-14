data_dir = "data"

import os

def list_files(data_dir, scene_name):
    files = []
    for file in os.listdir(os.path.join(data_dir, scene_name, "cropped_objects")):
        if file.endswith(".jpg"):
            files.append('_'.join(file.split('_')[:2]))
            print(files[-1])
    return files

scene_name = "scene0147_00"
files = list_files(data_dir, scene_name)


def count_files(data_dir, data_split):
    counts = {}
    for root, dirs, files in os.walk(data_dir):
        if not (data_split in root and "cropped" in root):
            continue
        scene_name = root.split("\\")[-2]
        if scene_name not in counts:
            counts[scene_name] = 0
        for file in files:
            if file.endswith(".jpg"):
                counts[scene_name] += 1
    return counts

def save_counts():
    import pandas as pd
    counts = count_files(data_dir, "mp3d")
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    df.to_csv("mp3d_counts.csv")

    counts = count_files(data_dir, "3rscan")
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    df.to_csv("3rscan_counts.csv")

    counts = count_files(data_dir, "scene")
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    df.to_csv("scannet_counts.csv")

def count_all_files(data_dir):
    mp3d_count, rscan_count, scannet_count = 0, 0, 0
    for root, dirs, files in os.walk(data_dir):
        if not "cropped" in root:
            continue
        for file in files:
            if file.endswith(".jpg"):
                if "mp3d" in root:
                    mp3d_count += 1
                elif "rscan" in root:
                    rscan_count += 1
                elif "scene" in root:
                    scannet_count += 1

    print("mp3d count:", mp3d_count) # 28260/34320 = 84.74%
    print("rscan count:", rscan_count) # 28869/41884 = 74.30%
    print("scannet count:", scannet_count) # 44273/51710 = 88.15%
    total_count = mp3d_count + rscan_count + scannet_count
    print("total count:", total_count) # 101,402/127,914 = 79.81%

    print(f"percentage of mp3d: {mp3d_count/total_count*100:.2f}%") # 27.87%
    print(f"percentage of rscan: {rscan_count/total_count*100:.2f}%") # 28.47%
    print(f"percentage of scannet: {scannet_count/total_count*100:.2f}%") # 43.66%