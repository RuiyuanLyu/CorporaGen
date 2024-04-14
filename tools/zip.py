import os
import shutil
import tarfile

input_dir = './data'
output_dir = './data_zip'
target_size = 2 * 1024 * 1024 * 1024  # 2GB

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
folder_sizes = [(folder, sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(folder) for file in files)) for folder in folders]
print(f'Folder sizes: {folder_sizes}')
batches_folders = []
current_batch = []
current_size = 0
for folder, size in folder_sizes:
    if current_size + size <= target_size:
        current_batch.append(folder)
        current_size += size
    else:
        batches_folders.append(current_batch)
        current_batch = [folder]
        current_size = size
batches_folders.append(current_batch)
num_batches = len(batches_folders)
print(f'Number of batches: {num_batches}')

from tqdm import tqdm
for i in range(num_batches):
    print(f'Batch {i+1} of {num_batches} started.')
    tar_filename = os.path.join(output_dir, f'data_batch_{i+1}.tar')
    with tarfile.open(tar_filename, 'w') as tar:
        for folder in tqdm(batches_folders[i]):
            tar.add(folder, arcname=os.path.basename(folder))
    print(f'Batch {i+1} of {num_batches} completed.')   

