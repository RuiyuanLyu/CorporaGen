# usage1: crop a picture and save it to a new file
# usage2: swap a patch in a picture and save it to a new file

import cv2
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='input', help='input picture file path')
    parser.add_argument('-p', '--patch', default='patch', help='patch picture file path')
    parser.add_argument('-o', '--output', default='output.jpg', help='output picture file path')
    parser.add_argument('--crop', action='store_true', help='crop mode')
    parser.add_argument('--swap', action='store_true', help='swap mode')
    parser.add_argument('--x', default=0, type=int, help='x coordinate of the top-left corner of the patch')
    parser.add_argument('--y', default=0, type=int, help='y coordinate of the top-left corner of the patch')
    parser.add_argument('--w', default=600, type=int, help='width of the patch')
    parser.add_argument('--h', default=600, type=int, help='height of the patch')
    args = parser.parse_args()
    return args

def crop_picture(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def swap_picture(img, patch, x, y):
    img[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
    return img

def get_input_filepath(filename):
    suffix = ['.jpg', '.png', '.jpeg']
    for s in suffix:
        if filename.endswith(s):
            return filename
    for s in suffix:
        if os.path.exists(filename + s):
            return filename + s
    return filename + '.png'
        

if __name__ == '__main__':
    args = get_args()
    img = cv2.imread(get_input_filepath(args.input))
    if img is None:
        print('Failed to read input picture')
    if args.crop:
        cropped = crop_picture(img, args.x, args.y, args.w, args.h)
        cv2.imwrite(get_input_filepath(args.patch), cropped)
    elif args.swap:
        patch = cv2.imread(get_input_filepath(args.patch))
        swapped = swap_picture(img, patch, args.x, args.y)
        cv2.imwrite(args.output, swapped)
    else:
        print('Invalid mode')
