"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license 
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Zhaoyang Lv
"""

import os
from numpy.lib.twodim_base import mask_indices
import torch.utils.data as data
import os.path as osp
import numpy as np
import re
from pathlib import Path, PosixPath

from imageio import imread


class RefreshDataset(data.Dataset):

    def __init__(self, root_dir='/mnt/lustre/yslan/Dataset/office2/'):
        """
        :param the directory of color images
        :param the directory of depth images
        """

        root_dir = Path(root_dir)
        assert root_dir.exists()

        # please ensure the four folders use the same number of synchronized files
        # self.ids = len(dmv_files) - 1
        self.corpus = []
        with open(root_dir / 'office2.txt', 'r') as f:
            sub_frames = f.readlines()
            for frame in sub_frames:
                frame_dir = Path(frame.rstrip())
                image_ids = sorted((frame_dir / 'images').glob('*.png'))
                for idx in image_ids:
                    idx = idx.stem
                    self.corpus.append([
                        frame_dir / 'depth_est' / (idx+'.pfm'),
                        frame_dir / 'estimated_mono_depth' / (idx+'.pfm'),
                        frame_dir / 'flow' / (idx+'.flo'),
                        frame_dir / 'depth' / (idx+'.png'),
                    ])

    def __getitem__(self, index):
        dsv_path, dmv_path, flow_path, depth_path = self.corpus[index]
        dsv, dmv, depth = (self._load_depth_tensor(p)
                           for p in (dsv_path, dmv_path, depth_path))
        flow = readFlow(flow_path)
        return dsv, dmv, flow, depth

    def __len__(self):
        return len(self.corpus)

    def _load_rgb_tensor(self, path):
        image = imread(path)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def _load_depth_tensor(self, path):
        assert type(path) is PosixPath
        if path.suffix == '.dpt':
            depth = dpt_depth_read(path)
        elif path.suffix == '.png':
            depth = imread(path) / 1000.0
            # clamp the inverse map(disparity here)
            depth = np.clip(depth, a_min=1e-4, a_max=10)
        elif path.suffix == '.pfm':
            depth = read_pfm(path)[0]
        else:
            raise NotImplementedError
        return depth[np.newaxis, :]


def dpt_depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    TAG_FLOAT = 202021.25
    TAG_CHAR = 'PIEH'
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


if __name__ == "__main__":
    dataset = RefreshDataset()
    print(len(dataset))
    print([x.shape for x in dataset[100]])
