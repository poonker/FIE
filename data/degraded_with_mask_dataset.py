import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image
import re


class DegradedWithMaskDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test' also.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_source = os.path.join(opt.dataroot, opt.phase + 'A')  # get the image directory
        self.dir_target = os.path.join(opt.dataroot, opt.phase + 'B')  # get the image directory
        #self.dir_source_mask = os.path.join(opt.dataroot, 'source_mask')  # get the image directory
        self.dir_target_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')  #

        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))  # get image paths
        #self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get image paths
        #self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get image paths
        #self.target_mask_paths = sorted(make_dataset(self.dir_target_mask, opt.max_dataset_size))  # get image paths

        self.source_size = len(self.source_paths)
        #self.target_size = len(self.target_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a degraded image given a random integer index
        source_path = self.source_paths[index]
        image_name = os.path.split(source_path)[-1]
        match = re.search(r'_(\d+)\.', image_name)
        if match:
            number = match.group(1)  # 获取匹配的数字部分
            real_imagename = image_name.replace(f'_{number}', '')  # 用空字符串替换数字部分
        source_path_mask_path = os.path.join(self.dir_target_mask,real_imagename)
        target_path = os.path.join(self.dir_target,real_imagename)

        noise_good = Image.open(source_path).convert('RGB')
        real_good = Image.open(target_path).convert('RGB')
        image_mask = Image.open(source_path_mask_path).convert('L')

        # 对输入和输出进行同样的transform（裁剪也继续采用）
        noise_good_transform_params = get_params(self.opt, noise_good.size)
        noise_good_transform, noise_good_mask_transform = get_transform_six_channel(self.opt, noise_good_transform_params, grayscale=(self.input_nc == 1))

        #real_good_transform_params = get_params(self.opt, real_good.size)
        #real_good_transform, real_good_mask_transform = get_transform_six_channel(self.opt, real_good_transform_params, grayscale=(self.input_nc == 1))

        noise_good = noise_good_transform(noise_good)
        #noise_good_mask = noise_good_mask_transform(image_mask)

        real_good = noise_good_transform(real_good)
        real_good_mask = noise_good_mask_transform(image_mask)

        # return {'noise_good':noise_good , 'noise_good_mask': noise_good_mask, 
        #         'real_good ': real_good , 'real_good_mask': real_good_mask,
        #         'noise_good_path': source_path, 'real_good_path': target_path}
        return {'noise_good':noise_good ,
                'real_good': real_good , 'real_good_mask': real_good_mask,
                'noise_good_path': source_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        "degraded images should be in source image folder"
        return len(self.source_paths)
