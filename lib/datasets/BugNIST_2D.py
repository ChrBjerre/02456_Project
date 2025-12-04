import torch
import os
import glob
import numpy as np
import tifffile as tiff
from PIL import Image

class BugNIST_2D(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', val_frac=0.15, test_frac=0.15,
                 seed=42, transform=None):
        'Initialization'

        assert split in {'train', 'val', 'validate', 'test'}

        if split == 'validate':
            split = 'val'

        self.transform = transform
        self.data_path = data_path

        # Get all labels
        self.classes = [os.path.basename(path) for path in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(path)]
        self.classes.sort()

        # Class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        firsts = sorted(glob.glob(os.path.join(data_path, '*/*_0.png')))

        # Make random (by seed) shuffle
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(firsts))
        firsts = [firsts[i] for i in perm]

        # Slice by fractions
        n = len(firsts)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        n_train = n - n_val - n_test

        if split == 'train':
            selected = firsts[:n_train]
        elif split == 'val':
            selected = firsts[n_train:n_train + n_val]
        else:
            selected = firsts[n_train + n_val:]

        self.length = len(selected)

        # Build samples for the IDs
        self.samples = []
        for first_image in selected:
            label = self.class_to_idx[os.path.basename(os.path.dirname(first_image))]
            second_image = first_image.replace('_0.png', '_1.png')
            third_image = first_image.replace('_0.png', '_2.png')
            fourth_image = first_image.replace('_0.png', '_3.png')
            fith_image = first_image.replace('_0.png', '_4.png')
            sixth_image = first_image.replace('_0.png', '_5.png')
            self.samples.append((first_image, second_image, third_image, fourth_image, fith_image, sixth_image, label))

    def __len__(self):
        'Returns the total number of samples'
        return self.length

    def __getitem__(self, idx):
        im1_path, im2_path, im3_path, im4_path, im5_path, im6_path, label = self.samples[idx]

        paths = [im1_path, im2_path, im3_path, im4_path, im5_path, im6_path]

        # Load and transform each image
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")  # RGB
            if self.transform is None:
                raise ValueError("No transform provided!")
            im = self.transform(im)
            imgs.append(im)

        # Stack into a single tensor: [6, 3, H, W]
        vol = torch.stack(imgs, dim=0)

        return vol, label
    


class BugNIST_2D_2views(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', val_frac=0.15, test_frac=0.15,
                 seed=42, transform=None):
        'Initialization'

        assert split in {'train', 'val', 'validate', 'test'}

        if split == 'validate':
            split = 'val'

        self.transform = transform
        self.data_path = data_path

        # Get all labels
        self.classes = [os.path.basename(path) for path in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(path)]
        self.classes.sort()

        # Class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        firsts = sorted(glob.glob(os.path.join(data_path, '*/*_0.png')))

        # Make random (by seed) shuffle
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(firsts))
        firsts = [firsts[i] for i in perm]

        # Slice by fractions
        n = len(firsts)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        n_train = n - n_val - n_test

        if split == 'train':
            selected = firsts[:n_train]
        elif split == 'val':
            selected = firsts[n_train:n_train + n_val]
        else:
            selected = firsts[n_train + n_val:]

        self.length = len(selected)

        # Build samples for the IDs
        self.samples = []
        for first_image in selected:
            label = self.class_to_idx[os.path.basename(os.path.dirname(first_image))]
            fourth_image = first_image.replace('_0.png', '_3.png')
            self.samples.append((first_image, fourth_image, label))

    def __len__(self):
        'Returns the total number of samples'
        return self.length

    def __getitem__(self, idx):
        im1_path, im4_path, label = self.samples[idx]

        paths = [im1_path, im4_path]

        # Load and transform each image
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")  # RGB
            if self.transform is None:
                raise ValueError("No transform provided!")
            im = self.transform(im)
            imgs.append(im)

        # Stack into a single tensor: [2, 3, H, W]
        vol = torch.stack(imgs, dim=0)

        return vol, label
    

class BugNIST_2D_1views(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', val_frac=0.15, test_frac=0.15,
                 seed=42, transform=None):
        'Initialization'

        assert split in {'train', 'val', 'validate', 'test'}

        if split == 'validate':
            split = 'val'

        self.transform = transform
        self.data_path = data_path

        # Get all labels
        self.classes = [os.path.basename(path) for path in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(path)]
        self.classes.sort()

        # Class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        firsts = sorted(glob.glob(os.path.join(data_path, '*/*_0.png')))

        # Make random (by seed) shuffle
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(firsts))
        firsts = [firsts[i] for i in perm]

        # Slice by fractions
        n = len(firsts)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        n_train = n - n_val - n_test

        if split == 'train':
            selected = firsts[:n_train]
        elif split == 'val':
            selected = firsts[n_train:n_train + n_val]
        else:
            selected = firsts[n_train + n_val:]

        self.length = len(selected)

        # Build samples for the IDs
        self.samples = []
        for first_image in selected:
            label = self.class_to_idx[os.path.basename(os.path.dirname(first_image))]
            self.samples.append((first_image, label))

    def __len__(self):
        'Returns the total number of samples'
        return self.length

    def __getitem__(self, idx):
        im1_path, label = self.samples[idx]

        paths = [im1_path]

        # Load and transform each image
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")  # RGB
            if self.transform is None:
                raise ValueError("No transform provided!")
            im = self.transform(im)
            imgs.append(im)

        # Stack into a single tensor: [1, 3, H, W]
        vol = torch.stack(imgs, dim=0)

        return vol, label