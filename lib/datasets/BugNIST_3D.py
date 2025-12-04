import torch
import os
import glob
import numpy as np
import tifffile as tiff


class BugNIST_3D(torch.utils.data.Dataset):
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

        ids = sorted(glob.glob(os.path.join(data_path, '*/*')))

        # Make random (by seed) shuffle
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ids))
        ids = [ids[i] for i in perm]

        # Slice by fractions
        n = len(ids)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        n_train = n - n_val - n_test

        if split == 'train':
            selected = ids[:n_train]
        elif split == 'val':
            selected = ids[n_train:n_train + n_val]
        else:
            selected = ids[n_train + n_val:]

        # Build samples for the IDs
        self.samples = []
        for id in selected:
            vol = id
            label = self.class_to_idx[os.path.basename(os.path.dirname(id))]
            self.samples.append((vol, label))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.samples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        vol_path, label = self.samples[idx]

        vol = tiff.imread(vol_path) # Z, X, Y
        vol = np.expand_dims(vol, 0) # 1, Z, X, Y
        if self.transform:
            X = self.transform(vol) # Apply MONAI transform
        else:
            X = torch.from_numpy(vol).unsqueeze(0)

        return X, label