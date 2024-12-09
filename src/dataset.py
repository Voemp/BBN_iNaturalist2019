import json

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class INatDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.annotations = data['annotations']
        self.num_classes = data['num_classes']
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        image = Image.open(anno['fpath']).convert('RGB')
        label = anno['category_id']

        # 获取混合样本
        sample_idx = np.random.randint(0, len(self.annotations))
        sample_anno = self.annotations[sample_idx]
        sample_image = Image.open(sample_anno['fpath']).convert('RGB')
        sample_label = sample_anno['category_id']

        if self.transform:
            image = self.transform(image)
            sample_image = self.transform(sample_image)

        meta = {"sample_image": sample_image, "sample_label": sample_label}
        return image, label, meta
