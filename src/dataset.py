import json
import torch
from torch.utils.data import Dataset
from PIL import Image

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
        if self.transform:
            image = self.transform(image)
        return image, label