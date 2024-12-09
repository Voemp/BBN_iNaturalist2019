import torch
from torch.utils.data import DataLoader

from dataset import INatDataset
from model import Network
from src.core.function import test_model
from utils import load_config


def test():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = INatDataset(config['data']['test_file'])
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False,
                             num_workers=config['train']['num_workers'], pin_memory=True)

    # 加载模型
    model = Network(mode="test", num_classes=config['data']['num_classes']).to(device)
    model.load_model("../data/models/epoch_1.pth", device)

    test_model(test_loader, model, device)


if __name__ == '__main__':
    test()
