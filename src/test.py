import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import INatDataset
from model import Network
from src.core.function import test_model
from utils import load_config, load_model

from src.core.evaluate import FusionMatrix


def test():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = (100, 100)

    # 构建 transform
    transform = transforms.Compose([
        transforms.Resize(int(input_size[0] / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    test_dataset = INatDataset(config['data']['test_file'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    print(f"Number of test samples: {len(test_loader.dataset)}")

    # 加载模型
    model = Network(mode="test", num_classes=config['data']['num_classes']).to(device)
    model = load_model(model, "F:\\iNaturalist\\resnet50-19c8e357.pth")

    print("Model loaded successfully")

    # 初始化混淆矩阵
    fusion_matrix = FusionMatrix(config['data']['num_classes'])
    top1_count, top2_count, top3_count, index = [], [], [], 0

    # 测试
    model.eval()
    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(test_loader):
            image = image.to(device)
            output = model(image)
            result = torch.nn.Softmax(dim=1)(output)
            _, top_k = result.topk(5, 1, True, True)

            # 更新混淆矩阵
            fusion_matrix.update(result.argmax(dim=1).cpu().numpy(), image_labels.numpy())

            # 统计 Top-k 准确率
            topk_result = top_k.cpu().tolist()
            top1_count += [topk_result[0][0] == image_labels[i]]
            top2_count += [image_labels[i] in topk_result[0:2]]
            top3_count += [image_labels[i] in topk_result[0:3]]
            index += 1

            now_acc = np.sum(top1_count) / index
            print(f"Now Top1:{now_acc * 100:.2f}%")

    # 输出最终精度
    top1_acc = np.mean(top1_count)
    top2_acc = np.mean(top2_count)
    top3_acc = np.mean(top3_count)
    print(f"Top1: {top1_acc * 100:.2f}% Top2: {top2_acc * 100:.2f}% Top3: {top3_acc * 100:.2f}%")

if __name__ == '__main__':
    test()
