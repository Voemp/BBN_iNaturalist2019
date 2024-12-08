import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BBN
from dataset import INatDataset
from utils import load_config, load_model

def test():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    test_dataset = INatDataset(config['data']['test_file'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # 加载模型
    model = BBN(config['data']['num_classes']).to(device)
    model = load_model(model, config['save_path'])
    model.eval()

    # 测试
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    test()