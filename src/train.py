import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BBN
from dataset import INatDataset
from utils import load_config, save_model
from plot import plot_training_curves

def train():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_dataset = INatDataset(config['data']['train_file'], transform=transform)
    val_dataset = INatDataset(config['data']['val_file'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # 模型、损失函数和优化器
    model = BBN(config['data']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])

    # 训练循环
    train_losses, val_losses, accuracies = [], [], []
    for epoch in range(config['train']['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    save_model(model, config['save_path'])
    plot_training_curves(train_losses, val_losses, accuracies)

if __name__ == '__main__':
    train()