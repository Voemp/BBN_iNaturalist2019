import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Network
from dataset import INatDataset
from src.core.combiner import Combiner
from utils import load_config, save_model
from plot import plot_training_curves
import os
import shutil
from shutil import rmtree
from core.function import train_model, valid_model
from core.evaluate import AverageMeter, accuracy


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
    num_epochs=config['train']['num_epochs']

    # 模型、损失函数和优化器
    model = Network(config['data']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3, 6, 9],gamma=0.1)

    combiner = Combiner(config, device)

    # close loop
    model_dir = os.path.join(config['output']['output_dir'], config['output']['name'], "models")
    code_dir = os.path.join(config['output']['output_dir'], config['output']['name'], "codes")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print(
            "This directory has already existed, Please remember to modify your cfg.output.name"
        )
        if not input("\033[1;31;40mContinue and override the former directory? (y/n): \033[0m").lower().startswith('y'):
            exit(0)
        rmtree(code_dir)  # 删除旧的代码目录

    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    best_result, best_epoch, start_epoch = 0, 0, 1

    # 设置起始训练参数
    start_epoch = 1
    best_result = 0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs + 1):
        scheduler.step()
        train_acc, train_loss = train_model(
            train_loader,
            model,
            epoch,
            num_epochs,
            optimizer,
            combiner,
            criterion,
            config  # ???????????????
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % config['save_step'] == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if config['valid_step'] != -1 and epoch % config['valid_step'] == 0:
            valid_acc, valid_loss = valid_model(
                val_loader, epoch, model, config, criterion, device
            )
            loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
            if valid_acc > best_result:
                best_result, best_epoch = valid_acc, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model.pth")
                )
        print("Training Finished: Best Epoch: {} with Accuracy: {:.2f}%".format(best_epoch, best_result * 100))

























































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