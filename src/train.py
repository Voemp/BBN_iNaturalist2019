import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import INatDataset
from model import Network
from plot import plot_training_curves
from src.core.combiner import Combiner
from src.core.function import train_model, valid_model
from utils import load_config, save_model


def train():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = config['train']['num_epochs']
    model_dir = os.path.join(config['output']['output_dir'], "models")


    # 硬编码的参数
    input_size = (100, 100)
    train_transforms = ["random_resized_crop", "random_horizontal_flip"]

    # 构建 transform
    transform = transforms.Compose([])

    if "random_resized_crop" in train_transforms:
        transform.transforms.append(
            transforms.RandomResizedCrop(
                size=input_size,  # 使用硬编码的输入尺寸
                scale=(0.08, 1.0),  # 固定 scale 范围
                ratio=(3. / 4., 4. / 3.)  # 固定 ratio 范围
            )
        )

    if "random_horizontal_flip" in train_transforms:
        transform.transforms.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

    # 添加 ToTensor() 转换
    transform.transforms.append(transforms.ToTensor())

    train_dataset = INatDataset(config['data']['train_file'], transform=transform)
    val_dataset = INatDataset(config['data']['val_file'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # 模型、损失函数和优化器
    model = Network(mode="train", num_classes=config['data']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)

    combiner = Combiner(config, device)

    # close loop
    # code_dir = os.path.join(config['output']['output_dir'], config['output']['name'], "codes")

    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # else:
    #     print(
    #         "This directory has already existed, Please remember to modify your cfg.output.name"
    #     )
    #     if not input("\033[1;31;40mContinue and override the former directory? (y/n): \033[0m").lower().startswith('y'):
    #         exit(0)
    #     rmtree(code_dir)  # 删除旧的代码目录

    # print("=> output model will be saved in {}".format(model_dir))
    # this_dir = os.path.dirname(__file__)
    # ignore = shutil.ignore_patterns(
    #     "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    # )

    # shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    # 设置起始训练参数
    start_epoch = 1
    best_result = 0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs + 1):
        train_acc, train_loss = train_model(
            train_loader,
            model,
            epoch,
            num_epochs,
            optimizer,
            combiner,
            criterion
        )
        scheduler.step()
        model_save_path = os.path.join(model_dir, "epoch_{}.pth".format(epoch))
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
        print(
            f"Epoch {epoch + 1}/{config['train']['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    save_model(model, config['save_path'])
    plot_training_curves(train_losses, val_losses, accuracies)


if __name__ == '__main__':
    train()
