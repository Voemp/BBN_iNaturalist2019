import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import INatDataset
from model import Network
from src.core.combiner import Combiner
from src.core.function import train_model, valid_model
from utils import load_config


def train():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = config['train']['num_epochs']
    model_dir = os.path.join(config['output']['output_dir'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_dataset = INatDataset(mode="train", json_file=config['data']['train_file'])
    val_dataset = INatDataset(mode="valid", json_file=config['data']['val_file'])

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True,
                              num_workers=config['train']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False,
                            num_workers=config['train']['num_workers'], pin_memory=True)

    # 模型、损失函数和优化器
    model = Network(mode="train", num_classes=config['data']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)

    combiner = Combiner(config, device)

    # 设置起始训练参数
    start_epoch = 1
    best_result = 0
    best_epoch = 0

    csv_file = open("../data/data.csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

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
        if epoch % config['save_step'] == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, "epoch_{}.pth".format(epoch)))

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}

        valid_acc, valid_loss = valid_model(
            val_loader,
            epoch,
            model,
            criterion,
            device
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
            }, os.path.join(model_dir, "best_model.pth"))
        print("--- Training Finished: Best Epoch: {} with Accuracy: {:.2f}%".format(best_epoch, best_result * 100))

        # 将数据写入 CSV 文件
        writer.writerow([epoch, train_loss, valid_loss, valid_acc])

    csv_file.close()


if __name__ == '__main__':
    train()
