import torch.backends.cudnn as cudnn
import sys
import argparse
import warnings
import ast
import stat
from core.loss import *
from config import cfg
import os
import shutil
import click
from core.function import train_model, valid_model
from utils import convert
from utils.utils import (
    get_model,
    get_category_list,
)
from core.combiner import Combiner
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter




if __name__ == "__main__":
    # 格式化json文件
    convert("train")
    convert("val")

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)

    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    num_class_list = [0] * num_classes
    for anno in annotations:
        category_id = anno["category_id"]
        num_class_list[category_id] += 1

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }

    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    epoch_number = cfg.TRAIN.MAX_EPOCH

    # 获取模型
    model = get_model(cfg, num_classes, device)   # 不冻结模型，使用gpu

    # 创建 Combiner 实例
    combiner = Combiner(cfg, device)

    # 设置优化器和学习率调度器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)

    # 最终模型构建器

# 训练数据集
    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
# 验证数据集
    validLoader = DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,  # 不打乱数据顺序，因为验证集数据应该按顺序进行评估
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # 关闭流程
    # 定义保存模型、代码和TensorBoard日志的目录路径
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")  # 模型保存路径
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")  # 代码保存路径
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")  # TensorBoard日志保存路径
        if cfg.TRAIN.TENSORBOARD.ENABLE  # 如果启用了TensorBoard
        else None
    )

    # 如果模型保存目录不存在，则创建该目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        # 如果用户同意覆盖已有目录，则执行覆盖操作
        if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
        ):
            exit(0)  # 用户选择不继续时退出
        # 删除旧的代码目录
        shutil.rmtree(code_dir)  # shutil.rmtree(code_dir)
        # 如果存在TensorBoard日志目录，则删除该目录
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)

    # 输出模型保存目录
    print("=> output model will be saved in {}".format(model_dir))

    # 获取当前脚本所在的目录
    this_dir = os.path.dirname(__file__)

    # 设置忽略拷贝的文件类型（如.pyc文件、.so文件等）
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )

    # 将当前目录（除忽略的文件类型外）的内容复制到代码目录
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    # 如果启用了TensorBoard日志，则创建并初始化TensorBoard日志
    if tensorboard_dir is not None:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)  # 创建一个假的输入张量，用于生成模型的图
        writer = SummaryWriter(log_dir=tensorboard_dir)  # 初始化TensorBoard writer
        writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input,))  # 将模型图添加到TensorBoard
    else:
        writer = None  # 如果没有启用TensorBoard，则不创建writer

    # 初始化一些变量
    best_result, best_epoch, start_epoch = 0, 0, 1  # 最佳结果、最佳epoch和开始epoch的初始化

    # 训练循环开始
    print(
        "-------------------Train start :{}  {}  {}-------------------".format(
            cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
        )
    )
    # 遍历每个 epoch，从 start_epoch 到 epoch_number
    for epoch in range(start_epoch, epoch_number + 1):
        scheduler.step()  # 更新学习率调度器

        # 调用 train_model 函数进行训练，返回训练集上的准确率和损失
        train_acc, train_loss = train_model(
            trainLoader,  # 训练数据加载器
            model,  # 当前模型
            epoch,  # 当前 epoch
            epoch_number,  # 总 epoch 数
            optimizer,  # 优化器
            combiner,  # 混合方法
            criterion,  # 损失函数
            cfg,  # 配置
            writer=writer,  # TensorBoard 日志写入器
        )

        # 设置当前 epoch 模型保存路径
        model_save_path = os.path.join(
            model_dir,  # 模型保存目录
            "epoch_{}.pth".format(epoch),  # 文件名
        )

        # 每隔 cfg.SAVE_STEP 个 epoch 保存模型
        if epoch % cfg.SAVE_STEP == 0:
            torch.save({  # 保存模型的状态字典和训练状态
                'state_dict': model.state_dict(),  # 模型参数
                'epoch': epoch,  # 当前 epoch
                'best_result': best_result,  # 最佳准确率
                'best_epoch': best_epoch,  # 最佳 epoch
                'scheduler': scheduler.state_dict(),  # 学习率调度器状态
                'optimizer': optimizer.state_dict()  # 优化器状态
            }, model_save_path)

        # 初始化用于记录损失和准确率的字典
        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}

        # 如果配置了验证步骤，并且当前 epoch 是验证的倍数
        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            # 在验证集上评估模型，返回验证集的准确率和损失
            valid_acc, valid_loss = valid_model(
                validLoader,  # 验证数据加载器
                epoch,  # 当前 epoch
                model,  # 当前模型
                cfg,  # 配置
                criterion,  # 损失函数
                device,  # 设备（CPU 或 GPU）
                writer=writer,  # TensorBoard 日志写入器
            )

            # 将验证损失和准确率记录到字典中
            loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc

            # 如果验证集准确率超过之前的最佳结果
            if valid_acc > best_result:
                # 更新最佳结果和最佳 epoch
                best_result, best_epoch = valid_acc, epoch

                # 保存当前的最佳模型
                torch.save({
                    'state_dict': model.state_dict(),  # 模型参数
                    'epoch': epoch,  # 当前 epoch
                    'best_result': best_result,  # 最佳准确率
                    'best_epoch': best_epoch,  # 最佳 epoch
                    'scheduler': scheduler.state_dict(),  # 学习率调度器状态
                    'optimizer': optimizer.state_dict(),  # 优化器状态
                }, os.path.join(model_dir, "best_model.pth"))

            # 输出当前的最佳结果到控制台
            print(
                "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                    best_epoch, best_result * 100
                )
            )

        # 如果启用了 TensorBoard，将训练和验证的损失与准确率写入日志
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/acc", acc_dict, epoch)  # 写入准确率
            writer.add_scalars("scalar/loss", loss_dict, epoch)  # 写入损失

    # 如果启用了 TensorBoard，在训练结束后关闭日志写入器
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()

    # 输出训练完成信息到控制台
    print(
        "-------------------Train Finished :{}-------------------".format(cfg.NAME)
    )
