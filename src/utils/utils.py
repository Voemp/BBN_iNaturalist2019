import logging
import time
import os

import torch
from ..net import Network


# def create_logger(cfg):
#     dataset = cfg.DATASET.DATASET
#     net_type = cfg.BACKBONE.TYPE
#     module_type = cfg.MODULE.TYPE
#     log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     time_str = time.strftime("%Y-%m-%d-%H-%M")
#     log_name = "{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
#     log_file = os.path.join(log_dir, log_name)
#     # set up logger
#     print("=> creating log {}".format(log_file))
#     head = "%(asctime)-15s %(message)s"
#     logging.basicConfig(filename=str(log_file), format=head)
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     console = logging.StreamHandler()
#     logging.getLogger("").addHandler(console)
#
#     logger.info("---------------------Cfg is set as follow--------------------")
#     logger.info(cfg)
#     logger.info("-------------------------------------------------------------")
#     return logger, log_file


# def get_optimizer(cfg, model):
#     base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
#     params = []
#
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             params.append({"params": p})
#
#     if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":
#         optimizer = torch.optim.SGD(
#             params,
#             lr=base_lr,
#             momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
#             weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
#             nesterov=True,
#         )
#     elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
#         optimizer = torch.optim.Adam(
#             params,
#             lr=base_lr,
#             betas=(0.9, 0.999),
#             weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
#         )
#     return optimizer


# def get_scheduler(cfg, optimizer):
#     if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep": #基于多步衰减的学习率调度器
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer,
#             cfg.TRAIN.LR_SCHEDULER.LR_STEP,
#             gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
#         )
#     elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine": #基于余弦退火的学习率调度器
#         if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
#             )
#         else:
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=1e-4
#             )
#     elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup": #自定义的调度器，结合了 warm-up 阶段和多步衰减，通常用于更平稳的学习率调整
#         scheduler = WarmupMultiStepLR(
#             optimizer,
#             cfg.TRAIN.LR_SCHEDULER.LR_STEP,
#             gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
#             warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
#         )
#     else:
#         raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))
#
#     return scheduler


def get_model(cfg, num_classes, device):
    model = Network(cfg, mode="train", num_classes=num_classes)

    if cfg.BACKBONE.FREEZE == True:  # 设置为 True，它会冻结模型的骨干网络
        model.freeze_backbone()
        print("Backbone has been freezed")

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model

def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list