from evaluate import accuracy, AverageMeter, FusionMatrix

import numpy as np
import torch
import time


def train_model(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    combiner,
    criterion,
    cfg
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0:
            print(
                f"Epoch:{epoch:>3d}  Batch:{i:>3d}/{number_batch}  "
                f"Batch_Loss:{all_loss.val:>5.3f}  Batch_Accuracy:{acc.val * 100:>5.2f}%"
            )
    end_time = time.time()
    print(
        f"---Epoch:{epoch:>3d}/{epoch_number}   Avg_Loss:{all_loss.avg:>5.3f}   "
        f"Epoch_Accuracy:{acc.avg * 100:>5.2f}%   Epoch_Time:{(end_time - start_time) / 60:>5.2f}min---"
    )
    return acc.avg, all_loss.avg


def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True)
            loss = criterion(output, label)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), label.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        print(
            f"------- Valid: Epoch:{epoch_number:>3d}  Valid_Loss:{all_loss.avg:>5.3f}   "
            f"Valid_Acc:{acc.avg * 100:>5.2f}% -------"
        )
    return acc.avg, all_loss.avg
