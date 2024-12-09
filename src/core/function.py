import time

import numpy as np
import torch
from tqdm import tqdm

from src.core.evaluate import AverageMeter, FusionMatrix, accuracy


def train_model(
        trainLoader,
        model,
        epoch,
        epoch_number,
        optimizer,
        combiner,
        criterion
):
    model.train()

    combiner.reset_epoch(epoch)

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

        if i % 25 == 0:
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
        dataLoader, epoch_number, model, criterion, device
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
            f"------- Valid: Epoch:{epoch_number:>3d}  Valid_Loss:{all_loss.avg:>5.3f}  "
            f"Valid_Acc:{acc.avg * 100:>5.2f}% -------"
        )
    return acc.avg, all_loss.avg


def test_model(dataLoader, model, device):
    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    top1_count, top2_count, top3_count, index, fusion_matrix = ([], [], [], 0, FusionMatrix(num_classes),)

    func = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)
            result = func(output)
            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()
            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )
    pbar.close()
