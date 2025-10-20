from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
import h5py

import torch.optim as optim
from dataloader import AVE_Fully_Dataset
from fully_model import psp_net
from measure import compute_acc, AVPSLoss
from Optim import ScheduledOptim

import warnings
warnings.filterwarnings("ignore")
import argparse
from collections import Counter


parser = argparse.ArgumentParser(description='Fully supervised AVE localization')

# data
parser.add_argument('--model_name', type=str, default='PSP', help='model name')
parser.add_argument('--dir_video', type=str, default="./data/all_visual_feature.h5", help='visual features')
parser.add_argument('--dir_audio', type=str, default='./data/all_audio_feature.h5', help='audio features')
parser.add_argument('--dir_labels', type=str, default='./data/all_right_labels.h5', help='labels of AVE dataset')

parser.add_argument('--dir_order_train', type=str, default='./data/train_order.h5', help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='./data/val_order.h5', help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='./data/test_order.h5', help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=128, help='number of batch size')
parser.add_argument('--save_epoch', type=int, default=5, help='number of epoch for saving models')
parser.add_argument('--check_epoch', type=int, default=5, help='number of epoch for checking accuracy of current models during training')
parser.add_argument('--LAMBDA', type=float, default=100, help='weight for balancing losses')
parser.add_argument('--threshold', type=float, default=0.099, help='key-parameter for pruning process')

parser.add_argument('--lambda_temp', type=float, default=0.1, help='weight for temporal consistency loss')
parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained model')
parser.add_argument('--train', action='store_true', default=False, help='train a new model')

# 标签文件路径（根据实际路径修改）
parser.add_argument('--test_list_file', type=str, default="./data/AVE/AVE_Dataset/AVE_Dataset/test_extraction_list.txt", help='test video ID list')
parser.add_argument('--test_label_file', type=str, default="./data/AVE/AVE_Dataset/AVE_Dataset/testSet.txt", help='test label file')


FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)


def compute_class_weights(label_h5_path, num_classes=30):
    with h5py.File(label_h5_path, 'r') as f:
        labels = f['avadataset'][:]  # (N, 10, 30)
    labels_idx = labels.argmax(axis=-1).flatten()
    counter = Counter(labels_idx.tolist())
    counts = np.array([counter.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def temporal_consistency_loss(out_prob):
    diff = out_prob[:, 1:, :] - out_prob[:, :-1, :]
    return torch.mean(diff.pow(2))


def train(args, net_model, optimizer, class_weights):
    AVEData = AVE_Fully_Dataset(
        video_dir=args.dir_video,
        audio_dir=args.dir_audio,
        label_dir=args.dir_labels,
        order_dir=args.dir_order_train,
        batch_size=args.batch_size,
        status='train'
    )
    nb_batch = AVEData.__len__() // args.batch_size
    print('nb_batch:', nb_batch)
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.cuda())

    for epoch in range(args.nb_epoch):
        net_model.train()
        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_avps = 0
        epoch_loss_temp = 0
        n = 0
        SHUFFLE_SAMPLES = True
        for i in range(nb_batch):
            audio_inputs, video_inputs, labels, segment_label_batch, segment_avps_gt_batch = AVEData.get_batch(i, SHUFFLE_SAMPLES)
            SHUFFLE_SAMPLES = False

            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            labels = labels.cuda()
            segment_label_batch = segment_label_batch.cuda()
            segment_avps_gt_batch = segment_avps_gt_batch.cuda()

            net_model.zero_grad()
            fusion, out_prob, cross_att = net_model(audio_inputs, video_inputs, args.threshold)

            loss_cls = ce_loss_fn(out_prob.permute(0, 2, 1), segment_label_batch)
            loss_avps = AVPSLoss(cross_att, segment_avps_gt_batch)
            loss_temp = temporal_consistency_loss(out_prob)

            loss = loss_cls + args.LAMBDA * loss_avps + args.lambda_temp * loss_temp

            epoch_loss += loss.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_avps += loss_avps.item()
            epoch_loss_temp += loss_temp.item()

            loss.backward()
            optimizer.step_lr()
            n += 1

        labels_np = labels.cpu().detach().numpy()
        x_labels_np = out_prob.cpu().detach().numpy()
        acc = compute_acc(labels_np, x_labels_np, nb_batch)

        print("=== Epoch {%s} | Loss: %.4f cls: %.4f avps: %.4f temp: %.4f | train_acc %.4f"
              % (str(epoch), epoch_loss/n, epoch_loss_cls/n, epoch_loss_avps/n, epoch_loss_temp/n, acc))

        if epoch % args.save_epoch == 0 and epoch != 0:
            val_acc = val(args, net_model)
            print('val accuracy:', val_acc, 'epoch=', epoch)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                print('best val accuracy: {} ***************************************'.format(best_val_acc))
        if epoch % args.check_epoch == 0 and epoch != 0:
            test_acc = test(args, net_model)
            print('test accuracy:', test_acc, 'epoch=', epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                print('best test accuracy: {} ======================================='.format(best_test_acc))
                torch.save(net_model, model_name + "_" + str(epoch) + "_fully.pt")
    print('[best val accuracy]: ', best_val_acc)
    print('[best test accuracy]: ', best_test_acc)


def val(args, net_model):
    net_model.eval()
    AVEData = AVE_Fully_Dataset(
        video_dir=args.dir_video,
        audio_dir=args.dir_audio,
        label_dir=args.dir_labels,
        order_dir=args.dir_order_val,
        batch_size=420,
        status='val'
    )
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels, _, _ = AVEData.get_batch(0)

    audio_inputs = audio_inputs.cuda()
    video_inputs = video_inputs.cuda()
    labels = labels.cuda()

    with torch.no_grad():
        fusion, out_prob, cross_att = net_model(audio_inputs, video_inputs, args.threshold)

    labels_np = labels.cpu().detach().numpy()
    x_labels_np = out_prob.cpu().detach().numpy()
    acc = compute_acc(labels_np, x_labels_np, nb_batch)

    print('[val]acc: ', acc)
    return acc



def test(args, net_model, model_path=None):
    import os
    import h5py
    import torch
    import numpy as np

    if model_path is not None:
        print(f"[test] Loading model from {model_path} ...")
        model = torch.load(
            model_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        model = net_model
    model.eval()

    # ====== 加载数据 ======
    AVEData = AVE_Fully_Dataset(
        video_dir=args.dir_video,
        audio_dir=args.dir_audio,
        label_dir=args.dir_labels,
        order_dir=args.dir_order_test,
        batch_size=80,
        status='test'
    )
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels, _, _ = AVEData.get_batch(0)

    audio_inputs = audio_inputs.cuda()
    video_inputs = video_inputs.cuda()
    labels = labels.cuda()  # (N, T, C)

    with torch.no_grad():
        fusion, out_prob, cross_att = model(audio_inputs, video_inputs, args.threshold)

    # ====== 计算整体准确率 ======
    labels_np = labels.cpu().detach().numpy()
    x_labels_np = out_prob.cpu().detach().numpy()
    acc = compute_acc(labels_np, x_labels_np, nb_batch)
    print('[test] acc: ', acc)

    # ====== 模型预测的类别 (N, T) ======
    predicted_classes = out_prob.argmax(dim=-1)  # (N, T)
    true_classes = labels.argmax(dim=-1)        # (N, T)

    # ====== 读取测试视频 ID 顺序 ======
    with h5py.File(args.dir_order_test, 'r') as hf:
        order = hf['order'][:]  # (N,)
    with open(args.test_list_file, 'r', encoding='utf-8') as f:
        all_ids = [line.strip() for line in f if line.strip()]
    test_video_ids = [all_ids[idx] for idx in order]

    # ====== 读取 testSet.txt 标签 ======
    gt_ann = {}
    with open(args.test_label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("&")
            if len(parts) != 5:
                continue
            category, video_id, quality, start, end = parts
            gt_ann[video_id] = {
                "line": line.strip(),
                "category": category,
                "start": int(start),
                "end": int(end)
            }

    # ====== 检查哪些正样本被模型正确预测（≥2帧正确） ======
    correct_positive_labels = []
    for i, vid_full in enumerate(test_video_ids):
        # ✅ 使用完整视频ID匹配
        if vid_full not in gt_ann:
            # print(f"Video {vid_full} not found in gt_ann")
            continue
        ann = gt_ann[vid_full]
        true_start, true_end = ann["start"], ann["end"]
        true_cls_name = ann["category"]

        # Ground truth 类别索引
        try:
            true_cls_idx = labels_np.shape[-1] - 1 if true_cls_name == "background" else \
                           np.where(labels_np[i].sum(axis=0) > 0)[0][0]
        except:
            continue

        # 预测与真实类别
        pred_seg = predicted_classes[i, true_start:true_end]
        true_seg = true_classes[i, true_start:true_end]

        # 如果预测里有 ≥2 帧正确 → 判定为正确预测
        if len(pred_seg) > 0:
            correct_count = (pred_seg == true_seg).sum().item()
            if correct_count >= 3:
                correct_positive_labels.append(ann["line"])

    # ====== 保存结果 ======
    output_dir = "./output_labels"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_correct_positive_labels1.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in correct_positive_labels:
            f.write(item + "\n")

    print(f"=== 模型正确预测的正样本标签已保存到 {output_file}, 共 {len(correct_positive_labels)} 条 ===")

    return acc


if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)

    model_name = args.model_name
    if model_name == "PSP":
        net_model = psp_net(128, 512, 128, 30)
    else:
        raise NotImplementedError
    net_model.cuda()
    optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
    optimizer = ScheduledOptim(optimizer)

    class_weights = compute_class_weights(args.dir_labels, num_classes=30)

    if args.train:
        train(args, net_model, optimizer, class_weights)
    else:
        test_acc = test(args, net_model, model_path=args.trained_model_path)
        print("[test] accuracy: ", test_acc)
