from typing import Type, Callable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict
import config
import os

from models import UNet, ResUNet, KiUNet_min, SegNet

from dataset.dataset_lits import LiTSDataset

from utils import metrics, common, weights_init, logger, loss_func


def val(model, val_loader, criterion, n_labels, device):
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 在评估过程不需要计算梯度
        val_loss = metrics.LossAverage()
        val_dice = metrics.DiceAverage(n_labels)

        for idx, (inputs, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            # 处理数据，并将数据放到设备上
            inputs, labels = inputs.float(), labels.long()
            labels = common.to_one_hot_3d(labels, n_labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss.update(loss.item(), inputs.size(0))

            # 计算dice
            val_dice.update(outputs, labels)

        # 计算验证集上的平均损失和dice
        dice_average = val_dice.get_average()[1] if n_labels == 2 else val_dice.get_average()[2]
        val_log = OrderedDict({'Val_Loss': val_loss.get_average(), 'Val_dice_liver': dice_average})
        return val_log


def train(model, train_loader, optimizer, criterion: Callable, n_labels, alpha, device):
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss0 = criterion(output[0], target)
        loss1 = criterion(output[1], target)
        loss2 = criterion(output[2], target)
        loss3 = criterion(output[3], target)

        loss = loss3 + alpha * (loss0 + loss1 + loss2)
        loss.backward()
        optimizer.step()

        train_loss.update(loss3.item(), data.size(0))
        train_dice.update(output[3], target)

    dice_average = train_dice.get_average()[1] if n_labels == 2 else train_dice.get_average()[2]
    val_log = OrderedDict({'Train_Loss': train_loss.get_average(), 'Train_dice_liver': dice_average})
    return val_log


def main():
    # 必要参数
    args = config.args

    # 创建保存路径
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 划分训练集和验证集，并创建数据加载器
    train_loader: DataLoader[Type[LiTSDataset]] = DataLoader(dataset=LiTSDataset(args, 'train'),
                                                             batch_size=args.batch_size,
                                                             num_workers=args.n_threads,
                                                             shuffle=True)
    val_loader: DataLoader[Type[LiTSDataset]] = DataLoader(dataset=LiTSDataset(args, 'val'),
                                                           batch_size=1,
                                                           num_workers=args.n_threads,
                                                           shuffle=False)

    # 初始化模型、损失函数、优化器
    model: torch.nn.Module = ResUNet(in_channel=1, out_channel=args.n_labels, training=True)
    model.apply(weights_init.init_model)
    common.print_network(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = loss_func.TverskyLoss()

    # 在GPU上进行训练
    device = torch.device('cpu' if args.cpu else 'cuda')
    model.to(device)

    log = logger.Logger(save_path, 'train_log')

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop计数器
    alpha = 0.4  # 深监督衰减系数初始值

    # 训练和验证的交替过程
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        print('========Epoch:{}=======lr:{}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_log = train(model, train_loader, optimizer, criterion, args.n_labels, alpha, device)
        val_log = val(model, val_loader, criterion, args.n_labels, device)
        log.update_train_val(epoch, train_log, val_log)

        # Save checkpoint
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch:{}|{}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0:
            alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print('=> early stopping')
                break
        torch.cuda.empty_cache()

    log.close()


if __name__ == '__main__':
    main()
