from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger, common
from dataset.dataset_lits_test import Test_Dataset, TestDataset
import SimpleITK as sitk
import os
import numpy as np
from models import ResUNet
from utils.common import to_one_hot_3d
from utils.metrics import DiceAverage
from collections import OrderedDict


def predict_one_img(model, img_dataset, args, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.merge_segmentation_results()
    pred = torch.argmax(pred, dim=1)

    pred_img = common.to_one_hot_3d(pred, args.n_labels)
    test_dice.update(pred_img, target)

    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels == 3:
        test_dice.update({'Dice_tumor': test_dice.avg[2]})

    pred = np.asarray(pred.numpy(), dtype='uint8')
    if args.postprocess:
        pass  # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))

    return test_dice, pred


def test(model, test_loader, args, device):
    model.eval()  # 设置模型为评估模式
    test_dice = DiceAverage(args.n_labels)

    with torch.no_grad():  # 在测试过程过不需要计算梯度
        total_correct = 0
        total_samples = 0

        for data, label in test_loader:
            data, label = data.float(), label.long()
            label = common.to_one_hot_3d(label, args.n_labels)
            # 将输入数据和标签转移到设备（GPU）上
            data, label = data.to(device), label.to(device)


def main():
    # 必要参数
    args = config.args

    # 创建保存路径
    save_path = os.path.join('./experiments', args.save)
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    # 创建测试数据加载器
    test_loader = DataLoader(dataset=TestDataset(args),
                             batch_size=1,
                             shuffle=False)

    # 初始化模型，加载已经训练的模型参数
    model = ResUNet(in_channel=1, out_channel=args.n_labels, training=False)
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    # 在GPU上进行训练
    device = torch.device('cpu' if args.cpu else 'cuda')
    model.to(device)

    test_log = logger.Logger(save_path, "test_log")

    test(model, test_loader, args, device)
    datasets = Test_Dataset(args.test_data_path, args=args)
    for img_dataset, file_idx in datasets:
        test_dice, pred_img = predict_one_img(model, img_dataset, args, device)
        test_log.update_test(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-' + file_idx + '.gz'))


if __name__ == '__main__':
    main()
