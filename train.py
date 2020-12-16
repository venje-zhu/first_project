"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training ClassNSeg (the proposed method)
"""

import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from torch.optim import Adam
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse
from PIL import Image
from model.ae import Encoder
from model.ae import Decoder
from model.ae import ActivationLoss
from model.ae import ReconstructionLoss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='datasets/deepfakes', help='path to dataset')
parser.add_argument('--train_set', default ='train', help='path to train dataset')
parser.add_argument('--val_set', default ='test', help='path to validation dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.01')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay. default=0.005')
parser.add_argument('--gamma', type=float, default=1, help='weight decay. default=5')
parser.add_argument('--eps', type=float, default=1e-07, help='epsilon. default=eps=1e-07')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints/full', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)  # 返回参数1和参数2之间的任意整数
    
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')

    encoder = Encoder(3)
    decoder = Decoder(3)
    act_loss_fn = ActivationLoss()  # 激活损失衡量隐空间划分的准确度
    rect_loss_fn = ReconstructionLoss()

    optimizer_encoder = Adam(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                             weight_decay=opt.weight_decay, eps=opt.eps)
    optimizer_decoder = Adam(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                             weight_decay=opt.weight_decay, eps=opt.eps)

    if opt.resume > 0:
        encoder.load_state_dict(torch.load(os.path.join(opt.outf, 'encoder_' + str(opt.resume) + '.pt')))
        encoder.train(mode=True)

        decoder.load_state_dict(torch.load(os.path.join(opt.outf, 'decoder_' + str(opt.resume) + '.pt')))
        decoder.train(mode=True)

        optimizer_encoder.load_state_dict(torch.load(os.path.join(opt.outf, 'optim_encoder_' + str(opt.resume) + '.pt')))
        optimizer_decoder.load_state_dict(torch.load(os.path.join(opt.outf, 'optim_decoder_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer_encoder.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

            for state in optimizer_decoder.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    if opt.gpu_id >= 0:
        encoder.cuda(opt.gpu_id)
        decoder.cuda(opt.gpu_id)
        act_loss_fn.cuda(opt.gpu_id)
        rect_loss_fn.cuda(opt.gpu_id)

    # 图片正则化处理
    class Normalize_3D(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        # __call__方法使得类能够像函数一样被调用，如Normalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        def __call__(self, tensor):
            """
                Tensor: Normalized image.
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:        """
            for t, m, s in zip(tensor, self.mean, self.std):  # zip返回参数组成的元组，列优先
                t.sub_(m).div_(s)  # 减去均值除以标准差
            return tensor

    # 用于重建图像
    class UnNormalize_3D(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
                Tensor: Normalized image.
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:        """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor

    transform_tns = transforms.Compose([transforms.ToTensor(), ])

    transform_pil = transforms.Compose([transforms.ToPILImage(), ])
    
    transform_norm = Normalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform_unnorm = UnNormalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_tns)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers))

    dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_tns)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                                 shuffle=False, num_workers=int(opt.workers))

    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_act_train = 0.0
        loss_rect_train = 0.0
        loss_act_test = 0.0
        loss_rect_test = 0.0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for fft_data, labels_data in tqdm(dataloader_train):
            optimizer_encoder.zero_grad()  # 梯度清零
            optimizer_decoder.zero_grad()

            fft_label = labels_data.numpy().astype(np.float)  # [0., 1.]
            labels_data = labels_data.float()
            # rgb.shape = (batch_size, channel, height, width)
            rgb = transform_norm(fft_data[:, :, :, 0:256])  # 输入一个batch的图片，只取脸部，掩膜部分被mask保留

            mask = fft_data[:, 0, :, 256:512]  # 保存一个batch的所有图像的mask，只取0通道(0通道颜色最深)
            mask[mask >= 0.5] = 1.0
            mask[mask < 0.5] = 0.0
            mask = mask.long()

            if opt.gpu_id >= 0:
                rgb = rgb.cuda(opt.gpu_id)
                mask = mask.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            latent = encoder(rgb).reshape(-1, 2, 64, 16, 16)
            zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)
            # 激活损失基于已编码特征的两个一半激活来衡量隐空间划分的准确性
            loss_act = act_loss_fn(zero, one, labels_data)
            loss_act_data = loss_act.item()

            y = torch.eye(2)  # [[1, 0] \n [0, 1]]
            if opt.gpu_id >= 0:
                y = y.cuda(opt.gpu_id)
            y = y.index_select(dim=0, index=labels_data.data.long())
            latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)

            _, rect = decoder(latent)

            loss_rect = rect_loss_fn(rect, rgb)
            loss_rect = loss_rect * opt.gamma
            loss_rect_data = loss_rect.item()

            # loss_total = loss_act + loss_seg + loss_rect
            loss_total = loss_act + loss_rect
            loss_total.backward()

            optimizer_decoder.step()  # 前向传播更新参数，跟在backward后面
            optimizer_encoder.step()
            '''
            fft_data.shape[0]指的是batch_size
            由前面得到的one和zero的均值大小来预测图片类别
            '''
            output_pred = np.zeros((fft_data.shape[0]), dtype=np.float)
            for i in range(fft_data.shape[0]):
                if one[i] >= zero[i]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0
            # 每次训练将一个batch_size的图片传进网络，用这个方法把所有的label连接起来
            tol_label = np.concatenate((tol_label, fft_label))  # 真实label
            tol_pred = np.concatenate((tol_pred, output_pred))  # 预测label

            loss_act_train += loss_act_data
            # loss_seg_train += loss_seg_data
            loss_rect_train += loss_rect_data
            count += 1

        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_act_train /= count
        # loss_seg_train /= count
        loss_rect_train /= count

        ########################################################################
        # do checkpointing & validation
        if epoch % 100 == 0:
            torch.save(encoder.state_dict(), os.path.join(opt.outf, 'encoder_%d.pt' % epoch))
            torch.save(optimizer_encoder.state_dict(), os.path.join(opt.outf, 'optim_encoder_%d.pt' % epoch))

            torch.save(decoder.state_dict(), os.path.join(opt.outf, 'decoder_%d.pt' % epoch))
            torch.save(optimizer_decoder.state_dict(), os.path.join(opt.outf, 'optim_decoder_%d.pt' % epoch))

        encoder.eval()
        decoder.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        count = 0

        for fft_data, labels_data in tqdm(dataloader_val):

            fft_label = labels_data.numpy().astype(np.float)
            labels_data = labels_data.float()

            rgb = transform_norm(fft_data[:, :, :, 0:256])
            mask = fft_data[:, 0, :, 256:512]  # 只取零通道
            mask[mask >= 0.5] = 1.0
            mask[mask < 0.5] = 0.0
            mask = mask.long()

            if opt.gpu_id >= 0:
                rgb = rgb.cuda(opt.gpu_id)
                mask = mask.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            latent = encoder(rgb).reshape(-1, 2, 64, 16, 16)

            zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)

            loss_act = act_loss_fn(zero, one, labels_data)
            loss_act_data = loss_act.item()

            y = torch.eye(2)
            if opt.gpu_id >= 0:
                y = y.cuda(opt.gpu_id)

            y = y.index_select(dim=0, index=labels_data.data.long())

            latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)

            _, rect = decoder(latent)

            loss_rect = rect_loss_fn(rect, rgb)
            loss_rect = loss_rect * opt.gamma
            loss_rect_data = loss_rect.item()

            output_pred = np.zeros((fft_data.shape[0]), dtype=np.float)

            for i in range(fft_data.shape[0]):
                if one[i] >= zero[i]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, fft_label))
            tol_pred = np.concatenate((tol_pred, output_pred))
            '''
            zero和one均保存每个图片属于zero和one的类别的均值（个人觉得可以理解为类似概率的表示，如果属于0类别，则对应的均值应该大一些），
            zero和one分别对应labels的class，为行向量，zero表示假fake，one表示真real
            zero.shape[0]得到的是batch_size，reshape操作将标量变为[batch_size, 1]列向量，其中每一行对应每一个样本属于0类的均值
            torch.cat将zero和one连接在一起，以dim=1即列的形式组合，最终得到的是[zero, one]，每行对应每个样本属于0类1类的均值
            softmax以dim=1即每列，计算本batch每一个样本属于0 1类别的概率，shape与输入一致
            '''
            pred_prob = torch.softmax(torch.cat((zero.reshape(zero.shape[0], 1), one.reshape(one.shape[0], 1)), dim=1), dim=1)
            tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:, 1].data.cpu().numpy()))

            loss_act_test += loss_act_data
            loss_rect_test += loss_rect_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_act_test /= count
        loss_rect_test /= count

        '''
        https://blog.csdn.net/w1301100424/article/details/84546194
        roc_curve接收参数：
        tol_label: 真实的样本标签，默认为{0，1}或者{-1，1}。如果要设置为其它值，则 pos_label 参数要设置为特定值。
                    例如要令样本标签为{1，2}，其中2表示正样本，则pos_label=2。
        tol_pred_prob：每个样本的预测结果
        pos_label: 正样本标签
        
        返回值：
        fpr: false positive rate
        tpr: true positive rate
        thresholds: fpr和tpr确定的阈值
        '''
        fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        print('[Epoch %d] Train: act_loss: %.4f  rect_loss: %.4f  acc: %.2f '
              '| Test: act_loss: %.4f  rect_loss: %.4f  acc: %.2f  eer: %.2f' %
              (epoch, loss_act_train, loss_rect_train, acc_train*100, loss_act_test, loss_rect_test, acc_test*100, eer*100))

        text_writer.write('%d,%.4f,%.4f,%.2f,%.4f,%.4f,%.2f,%.2f\n' %
                          (epoch, loss_act_train, loss_rect_train, acc_train*100, loss_act_test, loss_rect_test, acc_test*100, eer*100))

        text_writer.flush()

        ########################################################################

        real_img = transform_tns(Image.open(os.path.join(opt.outf, 'train_recon_image/real.jpg'))).unsqueeze(0)[:, :, :, 0:256]
        fake_img = transform_tns(Image.open(os.path.join(opt.outf, 'train_recon_image/fake.jpg'))).unsqueeze(0)[:, :, :, 0:256]

        rgb = torch.cat((real_img, fake_img), dim=0)  # [2, 3, 256, 256]，以batch_size形式组合两张图片
        rgb = transform_norm(rgb)

        # real = 1, fake = 0
        labels_data = torch.FloatTensor([1,0])

        if opt.gpu_id >= 0:
            rgb = rgb.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        latent = encoder(rgb).reshape(-1, 2, 64, 16, 16)

        zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)

        y = torch.eye(2)

        if opt.gpu_id >= 0:
            y = y.cuda(opt.gpu_id)

        y = y.index_select(dim=0, index=labels_data.data.long())

        latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)

        _, rect = decoder(latent)
        '''
        https://blog.csdn.net/qq_39709535/article/details/80804003
        detach()的作用：detach()返回一个新的torch Variable，将该Variable的参数从网络中隔离，不参与参数更新，
        当反向传播经过这个node时，梯度不会从这个node往前面传播
        '''

        rect = transform_unnorm(rect).detach().cpu()

        real_img = transform_pil(rect[0])  # 获取重建图像
        fake_img = transform_pil(rect[1])

        real_img.save(os.path.join(opt.outf, 'image', 'real_' + str(epoch).zfill(3) + '.jpg'))
        fake_img.save(os.path.join(opt.outf, 'image', 'fake_' + str(epoch).zfill(3) + '.jpg'))

        encoder.train(mode=True)
        decoder.train(mode=True)

    text_writer.close()
