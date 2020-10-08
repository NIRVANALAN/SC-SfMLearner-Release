import argparse
import pdb
import time
import csv
import datetime
import sys
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import models

import custom_transforms
from torchsummary import summary
from utils import tensor2array, save_checkpoint
from datasets.refresh_loader import RefreshDataset
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=[
                    'sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N',
                    help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float,
                    metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0,
                    type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv',
                    metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv',
                    metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true',
                    help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales',
                    type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-g', '--dmv-weight', type=float,
                    help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-l', '--dsv-weight', type=float,
                    help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('-s', '--sceneflow-weight', type=float,
                    help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-e', '--smooth-loss-weight', type=float,
                    help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--with-ssim', type=int,
                    default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1,
                    help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0,
                    help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1,
                    help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str,
                    choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp',
                    default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None,
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--up_conv1_inplanes', help='set in_channel number for UNet', nargs='+',
                    default=[192, 512, 768, 1280, 1024], type=list)
parser.add_argument('--up_conv1_outplane', help='set out_channel number for UNet', nargs='+',
                    default=[128, 256, 512, 512, 512], type=list)
parser.add_argument(
    '--in_nf', help='input channel number for UNet', default=4, type=int)
parser.add_argument('--pretrain', default=True, help='pretrain dfnet')


best_loss = sys.maxsize
n_iter = 0
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.autograd.set_detect_anomaly(True)  # context manager,for debug only.


def main():
    global best_loss, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()  # * donot raise error if the dir exists

    #! reproducibility
    # if args.reproducibility:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    #! benchmark flag enables the inbuilt cudnn auto-tuner to find the best algorithm for the algorithm
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize_img = custom_transforms.Normalize2(mean=[0.45, 0.45, 0.45],
                                                 std=[0.225, 0.225, 0.225])
    normalize_dmv = custom_transforms.Normalize2(
        mean=[750], std=[100],)
    normalize_dsv = custom_transforms.Normalize2(
        mean=[3000], std=[1000])  # for DMV
    transform = [normalize_img, normalize_dmv, normalize_dsv]

    # train_transform = custom_transforms.Compose([
    #     # custom_transforms.RandomHorizontalFlip(),
    #     # custom_transforms.RandomScaleCrop(),
    #     # custom_transforms.ArrayToTensor2(),
    #     normalize
    # ])
    # valid_transform = custom_transforms.Compose(
    #     [
    #         # custom_transforms.ArrayToTensor(),
    #         normalize
    #     ])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = RefreshDataset(list_dir='train.txt', transform=transform)
    # train_set = SequenceFolder(
    #     args.data,
    #     transform=train_transform,
    #     seed=args.seed,
    #     train=True,
    #     sequence_length=args.sequence_length,
    #     dataset=args.dataset
    # )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    # from datasets.validation_folders import ValidationSet
    val_set = RefreshDataset(list_dir='val.txt', transform=transform)
    # val_set = ValidationSet(
    #     args.data,
    #     transform=valid_transform,
    #     dataset=args.dataset
    # )
    print('{} samples found in train dataset'.format(
        len(train_set)))
    print('{} samples found in valid dataset'.format(
        len(val_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    df_net = models.FusionUNet(
        args.in_nf, args.up_conv1_inplanes, args.up_conv1_outplane).to(device)
    INPUT_SHAPE = (256, 256)
    INPUT_NF = args.in_nf
    print(summary(df_net, [
          (INPUT_NF, *INPUT_SHAPE), (INPUT_NF, *INPUT_SHAPE)], batch_size=1))

    #! load parameters. with strict=False, missing keys won't throw errors
    # if args.pretrained_disp:
    #     print("=> using pre-trained weights for DispResNet")
    #     weights = torch.load(args.pretrained_disp)
    #     disp_net.load_state_dict(weights['state_dict'], strict=False)

    # if args.pretrained_pose:
    #     print("=> using pre-trained weights for PoseResNet")
    #     weights = torch.load(args.pretrained_pose)
    #     pose_net.load_state_dict(weights['state_dict'], strict=False)

    # disp_net = torch.nn.DataParallel(disp_net)
    df_net = torch.nn.DataParallel(df_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': df_net.parameters(), 'lr': args.lr},
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)  # TODO

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss',
                         'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(
        len(train_loader), args.epoch_size), valid_size=len(val_loader))  # TODO
    logger.epoch_bar.start()
    trainer = pretrain if args.pretrain else train

    for epoch in range(args.epochs):
        # logger.epoch_bar.update(epoch)

        # # train for one epoch
        # logger.reset_train_bar()
        train_loss = trainer(args, train_loader, df_net, optimizer,
                             args.epoch_size, logger, training_writer)
        # logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        # logger.reset_valid_bar()
        # errors, error_names = validate_without_gt(
        #     args, val_loader, df_net, epoch, logger, output_writers)
        # error_string = ', '.join('{} : {:.3f}'.format(name, error)
        #                          for name, error in zip(error_names, errors))
        # logger.valid_writer.write(' * Avg {}'.format(error_string))

        # for error, name in zip(errors, error_names):
        #     training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        # decisive_error = errors[1]
        # if best_error < 0:
        #     best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = train_loss < best_loss
        best_loss = min(best_loss, train_loss)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': df_net.module.state_dict()
            }, is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss])
    logger.epoch_bar.finish()


def train(args, train_loader, df_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w_g, w_l, w_s, w_e = args.g, args.l, args.s, args.e

    # switch to train mode
    df_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_imgs, ref_imgs, flow, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        # raw dmv dsv depth_gt fg_mask
        tgt_imgs = [tensor.to(device) for tensor in tgt_imgs]
        ref_imgs = [tensor.to(device) for tensor in ref_imgs]
        intrinsics_inv = intrinsics_inv.to(device)
        flow = flow.to(device)

        # compute output
        tgt_depth, ref_depth = compute_depth(df_net, tgt_imgs, ref_imgs)
        # dmv dsv depth_gt fg_mask fg_mask
        loss_g = compute_depth_reconstruction_loss()

        loss_l = compute_scale_consistent_loss()
        loss_e = compute_laplacian_regularization()
        loss_s = compute_scene_flow_loss()

        loss = w_s*loss_g  # pre-train
        # loss = w_l*loss_l + w_s*loss_g + w_e*loss_e  # self-sup

        if log_losses:
            train_writer.add_scalar('loss_g', loss_g.item(), n_iter)
            # train_writer.add_scalar('loss_l', loss_l.item(), n_iter)
            # train_writer.add_scalar('loss_e', loss_e.item(), n_iter)
            # train_writer.add_scalar('loss_s', loss_e.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_l.item(),
                             loss_l.item(), loss_e.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write(
                'Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def pretrain(args, train_loader, df_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    # w_l, w_s, w_e = args.l, args.s, args.e

    # switch to train mode
    df_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (dsv, dmv, depth_gt, rigidity) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        # raw dmv dsv depth_gt fg_mask
        dsv, dmv, depth_gt, rigidity = [
            i.to(device, dtype=torch.float) for i in (dsv, dmv, depth_gt, rigidity)]

        # compute output
        pred_depth = df_net(dsv, dmv)
        # dmv dsv depth_gt fg_mask fg_mask
        loss_g = compute_depth_reconstruction_loss(pred_depth, depth_gt)

        loss = loss_g  # pre-train
        # loss = w_l*loss_l + w_s*loss_g + w_e*loss_e  # self-sup

        if log_losses:
            train_writer.add_scalar('loss_g', loss_g.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write(
                'Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def compute_depth(df_net, tgt_img, ref_imgs):
    # raw dmv dsv depth_gt fg_mask
    # dsv, dmv,  depth, rigidity, extrinsic = tgt_img
    fused_depth_tgt = df_net(tgt_img[0], tgt_img[1])
    fused_depth_ref = df_net(ref_imgs[0], ref_imgs[1])
    return fused_depth_tgt, fused_depth_ref


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image(
                    'val Input', tensor2array(tgt_img[0]), 0)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(
                                            1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(
                                            tgt_depth[0][0], max_value=10),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']


if __name__ == '__main__':
    main()
