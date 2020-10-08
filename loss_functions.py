from functools import reduce
import torch
from torch import nn
# import torch.nn.functional as F
import torch.nn.functional as tf
from inverse_warp import inverse_warp2, inverse_warp
import math

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


# TODO
def depth_relative_gradient(pred_map, mask, scale=[pow(x, 2) for x in (0, 1, 2, 3, 4)]):
    if type(scale) not in (tuple, list):
        scale = [scale]
    depth_gradient = []
    for n_scale in scale:
        dx, dy = gradient(pred_map, n_scale, normalization=True)
        depth_gradient.append((dx, dy))
    return depth_gradient


# Lg reconstruction loss, for DMV
def compute_depth_reconstruction_loss(pred_map, dmv, mask):
    static_mask = 1-mask
    pred_map *= static_mask
    dmv *= static_mask
    return nn.functional.l1_loss(pred_map, dmv)


def gradient(pred, step=1, normalization=False, keepdim=True):  # !
    # compute gradients of predicion map
    assert type(pred) is torch.Tensor and pred.ndim() == 4
    assert type(step) is int and step >= 1 and step <= pred.shape[-1]
    # [N,C,Y,X] gradient in increasing direction
    # pad starting from last dimension and moving forward
    x = tf.pad(pred, (0, step), 'replicate', 1)
    y = tf.pad(pred, (0, 0, 0, step), 'replicate', 1)  # pad bottom

    x1, x2, y1, y2 = x[:, :, :, :-step], x[:, :,
                                           :, step:], y[:, :, :-step], y[:, :, step:]

    if(normalization):
        D_dx = (x1-x2)/(x1.abs()+x2.abs())
        D_dy = (y1-y2)/(y1.abs()+y2.abs())
    else:
        D_dx = (x1-x2)
        D_dy = (y1-y2)
    return D_dx, D_dy


def compute_scale_consistent_loss(Drt, dsv, mask):
    # Ll, scale invariant depth gradient
    Drt *= mask
    dsv *= mask
    Drt_depth_gradients = depth_relative_gradient(Drt)
    dsv_depth_gradients = depth_relative_gradient(dsv)
    loss = 0.
    for i in range(len(Drt)):
        loss += nn.functional.l1_loss(Drt_depth_gradients[i],
                                      dsv_depth_gradients[i])
    return loss / len(Drt)  # FIXME


def compute_target_position(ref_x, Fx):
    """compute target position of warped ref_x into neighboring_x

    Args:
        ref_x (Tensor): [description]
        Fx (Tensor): [optical flow of ref_x -> neighboring_x]
    """
    pass


def compute_scene_flow_loss(tgt_img, ref_imgs, tgt_intrinsicts, ref_intrinsics,
                            tgt_img_depth, ref_imgs_depth, tgt_pose, ref_poses, rotation_mode='euler', padding_mode='zeros'):  # Ls, 3D scene motion
    # TODO
    # l1 loss of unprojected ref_x and neighbour_x
    # neighbour_x is computed by ref_x + F(ref_x)
    assert pose.size(1) == len(ref_imgs)
    scene_flow_loss = 0.
    b, _, h, w = tgt_img_depth.shape
    # tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
    # ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
    # intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

    tgt_img_warped, tgt_valid_points = inverse_warp(tgt_img, tgt_img_depth, tgt_pose,
                                                    tgt_intrinsicts, rotation_mode, padding_mode)

    warped_imgs = []
    diff_maps = []

    for i, ref_img in enumerate(ref_imgs):
        current_pose = ref_poses[:, i]

        ref_img_warped, valid_points = inverse_warp(ref_img, ref_imgs_depth, current_pose,
                                                    ref_imgs_depth, rotation_mode, padding_mode)
        diff = (tgt_img_warped - ref_img_warped) * \
            \
            \
            valid_points.unsqueeze(1).float()

        scene_flow_loss += diff.abs().mean()    # l1

        warped_imgs.append(ref_img_warped[0])
        diff_maps.append(diff[0])

    return scene_flow_loss

def apply_scene_flow(img, sceneflow):
    batch_size, _, height, width = img.size()

# Le loss
def compute_laplacian_regularization(pred_map, mask=None, smoothness_weight=1, ):
    # laplacian operator calculated unmixed second order derivatives
    assert mask is None or mask.shape == pred_map.shape[-2:]
    pred_map *= mask
    dx, dy = gradient(pred_map)
    dx2, _ = gradient(dx)
    _, dy2 = gradient(dy)
    # laplacian smoothness
    return torch.sum(map(lambda x: x.square(), [dx2, dy2])) * smoothness_weight


def smooth_loss(pred_map):  # from SfmLearner-Pytorch
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() +
                 dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

# DeepBlender reconstruction, L1 loss


def compute_reconstruciton_loss(pred_residual, gt):
    return nn.functional.l1_loss(pred_residual, gt, reduce='mean')

# GAN Loss


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


# photometric loss
# geometry consistency loss
def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode):

    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), max_scales)
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        for s in range(num_scales):

            # # downsample img
            # b, _, h, w = tgt_depth[s].size()
            # downscale = tgt_img.size(2)/h
            # if s == 0:
            #     tgt_img_scaled = tgt_img
            #     ref_img_scaled = ref_img
            # else:
            #     tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            #     ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            # intrinsic_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            # tgt_depth_scaled = tgt_depth[s]
            # ref_depth_scaled = ref_depth[s]

            # upsample depth
            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics
            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
            else:
                tgt_depth_scaled = F.interpolate(
                    tgt_depth[s], (h, w), mode='nearest')
                ref_depth_scaled = F.interpolate(
                    ref_depth[s], (h, w), mode='nearest')

            photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
            photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)

            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode):

    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(
        ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() /
                  (computed_depth + projected_depth)).clamp(0, 1)

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (
            tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask

    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth[0], tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth[0], ref_img)

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * \
            torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
