# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import point_sample
from torch import Tensor
import numpy as np

# from mmengine.runner import loops
# IterBasedTrainLoop


def my_get_uncertainty( mask_preds: Tensor, point_gt: list) -> Tensor:
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_preds' for the foreground class in `classes`.

    Args:
        mask_preds (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (Tensor): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    # if mask_preds.shape[1] == 1:
    #     p = mask_preds.clone()
    #     q = point_gt.clone()
    #     # global my_self_iter
    #     from mmengine.runner.loops import my_self_iter
    #     # print(my_self_iter)
    #     np.seterr(over='ignore')
    #     g = q/(1+np.exp(10000-my_self_iter))
    #     # -torch.abs(p-g)
    # else:
    #     # TODO
    #     exit(0)
    # return -torch.abs(p-g)
    return  -torch.abs(mask_preds.clone())


def my_get_uncertain_point_coords_with_randomness(
        mask_preds: Tensor, 
        mask_targets: Tensor, 
        mask_per_pic: list, num_points: int,
        oversample_ratio: float, importance_sample_ratio: float) -> Tensor:
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (Tensor): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_preds.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=mask_preds.device)
    point_logits = point_sample(mask_preds, point_coords)
    point_gt = point_sample(mask_targets, point_coords)
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = my_get_uncertainty(point_logits, point_gt)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_preds.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(
            batch_size, num_random_points, 2, device=mask_preds.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords
