"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0

Core utilities for Whitening and Coloring Transform operations.
"""
import torch
import numpy as np
from PIL import Image


def svd(feat, iden=False, device='cpu'):
    """
    Performs Singular Value Decomposition on a feature tensor.
    
    Args:
        feat (torch.Tensor): Input feature tensor.
        iden (bool): Whether to add identity matrix to covariance.
        device (str): Device to perform computation on.
        
    Returns:
        tuple: (u, e, v) matrices from SVD decomposition.
    """
    tensor_dims = feat.size()
    feature_mean = torch.mean(feat, 1)
    feature_mean = feature_mean.unsqueeze(1).expand_as(feat)
    adjusted_feat = feat.clone()
    adjusted_feat -= feature_mean
    
    # Calculate covariance matrix
    if tensor_dims[1] > 1:
        covariance = torch.mm(adjusted_feat, adjusted_feat.t()).div(tensor_dims[1] - 1)
    else:
        covariance = torch.mm(adjusted_feat, adjusted_feat.t())
        
    if iden:
        covariance += torch.eye(tensor_dims[0]).to(device)
        
    u_matrix, eigenvalues, v_matrix = torch.svd(covariance, some=False)
    return u_matrix, eigenvalues, v_matrix


def get_squeeze_feat(feat):
    """
    Squeezes and reshapes a feature tensor for processing.
    
    Args:
        feat (torch.Tensor): Input feature tensor.
        
    Returns:
        torch.Tensor: Reshaped feature tensor.
    """
    squeezed_feat = feat.squeeze(0)
    channels = squeezed_feat.size(0)
    return squeezed_feat.view(channels, -1).clone()


def get_rank(singular_values, dim, eps=0.00001):
    """
    Determines the effective rank of a matrix based on singular values.
    
    Args:
        singular_values (torch.Tensor): Singular values from SVD.
        dim (int): Maximum possible rank.
        eps (float): Threshold for significant singular values.
        
    Returns:
        int: Effective rank of the matrix.
    """
    effective_rank = dim
    for idx in range(dim - 1, -1, -1):
        if singular_values[idx] >= eps:
            effective_rank = idx + 1
            break
    return effective_rank


def wct_core(cont_feat, styl_feat, weight=1, registers=None, device='cpu'):
    """
    Core Whitening and Coloring Transform implementation.
    
    Args:
        cont_feat (torch.Tensor): Content feature.
        styl_feat (torch.Tensor): Style feature.
        weight (float): Weight factor for style transfer.
        registers (dict, optional): Pre-computed values for optimization.
        device (str): Device to perform computation on.
        
    Returns:
        torch.Tensor: Transformed feature tensor.
    """
    content_feature = get_squeeze_feat(cont_feat)
    content_min_val = content_feature.min()
    content_max_val = content_feature.max()
    
    # Center content feature
    content_mean = torch.mean(content_feature, 1).unsqueeze(1).expand_as(content_feature)
    content_feature -= content_mean

    if not registers:
        # Perform SVD on content feature
        _, c_eigenvalues, c_v_matrix = svd(content_feature, iden=True, device=device)

        # Process style feature
        style_feature = get_squeeze_feat(styl_feat)
        style_mean = torch.mean(style_feature, 1)
        _, s_eigenvalues, s_v_matrix = svd(style_feature, iden=True, device=device)
        
        # Get effective rank and create transform matrix
        k_style = get_rank(s_eigenvalues, style_feature.size()[0])
        style_d = (s_eigenvalues[0:k_style]).pow(0.5)
        style_transform = torch.mm(
            torch.mm(s_v_matrix[:, 0:k_style], torch.diag(style_d) * weight), 
            (s_v_matrix[:, 0:k_style].t())
        )

        if registers is not None:
            registers['EDE'] = style_transform
            registers['s_mean'] = style_mean
            registers['c_v'] = c_v_matrix
            registers['c_e'] = c_eigenvalues
    else:
        style_transform = registers['EDE']
        style_mean = registers['s_mean']
        _, c_eigenvalues, c_v_matrix = svd(content_feature, iden=True, device=device)

    # Whitening step
    k_content = get_rank(c_eigenvalues, content_feature.size()[0])
    content_d = (c_eigenvalues[0:k_content]).pow(-0.5)
    
    whitening_step1 = torch.mm(c_v_matrix[:, 0:k_content], torch.diag(content_d))
    whitening_step2 = torch.mm(whitening_step1, (c_v_matrix[:, 0:k_content].t()))
    whitened_content = torch.mm(whitening_step2, content_feature)

    # Apply style transformation
    transformed_feature = torch.mm(style_transform, whitened_content)
    transformed_feature = transformed_feature + style_mean.unsqueeze(1).expand_as(transformed_feature)
    transformed_feature.clamp_(content_min_val, content_max_val)

    return transformed_feature


def wct_core_segment(content_feat, style_feat, content_segment, style_segment,
                     label_set, label_indicator, weight=1, registers=None,
                     device='cpu'):
    """
    Segment-aware Whitening and Coloring Transform.
    
    Args:
        content_feat (torch.Tensor): Content feature.
        style_feat (torch.Tensor): Style feature.
        content_segment (numpy.ndarray): Content segmentation map.
        style_segment (numpy.ndarray): Style segmentation map.
        label_set (numpy.ndarray): Set of valid labels.
        label_indicator (numpy.ndarray): Indicator for valid transfer labels.
        weight (float): Weight factor for style transfer.
        registers (dict, optional): Pre-computed values for optimization.
        device (str): Device to perform computation on.
        
    Returns:
        torch.Tensor: Transformed feature tensor with segmentation awareness.
    """
    def resize(feat, target):
        """Resizes segmentation map to match feature dimensions."""
        new_size = (target.size(2), target.size(1))
        if len(feat.shape) == 2:
            return np.asarray(Image.fromarray(feat).resize(new_size, Image.NEAREST))
        else:
            return np.asarray(Image.fromarray(feat, mode='RGB').resize(new_size, Image.NEAREST))

    def get_index(feat, label):
        """Gets indices of pixels with specific label."""
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0]).to(device)

    content_tensor = content_feat.squeeze(0)
    style_tensor = style_feat.squeeze(0)

    content_flat = content_tensor.view(content_tensor.size(0), -1).clone()
    style_flat = style_tensor.view(style_tensor.size(0), -1).clone()

    # Resize segmentation maps to match feature dimensions
    resized_content_seg = resize(content_segment, content_tensor)
    resized_style_seg = resize(style_segment, style_tensor)

    result_feature = content_flat.clone()
    
    # Process each semantic region separately
    for label in label_set:
        if not label_indicator[label]:
            continue
            
        content_indices = get_index(resized_content_seg, label)
        style_indices = get_index(resized_style_seg, label)
        
        if content_indices is None or style_indices is None:
            continue
            
        # Extract features for this semantic region
        region_content_feat = torch.index_select(content_flat, 1, content_indices)
        region_style_feat = torch.index_select(style_flat, 1, style_indices)
        
        # Apply WCT to this region
        transformed_region = wct_core(region_content_feat, region_style_feat, weight, registers, device=device)
        
        # Update the result with transformed region
        if torch.__version__ >= '0.4.0':
            # Handle newer PyTorch versions
            transposed_result = torch.transpose(result_feature, 1, 0)
            transposed_result.index_copy_(0, content_indices,
                                          torch.transpose(transformed_region, 1, 0))
            result_feature = torch.transpose(transposed_result, 1, 0)
        else:
            # Handle older PyTorch versions
            result_feature.index_copy_(1, content_indices, transformed_region)
            
    return result_feature


def feature_wct(content_feat, style_feat, content_segment=None, style_segment=None,
                label_set=None, label_indicator=None, weight=1, registers=None, alpha=1, device='cpu'):
    """
    Applies Whitening and Coloring Transform to feature maps with optional segmentation.
    
    Args:
        content_feat (torch.Tensor): Content feature.
        style_feat (torch.Tensor): Style feature.
        content_segment (numpy.ndarray, optional): Content segmentation map.
        style_segment (numpy.ndarray, optional): Style segmentation map.
        label_set (numpy.ndarray, optional): Set of valid labels.
        label_indicator (numpy.ndarray, optional): Indicator for valid transfer labels.
        weight (float): Weight factor for style transfer.
        registers (dict, optional): Pre-computed values for optimization.
        alpha (float): Blending factor between content and transformed features.
        device (str): Device to perform computation on.
        
    Returns:
        torch.Tensor: Feature tensor after WCT transformation.
    """
    # Choose between segment-aware or regular WCT
    if label_set is not None:
        transformed_feat = wct_core_segment(
            content_feat, style_feat, content_segment, style_segment,
            label_set, label_indicator, weight, registers, device=device
        )
    else:
        transformed_feat = wct_core(content_feat, style_feat, device=device)
    
    # Reshape and blend with original content
    transformed_feat = transformed_feat.view_as(content_feat)
    blended_feat = alpha * transformed_feat + (1 - alpha) * content_feat
    
    return blended_feat
