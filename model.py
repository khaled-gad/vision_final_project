"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

This module implements the wavelet-based neural network models for style transfer.
"""

import torch
import torch.nn as nn
import numpy as np


def get_wav(in_channels, pool=True):
    """
    Constructs wavelet decomposition filters using convolution operations.
    
    Args:
        in_channels (int): Number of input channels for the wavelet transform.
        pool (bool): If True, creates downsampling filters, otherwise creates upsampling filters.
        
    Returns:
        tuple: Four filters for wavelet decomposition (LL, LH, HL, HH).
    """
    # Create Haar wavelet filter coefficients
    haar_low_pass = 1 / np.sqrt(2) * np.ones((1, 2))
    haar_high_pass = 1 / np.sqrt(2) * np.ones((1, 2))
    haar_high_pass[0, 0] = -1 * haar_high_pass[0, 0]

    # Construct 2D wavelet filters
    haar_filter_LL = np.transpose(haar_low_pass) * haar_low_pass
    haar_filter_LH = np.transpose(haar_low_pass) * haar_high_pass
    haar_filter_HL = np.transpose(haar_high_pass) * haar_low_pass
    haar_filter_HH = np.transpose(haar_high_pass) * haar_high_pass

    # Convert to PyTorch tensors
    tensor_LL = torch.from_numpy(haar_filter_LL).unsqueeze(0)
    tensor_LH = torch.from_numpy(haar_filter_LH).unsqueeze(0)
    tensor_HL = torch.from_numpy(haar_filter_HL).unsqueeze(0)
    tensor_HH = torch.from_numpy(haar_filter_HH).unsqueeze(0)

    # Select the appropriate network layer type
    layer_type = nn.Conv2d if pool else nn.ConvTranspose2d

    # Create the four wavelet filters as network layers
    low_low_filter = layer_type(in_channels, in_channels,
                       kernel_size=2, stride=2, padding=0, bias=False,
                       groups=in_channels)
    low_high_filter = layer_type(in_channels, in_channels,
                        kernel_size=2, stride=2, padding=0, bias=False,
                        groups=in_channels)
    high_low_filter = layer_type(in_channels, in_channels,
                        kernel_size=2, stride=2, padding=0, bias=False,
                        groups=in_channels)
    high_high_filter = layer_type(in_channels, in_channels,
                         kernel_size=2, stride=2, padding=0, bias=False,
                         groups=in_channels)

    # Set weights to be non-trainable
    low_low_filter.weight.requires_grad = False
    low_high_filter.weight.requires_grad = False
    high_low_filter.weight.requires_grad = False
    high_high_filter.weight.requires_grad = False

    # Initialize the filter weights
    low_low_filter.weight.data = tensor_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    low_high_filter.weight.data = tensor_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    high_low_filter.weight.data = tensor_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    high_high_filter.weight.data = tensor_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return low_low_filter, low_high_filter, high_low_filter, high_high_filter


class WavePool(nn.Module):
    """
    Module for wavelet pooling operation.
    
    Decomposes an input tensor into four frequency sub-bands using 
    Haar wavelet transform.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        """
        Forward pass for wavelet pooling.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            tuple: Four tensors representing different frequency components.
        """
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    """
    Module for wavelet unpooling operation.
    
    Reconstructs an image from its wavelet coefficients using
    inverse Haar wavelet transform.
    
    Args:
        in_channels (int): Number of input channels.
        option_unpool (str): Method for unpooling, either 'sum' or 'cat5'.
    """
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        """
        Forward pass for wavelet unpooling.
        
        Args:
            LL (torch.Tensor): Low-low frequency component.
            LH (torch.Tensor): Low-high frequency component.
            HL (torch.Tensor): High-low frequency component.
            HH (torch.Tensor): High-high frequency component.
            original (torch.Tensor, optional): Original input for concatenation in 'cat5' mode.
            
        Returns:
            torch.Tensor: Reconstructed tensor from wavelet coefficients.
            
        Raises:
            NotImplementedError: If option_unpool is not 'sum' or 'cat5'.
        """
        if self.option_unpool == 'sum':
            # Combine components through addition
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            # Combine components through concatenation with original
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError("Unsupported unpooling option")


class WaveEncoder(nn.Module):
    """
    Wavelet-based encoder network.
    
    Uses wavelet decomposition for multi-level feature extraction.
    
    Args:
        option_unpool (str): Unpooling method to use in the corresponding decoder.
    """
    def __init__(self, option_unpool):
        super(WaveEncoder, self).__init__()
        self.option_unpool = option_unpool

        # Basic operations
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        # Level 1 layers
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        # Level 2 layers
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        # Level 3 layers
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)

        # Level 4 layers
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        """
        Forward pass for the wavelet encoder.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Encoded features at the deepest level.
        """
        skip_connections = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skip_connections, level)
        return x

    def encode(self, x, skips, level):
        """
        Encode at a specific level of the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            skips (dict): Dictionary to store skip connections.
            level (int): Current level in the network (1-4).
            
        Returns:
            torch.Tensor: Encoded features at the current level.
            
        Raises:
            AssertionError: If level is not in {1, 2, 3, 4}.
        """
        assert level in {1, 2, 3, 4}, "Level must be between 1 and 4"
        
        if self.option_unpool == 'sum':
            # Encoding path for sum-based unpooling
            if level == 1:
                # Process level 1
                features = self.conv0(x)
                features = self.relu(self.conv1_1(self.pad(features)))
                features = self.relu(self.conv1_2(self.pad(features)))
                skips['conv1_2'] = features
                LL, LH, HL, HH = self.pool1(features)
                skips['pool1'] = [LH, HL, HH]  # Save high frequencies for skip connections
                return LL  # Return low frequency for next level
            elif level == 2:
                # Process level 2
                features = self.relu(self.conv2_1(self.pad(x)))
                features = self.relu(self.conv2_2(self.pad(features)))
                skips['conv2_2'] = features
                LL, LH, HL, HH = self.pool2(features)
                skips['pool2'] = [LH, HL, HH]
                return LL
            elif level == 3:
                # Process level 3
                features = self.relu(self.conv3_1(self.pad(x)))
                features = self.relu(self.conv3_2(self.pad(features)))
                features = self.relu(self.conv3_3(self.pad(features)))
                features = self.relu(self.conv3_4(self.pad(features)))
                skips['conv3_4'] = features
                LL, LH, HL, HH = self.pool3(features)
                skips['pool3'] = [LH, HL, HH]
                return LL
            else:  # level == 4
                # Process deepest level
                return self.relu(self.conv4_1(self.pad(x)))

        elif self.option_unpool == 'cat5':
            # Encoding path for concatenation-based unpooling
            if level == 1:
                features = self.conv0(x)
                features = self.relu(self.conv1_1(self.pad(features)))
                return features
            elif level == 2:
                features = self.relu(self.conv1_2(self.pad(x)))
                skips['conv1_2'] = features
                LL, LH, HL, HH = self.pool1(features)
                skips['pool1'] = [LH, HL, HH]
                features = self.relu(self.conv2_1(self.pad(LL)))
                return features
            elif level == 3:
                features = self.relu(self.conv2_2(self.pad(x)))
                skips['conv2_2'] = features
                LL, LH, HL, HH = self.pool2(features)
                skips['pool2'] = [LH, HL, HH]
                features = self.relu(self.conv3_1(self.pad(LL)))
                return features
            else:  # level == 4
                features = self.relu(self.conv3_2(self.pad(x)))
                features = self.relu(self.conv3_3(self.pad(features)))
                features = self.relu(self.conv3_4(self.pad(features)))
                skips['conv3_4'] = features
                LL, LH, HL, HH = self.pool3(features)
                skips['pool3'] = [LH, HL, HH]
                features = self.relu(self.conv4_1(self.pad(LL)))
                return features
        else:
            raise NotImplementedError("Unsupported unpooling option")


class WaveDecoder(nn.Module):
    """
    Wavelet-based decoder network.
    
    Reconstructs an image from encoded features using wavelet unpooling.
    
    Args:
        option_unpool (str): Method for unpooling, either 'sum' or 'cat5'.
    """
    def __init__(self, option_unpool):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool

        # Determine the input channel multiplier based on unpooling method
        input_multiplier = 1 if option_unpool == 'sum' else 5

        # Basic operations
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        
        # Deepest level layers
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        # Level 3 layers
        self.recon_block3 = WaveUnpool(256, option_unpool)
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(256*input_multiplier, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256*input_multiplier, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        # Level 2 layers
        self.recon_block2 = WaveUnpool(128, option_unpool)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128*input_multiplier, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*input_multiplier, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        # Level 1 layers
        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64*input_multiplier, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*input_multiplier, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):
        """
        Forward pass for the wavelet decoder.
        
        Args:
            x (torch.Tensor): Input features from the encoder.
            skips (dict): Skip connections from the encoder.
            
        Returns:
            torch.Tensor: Reconstructed image.
        """
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        """
        Decode at a specific level of the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            skips (dict): Dictionary of skip connections.
            level (int): Current level in the network (4-1).
            
        Returns:
            torch.Tensor: Decoded features at the current level.
            
        Raises:
            AssertionError: If level is not in {4, 3, 2, 1}.
        """
        assert level in {4, 3, 2, 1}, "Level must be between 4 and 1"
        
        if level == 4:
            # Process level 4 (deepest)
            features = self.relu(self.conv4_1(self.pad(x)))
            # Retrieve high frequency components from skip connections
            LH, HL, HH = skips['pool3']
            # Get original features for concatenation if available
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            # Apply wavelet unpooling
            features = self.recon_block3(features, LH, HL, HH, original)
            # Select appropriate convolution based on unpooling method
            conv3_4_layer = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
            features = self.relu(conv3_4_layer(self.pad(features)))
            features = self.relu(self.conv3_3(self.pad(features)))
            return self.relu(self.conv3_2(self.pad(features)))
        
        elif level == 3:
            # Process level 3
            features = self.relu(self.conv3_1(self.pad(x)))
            LH, HL, HH = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            features = self.recon_block2(features, LH, HL, HH, original)
            conv2_2_layer = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(conv2_2_layer(self.pad(features)))
        
        elif level == 2:
            # Process level 2
            features = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            features = self.recon_block1(features, LH, HL, HH, original)
            conv1_2_layer = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(conv1_2_layer(self.pad(features)))
        
        else:  # level == 1
            # Process level 1 (output)
            return self.conv1_1(self.pad(x))
