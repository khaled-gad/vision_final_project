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

This module implements the style transfer functionality using wavelet-based
feature transformation.
"""

import os
import tqdm
import argparse

import torch
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info


# List of supported image file extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    """
    Checks if a file is an image based on its extension.
    
    Args:
        filename (str): Name of the file to check.
        
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    """
    Wavelet-based Corrective Transform framework for style transfer.
    
    This class implements the wavelet-based approach to neural style transfer,
    allowing for multi-level feature transformation.
    
    Args:
        model_path (str): Path to pre-trained model checkpoints.
        transfer_at (list): Specifies which parts of the network to apply style transfer.
        option_unpool (str): Method for unpooling ('sum' or 'cat5').
        device (str): Device to run the model on ('cuda:0' or 'cpu').
        verbose (bool): Whether to print detailed information.
    """
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], 
                 option_unpool='cat5', device='cuda:0', verbose=False):

        # Validate transfer_at parameter
        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), f'invalid transfer_at: {transfer_at}'
        assert self.transfer_at, 'empty transfer_at'

        # Setup device and verbosity
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Initialize encoder and decoder
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        
        # Load pre-trained model weights
        encoder_path = os.path.join(model_path, f'wave_encoder_{option_unpool}_l4.pth')
        decoder_path = os.path.join(model_path, f'wave_decoder_{option_unpool}_l4.pth')
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def print_(self, msg):
        """
        Conditionally prints messages based on verbosity setting.
        
        Args:
            msg (str): Message to print.
        """
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        """
        Encodes input using the wavelet encoder at specified level.
        
        Args:
            x (torch.Tensor): Input tensor.
            skips (dict): Dictionary to store skip connections.
            level (int): Current encoding level.
            
        Returns:
            torch.Tensor: Encoded features.
        """
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        """
        Decodes features using the wavelet decoder at specified level.
        
        Args:
            x (torch.Tensor): Input features.
            skips (dict): Dictionary of skip connections.
            level (int): Current decoding level.
            
        Returns:
            torch.Tensor: Decoded features.
        """
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        """
        Extracts features from all levels of the network.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            tuple: (features, skip_connections) where features is a dictionary
                  of encoder and decoder features, and skip_connections contains
                  the wavelet coefficients.
        """
        skip_connections = {}
        extracted_features = {'encoder': {}, 'decoder': {}}
        
        # Forward pass through encoder
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skip_connections, level)
            if 'encoder' in self.transfer_at:
                extracted_features['encoder'][level] = x

        # Store the bottleneck features if not transferring at encoder
        if 'encoder' not in self.transfer_at:
            extracted_features['decoder'][4] = x
            
        # Forward pass through decoder
        for level in [4, 3, 2]:
            x = self.decode(x, skip_connections, level)
            if 'decoder' in self.transfer_at:
                extracted_features['decoder'][level - 1] = x
                
        return extracted_features, skip_connections

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        """
        Performs style transfer between content and style images.
        
        Args:
            content (torch.Tensor): Content image.
            style (torch.Tensor): Style image.
            content_segment (numpy.ndarray): Content segmentation map.
            style_segment (numpy.ndarray): Style segmentation map.
            alpha (float): Weight factor for blending transformed features.
            
        Returns:
            torch.Tensor: Stylized image.
        """
        # Compute label mappings for semantic-aware transfer
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        
        # Initialize content features and skip connections
        content_features = content
        content_skips = {}
        
        # Extract style features
        style_features, style_skips = self.get_all_feature(style)

        # Define levels for feature transformation
        transfer_enc_levels = [1, 2, 3, 4]
        transfer_dec_levels = [1, 2, 3, 4]
        transfer_skip_levels = ['pool1', 'pool2', 'pool3']

        # Forward pass through encoder with style transfer
        for level in [1, 2, 3, 4]:
            content_features = self.encode(content_features, content_skips, level)
            if 'encoder' in self.transfer_at and level in transfer_enc_levels:
                content_features = feature_wct(
                    content_features, style_features['encoder'][level],
                    content_segment, style_segment,
                    label_set, label_indicator,
                    alpha=alpha, device=self.device
                )
                self.print_(f'transfer at encoder {level}')
                
        # Apply style transfer to skip connections
        if 'skip' in self.transfer_at:
            for skip_level in transfer_skip_levels:
                for component_idx in [0, 1, 2]:  # Components: [LH, HL, HH]
                    content_skips[skip_level][component_idx] = feature_wct(
                        content_skips[skip_level][component_idx], 
                        style_skips[skip_level][component_idx],
                        content_segment, style_segment,
                        label_set, label_indicator,
                        alpha=alpha, device=self.device
                    )
                self.print_(f'transfer at skip {skip_level}')

        # Forward pass through decoder with style transfer
        for level in [4, 3, 2, 1]:
            if ('decoder' in self.transfer_at and 
                level in style_features['decoder'] and 
                level in transfer_dec_levels):
                content_features = feature_wct(
                    content_features, style_features['decoder'][level],
                    content_segment, style_segment,
                    label_set, label_indicator,
                    alpha=alpha, device=self.device
                )
                self.print_(f'transfer at decoder {level}')
            content_features = self.decode(content_features, content_skips, level)
            
        return content_features


def get_all_transfer():
    """
    Generates all valid combinations of transfer locations.
    
    Returns:
        list: List of sets, where each set contains a valid combination of
              transfer locations ('encoder', 'decoder', 'skip').
    """
    transfer_combinations = []
    for encoder_option in ['encoder', None]:
        for decoder_option in ['decoder', None]:
            for skip_option in ['skip', None]:
                valid_options = set([encoder_option, decoder_option, skip_option]) & set(['encoder', 'decoder', 'skip'])
                if valid_options:
                    transfer_combinations.append(valid_options)
    return transfer_combinations


def run_bulk(config):
    """
    Processes multiple images for style transfer based on configuration.
    
    Args:
        config (argparse.Namespace): Command-line arguments and configuration.
    """
    # Set up device (CPU or CUDA)
    device_name = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device_name)

    # Determine transfer locations based on config
    transfer_locations = set()
    if config.transfer_at_encoder:
        transfer_locations.add('encoder')
    if config.transfer_at_decoder:
        transfer_locations.add('decoder')
    if config.transfer_at_skip:
        transfer_locations.add('skip')

    # Find matching content and style files
    file_candidates = set(os.listdir(config.content)) & set(os.listdir(config.style))

    # Filter by segmentation maps if provided
    if config.content_segment and config.style_segment:
        file_candidates &= set(os.listdir(config.content_segment))
        file_candidates &= set(os.listdir(config.style_segment))

    # Process each file pair
    for filename in tqdm.tqdm(file_candidates):
        if not is_image_file(filename):
            print('invalid file (is not image), ', filename)
            continue
            
        # Construct file paths
        content_path = os.path.join(config.content, filename)
        style_path = os.path.join(config.style, filename)
        content_segment_path = os.path.join(config.content_segment, filename) if config.content_segment else None
        style_segment_path = os.path.join(config.style_segment, filename) if config.style_segment else None
        output_path = os.path.join(config.output, filename)

        # Load content and style images
        content_image = open_image(content_path, config.image_size).to(device)
        style_image = open_image(style_path, config.image_size).to(device)
        
        # Load segmentation maps if available
        content_seg_map = load_segment(content_segment_path, config.image_size)
        style_seg_map = load_segment(style_segment_path, config.image_size)     
        
        # Get file extension
        _, extension = os.path.splitext(filename)
        
        if not config.transfer_all:
            # Process with specified transfer locations
            with Timer('Elapsed time in whole WCT: {}', config.verbose):
                # Create output filename with transfer locations in the name
                location_suffix = '_'.join(sorted(list(transfer_locations)))
                output_filename = output_path.replace(
                    extension, 
                    f'_{config.option_unpool}_{location_suffix}.{extension}'
                )
                print('------ transfer:', output_path)
                
                # Initialize WCT2 model
                wct_model = WCT2(
                    transfer_at=transfer_locations, 
                    option_unpool=config.option_unpool, 
                    device=device, 
                    verbose=config.verbose
                )
                
                # Perform style transfer
                with torch.no_grad():
                    stylized_image = wct_model.transfer(
                        content_image, style_image, 
                        content_seg_map, style_seg_map, 
                        alpha=config.alpha
                    )
                
                # Save the result
                save_image(stylized_image.clamp_(0, 1), output_filename, padding=0)
        else:
            # Process with all possible transfer location combinations
            for transfer_combination in get_all_transfer():
                with Timer('Elapsed time in whole WCT: {}', config.verbose):
                    # Create output filename with transfer combination in the name
                    combination_suffix = '_'.join(sorted(list(transfer_combination)))
                    output_filename = output_path.replace(
                        extension,
                        f'_{config.option_unpool}_{combination_suffix}.{extension}'
                    )
                    print('------ transfer:', filename)
                    
                    # Initialize WCT2 model with current combination
                    wct_model = WCT2(
                        transfer_at=transfer_combination, 
                        option_unpool=config.option_unpool, 
                        device=device, 
                        verbose=config.verbose
                    )
                    
                    # Perform style transfer
                    with torch.no_grad():
                        stylized_image = wct_model.transfer(
                            content_image, style_image, 
                            content_seg_map, style_seg_map, 
                            alpha=config.alpha
                        )
                    
                    # Save the result
                    save_image(stylized_image.clamp_(0, 1), output_filename, padding=0)


if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Wavelet Corrective Transfer for Neural Style Transfer')
    parser.add_argument('--content', type=str, default='./examples/content',
                        help='Directory containing content images')
    parser.add_argument('--content_segment', type=str, default=None,
                        help='Directory containing content segmentation maps')
    parser.add_argument('--style', type=str, default='./examples/style',
                        help='Directory containing style images')
    parser.add_argument('--style_segment', type=str, default=None,
                        help='Directory containing style segmentation maps')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Directory to save stylized outputs')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size of the images (default: 512)')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Blending factor for style transfer (default: 1.0)')
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'],
                        help='Method for unpooling (default: cat5)')
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true',
                        help='Apply style transfer at encoder')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true',
                        help='Apply style transfer at decoder')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true',
                        help='Apply style transfer at skip connections')
    parser.add_argument('-a', '--transfer_all', action='store_true',
                        help='Try all combinations of transfer locations')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU even if CUDA is available')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    # Parse arguments
    config = parser.parse_args()
    print(config)

    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    # Example command to run the script:
    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    
    # Run the style transfer process
    run_bulk(config)
