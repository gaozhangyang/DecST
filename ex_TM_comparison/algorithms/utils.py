# import numpy as np
import torch

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    a = torch.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    b = a.permute(0, 1, 2, 4, 3, 5, 6)

    patch_tensor = torch.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size, seq_length, patch_height, patch_width, channels  = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor