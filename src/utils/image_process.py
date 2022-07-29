import torch

import PIL.Image as Image
import numpy as np
from src.model.model import *
import os
import nibabel as nib
from einops import rearrange

def a(images, outputs):
    images_ori = images.data.squeeze().cpu().numpy()
    images_ori = np.expand_dims(images_ori, axis=-1)
    images_ori = array_to_img(images_ori)
    images_ori = images_ori.convert("RGB")
    image_mask = outputs[0].data.squeeze().cpu().numpy()
    image_mask = Image.fromarray(image_mask.astype('uint8'), 'P')
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176]
    image_mask.putpalette(palettedata)
    image_mask = image_mask.convert('RGB')
    img = Image.blend(images_ori, image_mask, 0.7)  # blend_img = img1 * (1 – 0.3) + img2* alpha
    img.save('..' + '/result_overlap/pt_{}_compare_{}.png'.format(1, 2))


def array_to_img(x, scale=True):
    # target PIL image has format (height, width, channel) (512,512,1)
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image).'
                         'Got array with shape', x.shape)
    if scale:
        x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number:', x.shape[2])


# model = Model(1, 16).cpu()
# model.load_state_dict(torch.load(os.path.join('..', 'checkpoints', 'auto_save', 'model_onehot.pth')))
# x_image = nib.load(os.path.join('..', '..', 'data', 'AMOS22', './imagesTr/amos_0600.nii.gz')).dataobj
# # w,h,d
# x_image = torch.unsqueeze(torch.unsqueeze(torch.Tensor(np.array(x_image)).float(), 0), 0).cpu()
# y = model(x_image)
# out = torch.argmax(y, 1)
# out = torch.squeeze(torch.squeeze(out, 0), 0)
y = torch.rand([1, 1, 15, 150, 150])
z = torch.rand([1, 1, 15, 150, 150])
def bind(a, b):
    '''
        a shape -> b 1 d w h
        b shape -> b 1 d w h
        ori_image -> ground truth
        mask_image -> pred
    '''
    a = a.data.squeeze().cpu().numpy()
    b = b.data.squeeze().cpu().numpy()
    ori_image = np.expand_dims(a[int(a.shape[0]/2), :, :], -1)
    mask_image = np.expand_dims(b[int(b.shape[0]/2), :, :], -1)
    # ori_image = rearrange(ori_image, 'w h -> w h c')
    # mask_image = rearrange(mask_image, 'w h -> w h c')
    ori_image = array_to_img(ori_image)
    mask_image = array_to_img(mask_image)
    palettedata = [0, 0, 0, 102, 0, 255, 0, 255, 176]
    ori_image.putpalette(palettedata)
    ori_image = ori_image.convert('RGB')
    mask_image.putpalette(palettedata)
    mask_image = mask_image.convert('RGB')
    img = Image.blend(ori_image, mask_image, 0.7)  # blend_img = img1 * (1 – 0.3) + img2* alpha
    img.save('..' + '/result_overlap/pt_{}_compare_{}.png'.format(1, 2))


bind(y,z)