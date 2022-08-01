import numpy as np
from torch.utils.data import Dataset
from src.utils.file_load import *
import nibabel as nib
import pickle
from skimage.transform import resize
import SimpleITK as sitk


class data_set(Dataset):
    def __init__(self, is_train=True):
        self.task_json, self.path = load_json('AMOS22', 'task1_dataset.json')
        if is_train:
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'trainx.li_x.li'), 'rb+') as f:
                self.images_path = pickle.load(f)
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'trainx.li_y.li'), 'rb+') as f:
                # pickle.load(f)
                self.labels_path = pickle.load(f)
        else:
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_x.li'), 'rb+') as f:
                self.images_path = pickle.load(f)
            with open(os.path.join('..', 'checkpoints', 'tr_ts_inf', 'testx.li_y.li'), 'rb+') as f:
                self.labels_path = pickle.load(f)
        pass

    def __len__(self):
        return len(self.images_path)
        pass

    def __getitem__(self, index):
        x = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.path, str(self.images_path[index], 'utf-8')))).astype(np.int16)
        y = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.path, str(self.labels_path[index], 'utf-8')))).astype(np.int8)
        x = np.array(x)
        y = np.array(y)
        x = self.norm(x)
        x = np.expand_dims(x, 0)
        # y = np.expand_dims(y, 0)
        x = resize(x, (1, 64, 256, 256), preserve_range=True)
        y = np.round(resize(y, (64, 256, 256), preserve_range=True))
        return x, y

    def norm(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

