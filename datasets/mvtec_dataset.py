import torch
import glob
import numpy as np
import cv2
from collections import OrderedDict
import os
import math
import glob
import yaml
import scipy.io as sio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import joblib
from PIL import Image
from datasets.utils.pseudo_utils import augment_image

OBJECT_TYPE = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    'all'
]

DEFECT_TYPE = {
    'leather':['good', 'glue', 'cut', 'fold', 'color', 'poke'],
    'tile':['good', 'rough', 'crack', 'oil', 'glue_strip', 'gray_stroke'],
    'toothbrush':['good', 'defective'],
    'screw':['good', 'thread_top', 'manipulated_front', 'thread_side', 'scratch_head', 'scratch_neck'],
    'bottle':['good', 'broken_small', 'contamination', 'broken_large'],
    'zipper':['good', 'combined', 'rough', 'fabric_interior', 'split_teeth', 'squeezed_teeth', 'broken_teeth', 'fabric_border'],
    'grid':['good', 'thread', 'glue', 'bent', 'broken', 'metal_contamination'],
    'pill':['good', 'combined', 'faulty_imprint', 'crack', 'scratch', 'contamination', 'pill_type', 'color'],
    'wood':['good', 'liquid', 'combined', 'hole', 'scratch', 'color'],
    'metal_nut':['good', 'bent', 'scratch', 'flip', 'color'],
    'transistor':['good', 'damaged_case', 'misplaced', 'cut_lead', 'bent_lead'],
    'carpet':['good', 'thread', 'hole', 'metal_contamination', 'cut', 'color'],
    'hazelnut':['good', 'crack', 'hole', 'cut', 'print'],
    'cable':['good', 'missing_wire', 'cut_outer_insulation', 'missing_cable', 'combined', 'poke_insulation', 'cable_swap', 'bent_wire', 'cut_inner_insulation'],
    'capsule':['good', 'faulty_imprint', 'crack', 'squeeze', 'scratch', 'poke'],
    'all':['good', 'glue', 'cut', 'fold', 'color', 'poke',
           'rough', 'crack', 'oil', 'glue_strip', 'gray_stroke',
           'defective',
           'thread_top', 'manipulated_front', 'thread_side', 'scratch_head', 'scratch_neck',
           'broken_small', 'contamination', 'broken_large',
           'combined', 'rough', 'fabric_interior', 'split_teeth', 'squeezed_teeth', 'broken_teeth', 'fabric_border',
           'thread', 'glue', 'bent', 'broken', 'metal_contamination',
           'combined', 'faulty_imprint', 'crack', 'scratch', 'contamination', 'pill_type', 'color',
           'liquid', 'combined', 'hole', 'scratch', 'color',
           'bent', 'scratch', 'flip', 'color',
           'damaged_case', 'misplaced', 'cut_lead', 'bent_lead',
           'thread', 'hole', 'metal_contamination', 'cut', 'color',
           'crack', 'hole', 'cut', 'print',
           'missing_wire', 'cut_outer_insulation', 'missing_cable', 'combined', 'poke_insulation', 'cable_swap', 'bent_wire', 'cut_inner_insulation',
           'faulty_imprint', 'crack', 'squeeze', 'scratch', 'poke'
           ]
}

def get_inputs(file_addr, read_flag=None):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        if read_flag is not None:
            img = cv2.imread(file_addr, read_flag)
        else:
            img = cv2.imread(file_addr)

        # 如果图像是彩色图像，则将其从 BGR 转换为 RGB
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()

def _convert_to_rgb(image):
    return image.convert('RGB')

class mvtec_dataset(Dataset):
    def __init__(self, config, data_dir, spatial_size=240, mode="train", shot='few', preprocess=None):
        super(mvtec_dataset, self).__init__()
        self.shot = shot
        self.data_dir = data_dir
        self.spatial_size = spatial_size
        self.mask_transform = transforms.ToTensor()
        self.pre_transform = transforms.Compose([
                transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(240, 240)),
                _convert_to_rgb,
                transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
            ])
        self.post_transform = transforms.Compose([
                transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        self.obj_type = config['obj_type']
        self.use_defect_type = config['use_defect_type']
        self.mode = mode
        self.preprocess = preprocess
        self.dataset_init()
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(config['anomaly_source_path'], '**', '**', '*.jpg')))
        self.resize_shape = (spatial_size, spatial_size)
        

    def dataset_init(self):
        if self.shot == 'zero':
            self.img_paths = []
            return
        if self.obj_type == 'all':
            if self.mode == 'train':
                self.img_paths = glob.glob(os.path.join(self.data_dir, '**', 'train', 'good', '*.png'))
            else:
                self.img_paths = glob.glob(os.path.join(self.data_dir, '**', 'test', '**', '*.png'))
        else:
            type_dir = os.path.join(self.data_dir, self.obj_type)
            if self.mode == 'train':
                img_dir = os.path.join(type_dir, 'train', 'good')
                self.img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            else:
                self.img_paths = glob.glob(os.path.join(type_dir, 'test', '**', '*.png'))
        
        self.img_paths = sorted(self.img_paths)



    def __len__(self):
        return len(self.img_paths)

    def resize(self, sample):
        return np.array(cv2.resize(sample, (self.spatial_size, self.spatial_size)))

    def transform_image(self, image_path, anomaly_source_path, mode='train'):
        # load image
        image = Image.open(image_path)
        
        # pre-process
        preprocessed_image = self.pre_transform(image)
        preprocessed_image_np = np.array(preprocessed_image)
        
        # augment
        if mode == 'train':
            augmented_image, mask, has_anomaly = augment_image(preprocessed_image_np, anomaly_source_path)
        else:
            augmented_image = preprocessed_image_np
        # post_process
        transformed_image = self.post_transform(Image.fromarray(augmented_image.astype(np.uint8)))

        if mode == 'train':
            return transformed_image, mask, has_anomaly
        else:
            return transformed_image
    
    def __getitem__(self, indice):
        """
            returns:
                image: 3,256,256
                mask:  1,256,256
                has_anomaly: 1
                defect_type: 3
                dice: 10
        """
        if self.shot == 'zero':
            img_path = '/root/test/wwl/datasets/white_image.png'
        else:
            img_path = self.img_paths[indice]
        
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good' or self.shot == 'zero':
            has_anomaly = np.array([0], dtype=np.float32)
            mask = np.zeros((self.spatial_size, self.spatial_size)) / 255.0
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)

            has_anomaly = np.array([1], dtype=np.float32)
            mask = get_inputs(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        if self.preprocess is not None:
            image = self.preprocess(Image.open(img_path))
        else:
            # image = self.transform(Image.open(self.img_paths[indice]).convert("RGB"))
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            if self.mode == 'train':
                image, mask, has_anomaly = self.transform_image(img_path, self.anomaly_source_paths[anomaly_source_idx], mode=self.mode)
            else:
                image = self.transform_image(img_path, self.anomaly_source_paths[anomaly_source_idx], mode=self.mode)

        mask = self.resize(mask)
        mask = self.mask_transform(mask)
        if self.shot == 'zero':
            defect_type = np.array([-1], dtype=np.float32)
        else:
            defect_type = DEFECT_TYPE[self.obj_type].index(base_dir)
            defect_type = np.array([defect_type], dtype=np.float32)
        indice = np.array([indice], dtype=np.float32)

        return image, mask, has_anomaly, defect_type, indice