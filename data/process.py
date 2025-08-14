import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image3d']
        time = sample['time']
        status = sample['status']
        radiomic = sample['radiomic']
        if random.random() < 0.5:
            # 对除了前三个特征之外的其他特征添加噪声
            noise = np.random.normal(0, 0.1, len(radiomic) - 3)
            radiomic = radiomic[:3] + [r + n for r, n in zip(radiomic[3:], noise)]
        if random.random() < 0.5:
            image = np.flip(image, 0)

        if random.random() < 0.5:
            image = np.flip(image, 1)

        if random.random() < 0.5:
            image = np.flip(image, 2)
        return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}


class RandomIntensityChange(object):
    def __init__(self, intensity_shift_range=(-0.1, 0.1), intensity_scale_range=(0.9, 1.1)):
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range

    def __call__(self, sample):
        image = sample['image3d']
        shift_value = np.random.uniform(self.intensity_shift_range[0], self.intensity_shift_range[1])
        scale_value = np.random.uniform(self.intensity_scale_range[0], self.intensity_scale_range[1])
        image = image * scale_value + shift_value
        sample['image3d'] = image
        return sample


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image3d']
        time = sample['time']
        status = sample['status']
        radiomic = sample['radiomic']
        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}

class guiyihua(object):
    def __call__(self, sample):
        image = sample['image3d']
        time = sample['time']

        status = sample['status']
        radiomic = sample['radiomic']

        mean = np.mean(image, axis=(0, 1, 2))
        std = np.std(image, axis=(0, 1, 2))

        # 对每个像素点进行归一化
        for i in range(4):
            image[:, :, :, i] = (image[:, :, :, i] - mean[i]) / std[i]

        return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}



class Pad(object):
    def __call__(self, sample):
        image = sample['image3d']
        time = sample['time']

        status = sample['status']
        radiomic = sample['radiomic']

        image = np.pad(image, ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')

        return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}
    #(240,240,155)>(240,240,160)

class RandomNoiseAugmentor(object):
    def __call__(self, sample):
        image = sample['image3d']
        time = sample['time']
        status = sample['status']
        radiomic = sample['radiomic']
        noise_prob=0.5
        if np.random.rand() < noise_prob:
            noise_type = np.random.choice(['gaussian', 'salt_and_pepper'])
            if noise_type == 'gaussian':
                image=self.add_gaussian_noise(image)
                return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}
            elif noise_type == 'salt_and_pepper':
                image=self.add_salt_and_pepper_noise(image)
                return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}
        else:
            return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}

    def add_gaussian_noise(self, image):
        mean = 0
        noise_strength=random.uniform(0, 0.1)
        var = noise_strength ** 2
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, -2, 2)
        return noisy_image.astype(np.float32)

    def add_salt_and_pepper_noise(self, image):
        p = random.uniform(0, 0.1)
        noisy_image = np.copy(image)
        salt = np.random.choice([0, 1], size=image.shape, p=[1 - p, p])
        pepper = np.random.choice([0, 1], size=image.shape, p=[1 - p, p])
        noisy_image[salt == 1] = 1
        noisy_image[pepper == 1] = -1
        return noisy_image.astype(np.float32)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image3d']
        time = sample['time']
        status = sample['status']
        radiomic=sample['radiomic']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))

        image = torch.from_numpy(image.copy()).float()

        return {'image3d': image, 'time':time,'status':status,'radiomic':radiomic}



def transform3d(sample):
    trans = transforms.Compose([
        guiyihua(),
        Pad(),
        RandomNoiseAugmentor(),
        RandomIntensityChange(),
        Random_rotate(),  # time-consuming
        Random_Flip(),
        ToTensor()
    ])
    return trans(sample)


def transform_valid3d(sample):
    trans = transforms.Compose([
        guiyihua(),
        Pad(),
        ToTensor()
    ])

    return trans(sample)


class process(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name)
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]

        if self.mode == 'train':

            image3d,time,status,radiomic = pkload(path + 'sr.pkl')
            sample = {'image3d': image3d,'time': time,'status':status,'radiomic':radiomic}
            sample = transform3d(sample)
            sample['radiomic'] = np.array(sample['radiomic']).astype(np.float32)
            return sample['image3d'], sample['time'], sample['status'], sample['radiomic']
            # return sample['image3d'], sample['time'],sample['status'],sample['radiomic'],sample['clinical'],sample['hcr']
        elif self.mode == 'valid':
            image3d,image2d, time,status,radiomic = pkload(path + 'fusion.pkl')
            sample = {'image3d': image3d, 'time': time, 'status': status, 'radiomic': radiomic}
            sample = transform_valid3d(sample)
            sample['radiomic'] = np.array(sample['radiomic']).astype(np.float32)

            return sample['image3d'], sample['time'], sample['status'], sample['radiomic']
        else:
            image = pkload(path + 'fusion.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]




