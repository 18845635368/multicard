"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import os
from pathlib import Path
from typing import List

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, IterableDataset

from .utils import extract_bb

'''
:method definition 
-------------------------------
:param record 对应照片的df
:param root 对应数据集切下来的脸所在的目录
:param size 图片size
:param scale 图片需要进行的transform的方式
:param transformer  图片transform用的处理器
-------------------------------
:return face[ndarray] 经处理器处理过后的face


'''


def load_face(record: pd.Series, root: str, size: int, scale: str, transformer: A.BasicTransform) -> torch.Tensor:
    path = os.path.join(str(root), str(record.name))
    # 要是图片size<256，或者使用tight策略时，采用autocache
    autocache = size < 256 or scale == 'tight'
    if scale in ['crop', 'scale', ]:
        cached_path = str(Path(root).joinpath('autocache', scale, str(
            size), str(record.name)).with_suffix('.jpg'))
    else:
        # when self.scale == 'tight' the extracted face is not dependent on size
        cached_path = str(Path(root).joinpath(
            'autocache', scale, str(record.name)).with_suffix('.jpg'))

    face = np.zeros((size, size, 3), dtype=np.uint8)
    # 要是图片已经载入到autocache中，直接用即可
    if os.path.exists(cached_path):
        try:
            face = Image.open(cached_path)
            face = np.array(face)
            if len(face.shape) != 3:
                raise RuntimeError('Incorrect format: {}'.format(path))
        except KeyboardInterrupt as e:
            # We want keybord interrupts to be propagated
            raise e
        except (OSError, IOError) as e:
            print('Deleting corrupted cache file: {}'.format(cached_path))
            print(e)
            os.unlink(cached_path)
            face = np.zeros((size, size, 3), dtype=np.uint8)
    # 要是图片并未被载入，将其根据df给定的boundingbox进行裁切，调整大小，保存，转化为ndarray，然后通过处理器，得到处理过后的ndarray
    if not os.path.exists(cached_path):
        try:
            frame = Image.open(path)
            bb = record['left'], record['top'], record['right'], record['bottom']
            face = extract_bb(frame, bb=bb, size=size, scale=scale)

            if autocache:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                # PIL.image.save(path,此处95即为最高质量，)
                face.save(cached_path, quality=95, subsampling='4:4:4')

            face = np.array(face)
            if len(face.shape) != 3:
                raise RuntimeError('Incorrect format: {}'.format(path))
        except KeyboardInterrupt as e:
            # We want keybord interrupts to be propagated
            raise e
        except (OSError, IOError) as e:
            print('Error while reading: {}'.format(path))
            print(e)
            face = np.zeros((size, size, 3), dtype=np.uint8)

    face = transformer(image=face)['image']

    return face


class FrameFaceIterableDataset(IterableDataset):

    def __init__(self,
                 roots: List[str],
                 dfs: List[pd.DataFrame],
                 size: int, scale: str,
                 num_samples: int = -1,
                 transformer: A.BasicTransform = ToTensorV2(),
                 output_index: bool = False,
                 labels_map: dict = None,
                 seed: int = None):
        """

        :param roots: List of root folders for frames cache
        :param dfs: List of DataFrames of cached frames with 'bb' column as array of 4 elements (left,top,right,bottom)
                   and 'label' column
        :param size: face size
        :param num_samples: the size of maximum_len(df_real,df_fake)
        :param scale: Rescale the face to the given size, preserving the aspect ratio.
                      If false crop around center to the given size
        :param transformer:
        :param output_index: enable output of df_frames index
        :param labels_map: map from 'REAL' and 'FAKE' to actual labels
        """

        self.dfs = dfs
        self.size = int(size)
        # 要是初始化时没有给seed，那么随机生成一个
        self.seed0 = int(
            seed) if seed is not None else np.random.choice(2 ** 32)

        # adapt indices
        dfs_adapted = [df.copy() for df in self.dfs]
        for df_idx, df in enumerate(dfs_adapted):
            mi = pd.MultiIndex.from_tuples(
                [(df_idx, key) for key in df.index], names=['df_idx', 'df_key'])
            df.index = mi
        # Concat
        self.df = pd.concat(dfs_adapted, axis=0, join='inner')

        self.df_real = self.df[self.df['label'] == 0]
        # label为True的都是有篡改的
        self.df_fake = self.df[self.df['label'] == 1]

        # 给定该数据集的主体
        self.longer_set = 'real' if len(
            self.df_real) > len(self.df_fake) else 'fake'
        self.num_samples = max(len(self.df_real), len(self.df_fake)) * 2
        self.num_samples = min(
            self.num_samples, num_samples) if num_samples > 0 else self.num_samples

        self.output_idx = bool(output_index)

        self.scale = str(scale)
        self.roots = [str(r) for r in roots]
        self.transformer = transformer

        self.labels_map = labels_map
        if self.labels_map is None:
            self.labels_map = {False: np.array([0., ]), True: np.array([1., ])}
        else:
            self.labels_map = dict(self.labels_map)

    '''
    :method definition 通过dataframe的index从而获取对应的
    ---------------------------------------------------
    :param item[pd.index] 用于存储需要获得脸部所属dataframe的index
    ---------------------------------------------------
    :return face[ndarray] 经过处理后脸部照片转化而来的ndarray
    :return label[int] 若是假图则为1，真图则为0
    '''

    def _get_face(self, item: pd.Index) -> (torch.Tensor, torch.Tensor) or (torch.Tensor, torch.Tensor, str):
        # 首先定位数据集，然后根据index找到对应图片的df
        record = self.dfs[item[0]].loc[item[1]]
        face = load_face(record=record,
                         root=self.roots[item[0]],
                         size=self.size,
                         scale=self.scale,
                         transformer=self.transformer)
        # 至此，face已经被提取并且经过了预处理
        label = self.labels_map[record.label]
        if self.output_idx:
            return face, label, record.name
        else:
            return face, label

    def __len__(self):
        return self.num_samples

    # 可迭代对象需要实现的一个接口，当对其进行迭代时输出的内容

    '''
    method definition:这是可迭代对象必须实现的一个方法，对其迭代时进行相应返回
    ------------------
    :return 每次迭代会返回一张真脸的ndarray，还有其对应的label
    '''

    def __iter__(self):

        random_fake_idxs, random_real_idxs = get_iterative_real_fake_idxs(
            df_real=self.df_real,
            df_fake=self.df_fake,
            num_samples=self.num_samples,
            seed0=self.seed0
        )

        while len(random_fake_idxs) >= 1 and len(random_real_idxs) >= 1:
            yield self._get_face(random_fake_idxs.pop())
            yield self._get_face(random_real_idxs.pop())


'''
method definition:
------------------
:param df_real[pd.DataFrame]    数据集中所有真实face对应的dataframe
:param df_fake[pd.DataFrame]    数据集中所有篡改face对应的dataframe
:param num_samples[int]         2*max_len(df_real,df_fake)
:param seed0[int]               np.random的random seed
------------------
:return random_fake_idxs[list[df.index]] size为numsamples//2大小的随机排序后的假脸df的index[index 为multiindex，index的首位对应数据集，第二位对应数据集中的图片]
:return random_real_idxs[list[df.index]] size为numsamples//2大小的随机排序后的真脸df的index
'''


def get_iterative_real_fake_idxs(df_real: pd.DataFrame, df_fake: pd.DataFrame,
                                 num_samples: int, seed0: int):
    longer_set = 'real' if len(df_real) > len(df_fake) else 'fake'
    worker_info = torch.utils.data.get_worker_info()
    # 要是没有worker信息就
    if worker_info is None:
        seed = seed0
        np.random.seed(seed)
        # 每次取中数据集的一半
        worker_num_couple_samples = num_samples // 2
        # 要是这是一个真图占主体的训练集，那么假图可以被多次选中
        fake_idxs_portion = np.random.choice(df_fake.index, worker_num_couple_samples,
                                             replace=longer_set == 'real')
        real_idxs_portion = np.random.choice(df_real.index, worker_num_couple_samples,
                                             replace=longer_set == 'fake')
    else:

        worker_id = worker_info.id
        seed = seed0 + worker_id
        np.random.seed(seed)
        # 要是有给线程数，那么就分摊一下
        worker_num_couple_samples = (
            num_samples // 2) // worker_info.num_workers
        if longer_set == 'fake':
            fake_idxs_portion = df_fake.index[
                worker_id * worker_num_couple_samples:(worker_id + 1) * worker_num_couple_samples]
            real_idxs_portion = np.random.choice(
                df_real.index, worker_num_couple_samples, replace=True)
        else:
            real_idxs_portion = df_real.index[
                worker_id * worker_num_couple_samples:(worker_id + 1) * worker_num_couple_samples]
            fake_idxs_portion = np.random.choice(df_fake.index, worker_num_couple_samples,
                                                 replace=True)
    # 最后还得随机调整一下顺序
    random_fake_idxs = list(np.random.permutation(fake_idxs_portion))
    random_real_idxs = list(np.random.permutation(real_idxs_portion))

    assert (len(random_fake_idxs) == len(random_real_idxs))
    # 最终返回的是list，list的元素是真假dataframe的index
    return random_fake_idxs, random_real_idxs


class FrameFaceDataset(Dataset):

    def __init__(self,
                 root: str,
                 df: pd.DataFrame,
                 size: int, scale: str,
                 transformer: A.BasicTransform = ToTensorV2(),
                 labels_map: dict = None,
                 aug_transformers: List[A.BasicTransform] = None):
        """

        :param root: root folder for frames cache
        :param df: DataFrame of cached frames with 'bb' column as array of 4 elements (left,top,right,bottom)
                   and 'label' column
        :param size: face size
        :param num_samples:
        :param scale: Rescale the face to the given size, preserving the aspect ratio.
                      If false crop around center to the given size
        :param transformer:
        :param labels_map: dcit to map df labels
        :param aug_transformers: if not None, creates multiple copies of the same sample according to the provided augmentations
        """

        self.df = df
        self.size = int(size)

        self.scale = str(scale)
        self.root = str(root)
        self.transformer = transformer
        self.aug_transformers = aug_transformers

        self.labels_map = labels_map
        if self.labels_map is None:
            self.labels_map = {False: np.array([0., ]), True: np.array([1., ])}
        else:
            self.labels_map = dict(self.labels_map)

    def _get_face(self, item: pd.Index) -> (torch.Tensor, torch.Tensor) or (torch.Tensor, torch.Tensor, str):
        record = self.df.loc[item]
        label = self.labels_map[record.label]
        if self.aug_transformers is None:
            face = load_face(record=record,
                             root=self.root,
                             size=self.size,
                             scale=self.scale,
                             transformer=self.transformer)
            return face, label
        else:
            faces = []
            for aug_transf in self.aug_transformers:
                faces.append(
                    load_face(record=record,
                              root=self.root,
                              size=self.size,
                              scale=self.scale,
                              transformer=A.Compose(
                                  [aug_transf, self.transformer])
                              ))
            faces = torch.stack(faces)
            return faces, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self._get_face(self.df.index[index])
