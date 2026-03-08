import os
import random

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.ffingers_help import load_data, generate_same_person_same_gesture, \
    generate_different_person_different_gesture

def create_ffinger_sets(file_path, wave_length=8, split_ratio=0.5):

    # 1. 生成正负样本对
    data_dict = load_data(file_path)
    positive_pairs = generate_same_person_same_gesture(data_dict)
    negative_pairs = generate_different_person_different_gesture(data_dict)

    # 2. 正负样本对数一致
    num_positive = len(positive_pairs)
    num_negative = len(negative_pairs)
    # 如果正负样本对数量不一致，随机采样数量多的样本对
    if num_positive != num_negative:
        min_count = min(num_positive, num_negative)
        if num_positive > num_negative:
            positive_pairs = random.sample(positive_pairs, min_count)
        else:
            negative_pairs = random.sample(negative_pairs, min_count)

    labeled_positive_pairs = [(pair, 1) for pair in positive_pairs]
    labeled_negative_pairs = [(pair, 0) for pair in negative_pairs]
    all_pairs = labeled_positive_pairs + labeled_negative_pairs
    print("all_pairs length: ", len(all_pairs))
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.4, random_state=42)

    # 5. 计算最大时间步长
    max_train_len = max(
        max(
            max(
                len(sub_feature) for feature in sample['time_series_data'] for sub_feature in feature
            )
            for sample in [pair[0]['sample1'], pair[0]['sample2']]
        )
        for pair in train_pairs
    )
    max_val_len = max(
        max(
            max(
                len(sub_feature) for feature in sample['time_series_data'] for sub_feature in feature
            )
            for sample in [pair[0]['sample1'], pair[0]['sample2']]
        )
        for pair in val_pairs
    )

    max_train_len = (max_train_len + (wave_length - max_train_len % wave_length)) if max_train_len % wave_length != 0 else max_train_len

    max_val_len = (max_val_len + (wave_length - max_val_len % wave_length)) if max_val_len % wave_length != 0 else max_val_len

    data_shape = (25, max_train_len)
    return train_pairs, val_pairs, max_train_len, max_val_len, data_shape

class Ffinger_Dataset(Dataset):
    def __init__(self, val_pairs):
        """
        :param labeled_pairs: 每个元素是一个元组 (样本对, label)
        """
        self.val_pairs = val_pairs

    def __len__(self):
        return len(self.val_pairs)

    def __getitem__(self, idx):
        # 获取样本对和label
        sample_pair, label = self.val_pairs[idx]
        # 从sample1和sample2中获取'time_series_data'
        time_series_1 = sample_pair['sample1']['time_series_data']
        time_series_2 = sample_pair['sample2']['time_series_data']
        # 返回命名为 seq1, seq2 和label
        return time_series_1, time_series_2, label

def Ffinger_dataloader(pairs, max_len, batch_size=4, shuffle=True):

   # 创建训练数据集
    dataset = Ffinger_Dataset(pairs)
    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: _collate_fn(batch, max_len))
    return data_loader

def _collate_fn(batch, max_data_len):
    seq1_list = []
    seq2_list = []
    labels_list = []
    seq_lengths = []  # 用于记录每个样本对的最大长度

    # 遍历每个样本对，收集数据并记录每个样本对的最大长度
    for seq1, seq2, label in batch:
        seq1_list.append(seq1)
        seq2_list.append(seq2)
        labels_list.append(label)

        seq1_max_len = max(len(finger_seq[0]) for finger_seq in seq1)  # seq1中每根手指的最大时间步数
        seq2_max_len = max(len(finger_seq[0]) for finger_seq in seq2)  # seq2中每根手指的最大时间步数

        # 记录每个样本对的最大长度
        max_len = max(seq1_max_len, seq2_max_len)
        seq_lengths.append(max_len)

    # 计算整个批次的最大序列长度
    # max_seq_length = max(seq_lengths)
    max_seq_length = max_data_len

    padded_seq1_list = []
    padded_seq2_list = []
    seq1_masks = []
    seq2_masks = []

    for seq1, seq2 in zip(seq1_list, seq2_list):
        # 处理seq1
        padded_seq1 = []
        seq1_max_len = max(len(finger_seq[0]) for finger_seq in seq1)
        for finger_seq in seq1:
            # 填充每根手指的时间序列到 max_seq_length
            padded_finger = []
            for feature in finger_seq:
                # 填充每个特征到 max_seq_length
                padded_feature = feature + [0.0] * (max_seq_length - len(feature))
                padded_finger.append(padded_feature)
            padded_seq1.append(padded_finger)
        padded_seq1_list.append(padded_seq1)

        # 处理seq2
        padded_seq2 = []
        seq2_max_len = max(len(finger_seq[0]) for finger_seq in seq2)
        for finger_seq in seq2:
            padded_finger = []
            for feature in finger_seq:
                padded_feature = feature + [0.0] * (max_seq_length - len(feature))
                padded_finger.append(padded_feature)
            padded_seq2.append(padded_finger)
        padded_seq2_list.append(padded_seq2)

        # 生成掩码
        mask1 = [1] * seq1_max_len + [0] * (max_seq_length - seq1_max_len)
        mask2 = [1] * seq2_max_len + [0] * (max_seq_length - seq2_max_len)
        seq1_masks.append(mask1)
        seq2_masks.append(mask2)

    # 转换为张量
    # padded_seq1_list 和 padded_seq2_list 的形状: [batch_size, 5, num_features, max_seq_length]
    padded_seq1_batch = torch.tensor(padded_seq1_list,
                                     dtype=torch.float32)
    padded_seq2_batch = torch.tensor(padded_seq2_list,
                                     dtype=torch.float32)

    # 生成掩码张量
    seq1_mask_batch = torch.tensor(seq1_masks, dtype=torch.float32)  # [batch_size, max_seq_length]
    seq2_mask_batch = torch.tensor(seq2_masks, dtype=torch.float32)  # [batch_size, max_seq_length]

    # label张量
    labels_batch = torch.tensor(labels_list, dtype=torch.long)  # [batch_size]

    batch_size, num_features, max_seq_length = padded_seq1_batch.shape[0], padded_seq1_batch.shape[1], padded_seq1_batch.shape[3]
    padded_seq1_batch = padded_seq1_batch.view(batch_size, -1,
                                               max_seq_length)  # [batch_size, 5*num_features, max_seq_length]
    padded_seq2_batch = padded_seq2_batch.view(batch_size, -1,
                                               max_seq_length)  # [batch_size, 5*num_features, max_seq_length]

    return padded_seq1_batch, padded_seq2_batch, labels_batch, seq1_mask_batch, seq2_mask_batch