import itertools
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def read_and_group_by_user(file_path, doc_type_value=None, phone_orientaton=None):
    """
    读取CSV文件，并根据user_id进行分类。
    :param file_path: CSV文件路径
    :param doc_type_value: 要筛选的doc_type值，如果为None则不过滤
    :return: 按user_id分类的数据
    """
    # 读取CSV文件
    pd.set_option('display.float_format', '{:.12f}'.format)
    df = pd.read_csv(file_path, usecols=lambda column: column not in ['Unnamed: 10'])
    # 如果提供了doc_type_value，过滤doc_type
    if doc_type_value is not None:
        df = df[df['doc_type'] == doc_type_value]
    if phone_orientaton is not None:
        df = df[df['phone_orientation'] == phone_orientaton]
    # 按user_id进行分类
    grouped_data = df.groupby(['user_id'])
    # 返回按user_id分类后的数据
    return grouped_data

def extract_sequences(user_data):
    """
    从每个用户的数据中提取符合 action = 0 -> action = 2 -> action = 1 的多维时间序列。
    :param user_data: 某个用户的所有时间序列数据
    :return: 该用户提取的时间序列列表
    """
    # 定义要提取的特征列
    features = ['time', 'x_coor', 'y_coor', 'pressure', 'finger_area']
    sequences = []  # 用于存储所有有效的时间序列
    current_sequence = []  # 当前正在构建的序列
    valid_sequence = False  # 标记当前序列是否有效
    last_action = None  # 跟踪上一次的 action 值
    # 遍历用户的每一行数据
    for index, row in user_data.iterrows():
        if row['action'] == 0:  # 如果 action 是 0，开始新的序列
            if valid_sequence and current_sequence:
                sequences.append(current_sequence)  # 将当前序列保存为有效序列
            current_sequence = [[row[feature] for feature in features]]  # 开始新的序列
            valid_sequence = False  # 重新设置序列有效标志
            last_action = row['action']  # 更新上一次的 action 值

        elif row['action'] == 2 and current_sequence:  # 如果 action 是 2，继续构建序列
            if last_action == 0 or last_action == 2:
                current_sequence.append([row[feature] for feature in features])
                last_action = row['action']  # 更新上一次的 action 值
            else:
                # 遇到不符合顺序的情况，清空当前序列
                current_sequence = []
                valid_sequence = False
                last_action = None
        elif row['action'] == 1 and current_sequence:  # 如果 action 是 1，结束序列
            if last_action == 2:
                current_sequence.append([row[feature] for feature in features])
                valid_sequence = True  # 标记该序列有效
                last_action = row['action']  # 更新上一次的 action 值
            else:
                # 遇到不符合顺序的情况，清空当前序列
                current_sequence = []
                valid_sequence = False
                last_action = None
    # 如果最后一个序列是有效的，也需要添加
    if valid_sequence and current_sequence:
        sequences.append(current_sequence)
    return sequences

# 提取每个用户的时间序列
def extract_all_user_sequences(grouped_data):
    """
    从分组后的用户数据中提取所有用户的多维时间序列。
    :param grouped_data: 按 user_id 分组的数据
    :return: 包含所有用户序列的字典 {user_id: sequences}
    """
    user_sequences = {}
    for user_id, user_data in grouped_data:  # 遍历每个用户分组
        sequences = extract_sequences(user_data)  # 提取当前用户的时间序列
        user_sequences[user_id] = sequences  # 将序列存储到字典中
    print(f"Total users: {len(user_sequences)}")
    return user_sequences


def filter_extreme_sequences(user_sequences, lower_percentile=0.2, upper_percentile=0.8, verbose=True):
    """
    过滤掉每个用户中最长的 upper_percentile 和最短的 lower_percentile 的时间序列。
    :param user_sequences: 每个用户的多维时间序列
    :param lower_percentile: 最短的百分比（默认为 0.2，表示最短的 20%）
    :param upper_percentile: 最长的百分比（默认为 0.8，表示最长的 80%）
    :param verbose: 是否打印每个用户的过滤范围
    :return: 过滤后的时间序列
    """
    filtered_sequences_by_user = {}
    for user_id, sequences in user_sequences.items():
        # 计算每个时间序列的长度
        lengths = [len(seq) for seq in sequences]
        if len(lengths) < 3:  # 序列太少（少于3个），跳过过滤
            filtered_sequences_by_user[user_id] = sequences
            if verbose:
                print(f"用户 {user_id}: 序列数量太少（{len(lengths)} 个），跳过过滤")
            continue

        # 按长度排序
        sorted_lengths = sorted(lengths)
        # 计算 lower 和 upper 百分位的索引
        lower_index = max(0, int(len(sorted_lengths) * lower_percentile))
        upper_index = min(len(sorted_lengths) - 1, int(len(sorted_lengths) * upper_percentile) - 1)
        # lower_index = sorted_lengths[max(0, int(len(sorted_lengths) * lower_percentile))]
        # upper_index = sorted_lengths[max(0, int(len(sorted_lengths) * upper_percentile))]
        # 获取阈值范围
        lower_threshold = sorted_lengths[lower_index]
        upper_threshold = sorted_lengths[upper_index - 1]  # upper_index - 1 确保包括上边界
        # 打印阈值范围
        if verbose:
            print(f"用户 {user_id}: 序列长度范围 {lower_threshold} 到 {upper_threshold} （共 {len(lengths)} 个序列）")
        # 过滤掉不在长度范围内的时间序列
        filtered_sequences = [seq for seq in sequences if lower_threshold <= len(seq) <= upper_threshold]
        # 打印过滤后的信息
        if verbose:
            print(f"用户 {user_id}: 过滤后保留 {len(filtered_sequences)} 个序列")
        # 保存过滤后的序列
        filtered_sequences_by_user[user_id] = filtered_sequences

    return filtered_sequences_by_user


def preprocess_and_normalize(dataset, epsilon=1e-8):
    """
    对时间序列数据进行预处理：
    1. 对 'time', 'x_coor', 'y_coor' 特征进行差分处理，并将第一时刻设置为 0。
    2. 对 'pressure' 和 'finger_area' 特征进行 Z-Score 标准化。

    :param dataset: 提取到的时间序列，格式为 [[[feature1, feature2, ...], ...], ...]
    :param epsilon: 防止除零的微小值，默认值为 1e-8
    :return: 预处理后的数据集
    """
    processed_data = []

    for time_series in dataset:
        num_features = len(time_series[0])
        processed_time_series = [[0] * num_features for _ in range(len(time_series))]

        # 1. 对 'time', 'x_coor', 'y_coor'（前三个特征）进行差分处理
        for feature_idx in range(3):  # 前三个特征
            # 提取当前特征的所有时间步的值
            feature_values = [timestep[feature_idx] for timestep in time_series]
            # 差分：第一时刻为 0，其余为差分值
            diff_values = [0] + [feature_values[t + 1] - feature_values[t] for t in range(len(feature_values) - 1)]
            # 将差分后的值写入 processed_time_series
            for timestep_idx, value in enumerate(diff_values):
                processed_time_series[timestep_idx][feature_idx] = value

        # 2. 对 'pressure' 和 'finger_area'（后两个特征）进行 Z-Score 标准化
        for feature_idx in range(3, num_features):  # 后两个特征
            # 提取当前特征的所有时间步的值
            feature_values = [timestep[feature_idx] for timestep in time_series]
            mean = sum(feature_values) / len(feature_values)  # 计算均值
            std = (sum([(x - mean) ** 2 for x in feature_values]) / len(feature_values)) ** 0.5  # 计算标准差
            # 标准化处理
            normalized_values = [(x - mean) / (std + epsilon) for x in feature_values]
            # 将标准化后的值写入 processed_time_series
            for timestep_idx, value in enumerate(normalized_values):
                processed_time_series[timestep_idx][feature_idx] = value

        # 保存处理后的时间序列
        processed_data.append(processed_time_series)

    return processed_data

def normalize_user_sequences(sequence):
    """
    对每个设备和用户分组的所有多维时间序列进行归一化处理。
    :param sequences_by_user: 按device_id和user_id分组的时间序列
    :return: 归一化后的时间序列
    """
    normalized_sequences_by_user = {}
    for user_id, sequences in sequence.items():
        # print(f"对设备 {device_id}, 用户 {user_id} 进行归一化处理...")
        normalized_sequences = preprocess_and_normalize(sequences)
        normalized_sequences_by_user[(user_id)] = normalized_sequences
    return normalized_sequences_by_user

def generate_sample_pairs(sequences_by_user):
    """
    构建正负样本对，正样本对是同一分组内部的两两组合，
    负样本对是不同分组之间的两两组合。
    :param sequences_by_user: 按user_id分组的时间序列
    :return: 正样本对列表，负样本对列表
    """
    positive_pairs = []
    negative_pairs = []
    # 生成正样本对
    for sequences in sequences_by_user.values():
        positive_pairs.extend(generate_positive_pairs(sequences))

    # 生成负样本对
    negative_pairs.extend(generate_negative_pairs(sequences_by_user))
    # 保持正负样本对数量一致
    min_samples = min(len(positive_pairs), len(negative_pairs))
    if len(positive_pairs) > min_samples:
        positive_pairs = random.sample(positive_pairs, min_samples)
    elif len(negative_pairs) > min_samples:
        negative_pairs = random.sample(negative_pairs, min_samples)
    print(f"正样本对数量: {len(positive_pairs)}")
    print(f"负样本对数量: {len(negative_pairs)}")

    return positive_pairs, negative_pairs

def generate_negative_pairs(sequences_by_device_user):
    """
    生成负样本对，在 device_id 和 user_id 都不同的分组之间两两组合并打标签0。
    :param sequences_by_device_user: 按 device_id 和 user_id 分组的时间序列
    :return: 负样本对的列表
    """
    negative_pairs = []
    # 获取所有分组的组合
    for (user_id_1, sequences1), (user_id_2, sequences2) in itertools.combinations(sequences_by_device_user.items(), 2):
        if user_id_1 != user_id_2:
            # 如果 device_id 和 user_id 都不同，则生成负样本对
            for seq1, seq2 in itertools.product(sequences1, sequences2):
                negative_pairs.append((seq1, seq2, 0))  # 标签 0 表示负样本
    return negative_pairs

def generate_positive_pairs(sequences):
    """
    生成正样本对，在同一分组内两两组合并打标签1
    :param sequences: 当前分组的时间序列列表
    :return: 正样本对的列表
    """
    positive_pairs = []
    for seq1, seq2 in itertools.combinations(sequences, 2):
        positive_pairs.append((seq1, seq2, 1))  # 标签 1 表示正样本
    return positive_pairs

def create_touchalytics_sets(file_path, wave_length=8):
    """
    处理整个流程：读取数据，归一化，过滤序列，最后生成正负样本对。
    :param file_path: CSV 文件路径
    :param sample_percent: 随机选择正负样本的百分比（默认10%）
    :return: 正样本对列表，负样本对列表
    """

    sample_percent = 0.5
    # 验证 sample_percent 参数
    if not (0 <= sample_percent <= 1):
        raise ValueError("sample_percent 必须在 [0, 1] 范围内")

    # 第一步：读取 CSV 文件并按 user_id 分组
    grouped_data = read_and_group_by_user(file_path, doc_type_value=1)

    # 第二步：提取时间序列
    user_sequences = extract_all_user_sequences(grouped_data)
    total_sample = sum(len(v) for v in user_sequences.values())

    # 第三步：归一化时间序列
    normalized_user_sequences = normalize_user_sequences(user_sequences)

    # 第四步：过滤掉最长和最短的 10% 时间序列
    filtered_user_sequences = filter_extreme_sequences(normalized_user_sequences)
    # 第五步：生成正负样本对
    positive_pairs, negative_pairs = generate_sample_pairs(filtered_user_sequences)

    # 第六步：随机抽取一定比例的正负样本
    num_positive_samples = int(len(positive_pairs) * sample_percent)
    num_negative_samples = int(len(negative_pairs) * sample_percent)

    if num_positive_samples > 0:
        positive_pairs = random.sample(positive_pairs, num_positive_samples)
    if num_negative_samples > 0:
        negative_pairs = random.sample(negative_pairs, num_negative_samples)

    all_pairs = positive_pairs + negative_pairs
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.4, random_state=42)

    max_train_len = 0
    max_val_len = 0
    # 5. 计算最大时间步长
    for seq1, seq2, _ in positive_pairs:
        max_train_len = max(max_train_len, len(seq1), len(seq2))

    for seq1, seq2, _ in negative_pairs:
        max_val_len = max(max_val_len, len(seq1), len(seq2))

    # max_train_len = 0
    # max_val_len = 0
    # total_train_len = 0
    # train_count = 0
    #
    # # 计算最大和平均训练时间步长度
    # for seq1, seq2, _ in positive_pairs:
    #     max_train_len = max(max_train_len, len(seq1), len(seq2))
    #     total_train_len += len(seq1) + len(seq2)
    #     train_count += 2
    #
    # avg_train_len = total_train_len / train_count if train_count > 0 else 0
    #
    # # 验证集类似
    # total_val_len = 0
    # val_count = 0
    # for seq1, seq2, _ in positive_pairs:
    #     max_val_len = max(max_val_len, len(seq1), len(seq2))
    #     total_val_len += len(seq1) + len(seq2)
    #     val_count += 2
    #
    # avg_val_len = total_val_len / val_count if val_count > 0 else 0

    max_train_len = (max_train_len + (
                wave_length - max_train_len % wave_length)) if max_train_len % wave_length != 0 else max_train_len

    max_val_len = (max_val_len + (
                wave_length - max_val_len % wave_length)) if max_val_len % wave_length != 0 else max_val_len

    data_shape = (5, max_train_len)

    return train_pairs, val_pairs, max_train_len, max_val_len, data_shape


class TouchalyticsDataset(Dataset):
    def __init__(self, pairs):
        """
        :param labeled_pairs: 每个元素是一个元组 (样本对, label)
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        根据索引返回样本对和对应的标签
        :param idx: 索引
        :return: (seq1, seq2, label)
        """
        seq1, seq2, label = self.pairs[idx]
        # 转换为 PyTorch 张量，并确保维度顺序正确
        seq1 = torch.tensor(seq1, dtype=torch.float32)
        seq2 = torch.tensor(seq2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return seq1, seq2, label


# 取负样本的比例
def Touchalytics_dataloader(pairs, max_len, batch_size=4, shuffle=True):
    # 创建训练数据集
    dataset = TouchalyticsDataset(pairs)
    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: _collate_fn(batch, max_len), num_workers=8)
    return data_loader


def _collate_fn(batch, max_data_len):
    seq1_list = []
    seq2_list = []
    labels_list = []
    seq_lengths = []

    # 遍历每个样本对
    for seq1, seq2, label in batch:
        seq1_list.append(seq1)
        seq2_list.append(seq2)
        labels_list.append(label)
        seq_lengths.append(max(seq1.size(0), seq2.size(0)))  # 记录最长的时间步长度
    # 找到批次中最大时间步长度
    max_seq_length = max_data_len
    # 填充时间序列
    padded_seq1_list = [torch.nn.functional.pad(seq1, (0, 0, 0, max_seq_length - seq1.size(0))) for seq1 in seq1_list]
    padded_seq2_list = [torch.nn.functional.pad(seq2, (0, 0, 0, max_seq_length - seq2.size(0))) for seq2 in seq2_list]
    # 生成掩码，1 表示有效数据，0 表示填充部分
    seq1_mask = torch.stack([torch.arange(max_seq_length) < seq.size(0) for seq in seq1_list]).bool()
    seq2_mask = torch.stack([torch.arange(max_seq_length) < seq.size(0) for seq in seq2_list]).bool()

    # 堆叠成批次
    padded_seq1_batch = torch.stack(padded_seq1_list).permute(0, 2, 1)  # [batch_size, input_channels, max_seq_length]
    padded_seq2_batch = torch.stack(padded_seq2_list).permute(0, 2, 1)
    # 将标签列表转换为张量
    labels_batch = torch.tensor(labels_list, dtype=torch.float32)
    return padded_seq1_batch, padded_seq2_batch, labels_batch, seq1_mask, seq2_mask