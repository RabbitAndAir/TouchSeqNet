import pandas as pd
import random
import itertools
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def read_and_group_by_user(file_path, doc_type_value=None):
    """
    读取CSV文件，并根据user_id进行分类。
    :param file_path: CSV文件路径
    :param doc_type_value: 要筛选的doc_type值，如果为None则不过滤
    :return: 按user_id分类的数据
    """
    # 读取CSV文件
    df = pd.read_csv(file_path, usecols=lambda column: column not in ['Unnamed: 10'])
    # 如果提供了doc_type_value，过滤doc_type
    if doc_type_value is not None:
        df = df[df['doc_type'] == doc_type_value]
    # 按user_id进行分类
    grouped_data = df.groupby(['device_id', 'user_id'])
    # 返回按user_id分类后的数据
    return grouped_data

def extract_sequences(user_data):
    """
    从每个用户的数据中提取符合 action = 0 -> action = 2 -> action = 1 的多维时间序列。
    :param user_data: 某个用户的所有时间序列数据
    :return: 该用户提取的时间序列列表
    """
    # 定义要提取的特征列
    # features = ['time', 'x_coor', 'y_coor', 'pressure', 'finger_area']
    features = ['x_coor', 'y_coor', 'pressure', 'finger_area']
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

def process_device_user_sequences(grouped_data):
    """
    对所有device_id和user_id的分组数据进行处理，提取时间序列。
    :param grouped_data: 按device_id和user_id分组的数据
    :return: 每个分组的多维时间序列列表
    """
    sequences_by_device_user = {}
    total_sequences = 0  # 计数器，记录提取的总序列数量
    for (device_id, user_id), user_data in grouped_data:
        # print(f"处理设备: {device_id}, 用户: {user_id}")
        sequences = extract_sequences(user_data)
        sequences_by_device_user[(device_id, user_id)] = sequences
        # 获取每个用户的提取到的序列数量
        user_sequence_count = len(sequences)
        total_sequences += user_sequence_count  # 累加到总序列计数器
        print(f"用户 {user_id}, 设备 {device_id} 提取到的序列数量: {len(sequences)}")
    print(f"提取到的总序列数量: {total_sequences}")
    return sequences_by_device_user

def z_score_normalize(dataset, epsilon=1e-8):
    """
    z-score归一化（标准化），即将每个特征的均值减去，再除以标准差。这样可以更好地处理数据中的极端值，避免因某些特征范围过大而影响模型学习。
    :param dataset: 提取到的时间序列
    :return: 归一化后的数据集
    """
    normalized_data = []
    for time_series in dataset:
        num_features = len(time_series[0])
        normalized_time_series = [[0] * num_features for _ in range(len(time_series))]
        for feature_idx in range(num_features):
            feature_values = [timestep[feature_idx] for timestep in time_series]
            mean = sum(feature_values) / len(feature_values)
            std = (sum([(x - mean) ** 2 for x in feature_values]) / len(feature_values)) ** 0.5
            for timestep_idx in range(len(time_series)):
                normalized_value = (time_series[timestep_idx][feature_idx] - mean) / (std + epsilon)
                normalized_time_series[timestep_idx][feature_idx] = normalized_value
        normalized_data.append(normalized_time_series)
    return normalized_data

def normalize_device_user_sequences(sequences_by_device_user):
    """
    对每个设备和用户分组的所有多维时间序列进行归一化处理。
    :param sequences_by_device_user: 按device_id和user_id分组的时间序列
    :return: 归一化后的时间序列
    """
    normalized_sequences_by_device_user = {}
    for (device_id, user_id), sequences in sequences_by_device_user.items():
        # print(f"对设备 {device_id}, 用户 {user_id} 进行归一化处理...")
        normalized_sequences = z_score_normalize(sequences)
        normalized_sequences_by_device_user[(device_id, user_id)] = normalized_sequences
    return normalized_sequences_by_device_user

def filter_extreme_sequences(user_sequences, lower_percentile=0.2, upper_percentile=0.8, verbose=True):
    """
    过滤掉每个用户中最长的 upper_percentile 和最短的 lower_percentile 的时间序列。
    :param user_sequences: 每个用户的多维时间序列
    :param lower_percentile: 最短的百分比（默认为 0.1，表示最短的 10%）
    :param upper_percentile: 最长的百分比（默认为 0.9，表示最长的 90%）
    :param verbose: 是否打印每个用户的过滤范围
    :return: 过滤后的时间序列
    """
    filtered_sequences_by_user = {}
    for user_id, sequences in user_sequences.items():
        # 计算每个时间序列的长度
        lengths = [len(seq) for seq in sequences]
        if len(lengths) < 2:
            # 如果序列数量太少，跳过过滤
            filtered_sequences_by_user[user_id] = sequences
            if verbose:
                print(f"用户 {user_id}: 序列数量太少，跳过过滤")
            continue
        # 按长度排序
        sorted_lengths = sorted(lengths)
        # 计算 lower 和 upper 百分位的索引
        lower_index = max(0, int(len(sorted_lengths) * lower_percentile))
        upper_index = min(len(sorted_lengths) - 1, int(len(sorted_lengths) * upper_percentile) - 1)
        # 获取阈值范围
        lower_threshold = sorted_lengths[lower_index]
        upper_threshold = sorted_lengths[upper_index]
        # if verbose:
        #     print(f"用户 {user_id}: 序列长度范围 {lower_threshold} 到 {upper_threshold}")
        # 过滤掉不在长度范围内的时间序列
        filtered_sequences = [seq for seq in sequences if lower_threshold <= len(seq) <= upper_threshold]
        # 保存过滤后的序列
        filtered_sequences_by_user[user_id] = filtered_sequences

    return filtered_sequences_by_user


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

def generate_negative_pairs(sequences_by_device_user):
    """
    生成负样本对，在 device_id 和 user_id 都不同的分组之间两两组合并打标签0。
    :param sequences_by_device_user: 按 device_id 和 user_id 分组的时间序列
    :return: 负样本对的列表
    """
    negative_pairs = []
    # 获取所有分组的组合
    for (dev_user_1, sequences1), (dev_user_2, sequences2) in itertools.combinations(sequences_by_device_user.items(), 2):
        # 确保 device_id 和 user_id 都不同
        device_id_1, user_id_1 = dev_user_1
        device_id_2, user_id_2 = dev_user_2

        if device_id_1 != device_id_2 and user_id_1 != user_id_2:
            # 如果 device_id 和 user_id 都不同，则生成负样本对
            for seq1, seq2 in itertools.product(sequences1, sequences2):
                negative_pairs.append((seq1, seq2, 0))  # 标签 0 表示负样本
    return negative_pairs

def generate_sample_pairs(sequences_by_device_user):
    """
    构建正负样本对，正样本对是同一分组内部的两两组合，
    负样本对是不同分组之间的两两组合。
    :param sequences_by_device_user: 按device_id和user_id分组的时间序列
    :return: 正样本对列表，负样本对列表
    """
    positive_pairs = []
    negative_pairs = []
    # 生成正样本对
    for sequences in sequences_by_device_user.values():
        positive_pairs.extend(generate_positive_pairs(sequences))

    # 生成负样本对
    negative_pairs.extend(generate_negative_pairs(sequences_by_device_user))
    # 保持正负样本对数量一致
    min_samples = min(len(positive_pairs), len(negative_pairs))
    if len(positive_pairs) > min_samples:
        positive_pairs = random.sample(positive_pairs, min_samples)
    elif len(negative_pairs) > min_samples:
        negative_pairs = random.sample(negative_pairs, min_samples)
    print(f"正样本对数量: {len(positive_pairs)}")
    print(f"负样本对数量: {len(negative_pairs)}")

    return positive_pairs, negative_pairs


def create_biodent_sets(file_path, wave_length=8):
    """
    处理整个流程：读取数据，归一化，过滤序列，最后生成正负样本对。
    :param file_path: CSV 文件路径
    :param user_id: 当前用户 ID
    :param sample_percent: 随机选择负样本的百分比（默认10%）
    :return: 正样本对列表，负样本对列表
    """
    sample_percent = 0.5
    # 第一步：读取 CSV 文件并按 user_id 分组
    grouped_data = read_and_group_by_user(file_path, doc_type_value=1)
    # 第二步：提取时间序列
    user_sequences = process_device_user_sequences(grouped_data)
    total_sample = sum(len(v) for v in user_sequences.values())
    # 第三步：归一化时间序列
    normalized_user_sequences = normalize_device_user_sequences(user_sequences)
    # 第四步：过滤掉最长和最短的 10% 时间序列
    filtered_user_sequences = filter_extreme_sequences(normalized_user_sequences)
    # filtered_user_sequences = normalized_user_sequences  #测试数据流
    # 第五步：生成正负样本对
    positive_pairs, negative_pairs = generate_sample_pairs(filtered_user_sequences)
    # 取正负样本的一定百分比
    num_positive_samples = int(len(positive_pairs) * sample_percent)
    num_negative_samples = int(len(negative_pairs) * sample_percent)

    # 随机抽取正样本对和负样本对
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

    data_shape = (4, max_train_len)

    return train_pairs, val_pairs, max_train_len, max_val_len, data_shape

class BiodentDataset(Dataset):
    def __init__(self, pairs):
        """
        初始化数据集，传入样本对和对应的标签
        :param pairs: 样本对列表，列表中的每个元素为 (seq1, seq2)
        :param labels: 标签列表，列表中的每个元素为标签（0 或 1）
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
        seq1 = torch.tensor(seq1, dtype=torch.float32)  # [sequence_length, num_features]
        seq2 = torch.tensor(seq2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return seq1, seq2, label

def Biodent_dataloader(pairs, max_len, batch_size=4, shuffle=True):
    # 创建训练数据集
    dataset = BiodentDataset(pairs)
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



