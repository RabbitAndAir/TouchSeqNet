import os
import re
import itertools
import pandas as pd
from collections import defaultdict


def z_score_normalize(dataset, epsilon=1e-8):
    """
    z-score归一化（标准化），即将每个特征的均值减去，再除以标准差。
    :param dataset: 提取到的时间序列
    :return: 归一化后的loss集
    """
    normalized_data = []

    for time_series in dataset:
        # 1. 剔除第3、7、8、9、10个特征（即二级列表中的第3个和最后四个特征）
        time_series = [time_series[i] for i in range(len(time_series)) if i not in [2, 6, 7, 8, 9]]

        # 2. 对第1个和第2个特征进行差分处理
        for feature_idx in range(2):  # 只处理前两个特征
            feature_values = time_series[feature_idx]
            # 对每个时间步进行差分，第一时刻设置为0
            diff_values = [0] + [feature_values[t + 1] - feature_values[t] for t in range(len(feature_values) - 1)]
            time_series[feature_idx] = diff_values  # 将差分后的值赋回去

        # 3. 对剩余的特征进行标准化处理
        # 使用z-score归一化处理
        for feature_idx in range(2, len(time_series)):
            feature_values = time_series[feature_idx]
            num_timesteps = len(feature_values)
            mean = sum(feature_values) / num_timesteps
            std = (sum([(x - mean) ** 2 for x in feature_values]) / num_timesteps) ** 0.5
            # 对每个时间步进行标准化
            for timestep_idx in range(num_timesteps):
                time_series[feature_idx][timestep_idx] = (feature_values[timestep_idx] - mean) / (std + 1e-8)
        normalized_data.append(time_series)
    return normalized_data

def load_time_series_data(csv_file_path):
    """
    从 CSV 文件中按列提取时间序列loss。
    Args:
        csv_file_path (str): CSV 文件路径。
    Returns:
        list: 时间序列loss，格式为二级列表，每个子列表表示一个特征列的时间序列。
    """
    try:
        df = pd.read_csv(csv_file_path)
        # 转为列列表形式，每列作为一个特征
        time_series_data = [df[col].tolist() for col in df.columns]
        return time_series_data
    except Exception as e:
        print(f"读取 CSV 文件失败: {csv_file_path}, 错误: {e}")
        return None

def extract_samples(files):
    """
    根据文件列表构建样本loss，每条loss包含五个 CSV 文件和一张图片。
    Args:
        files (list): 当前 gesture_code 下的所有文件路径。
    Returns:
        list: 样本loss列表，每条loss为字典，包含 time_series_data 和 image。
    """
    samples = []
    # 使用正则表达式匹配文件名中的记录编号和手指编号
    record_pattern = re.compile(r"_(\d+)_(\d+)\.csv$")
    image_pattern = re.compile(r"_(\d+)_image\.png$")

    # 将文件按记录编号分组
    groups = {}
    for file in files:
        # 匹配 CSV 文件
        csv_match = record_pattern.search(file)
        if csv_match:
            record_index = int(csv_match.group(1))  # 记录编号（1, 2, 3, ...）
            finger_index = int(csv_match.group(2))  # 手指编号（0, 1, 2, 3, 4）

            if record_index not in groups:
                groups[record_index] = {"csv": [None] * 5, "image": None}

            # 将文件加入相应位置
            groups[record_index]["csv"][finger_index] = file

        # 匹配图片文件
        img_match = image_pattern.search(file)
        if img_match:
            record_index = int(img_match.group(1))  # 记录编号
            if record_index not in groups:
                groups[record_index] = {"csv": [None] * 5, "image": None}
            groups[record_index]["image"] = file

    # 对每个记录编号创建样本
    for record in sorted(groups.keys()):
        csv_files = groups[record]["csv"]
        image_file = groups[record]["image"]

        # 检查loss完整性
        if all(csv_files) and image_file:
            # 加载并归一化时间序列loss
            time_series_data = [load_time_series_data(csv_file) for csv_file in csv_files]
            normalized_time_series = z_score_normalize(time_series_data)
            samples.append({
                "time_series_data": normalized_time_series,
            })
        else:
            print(f"记录 {record} loss不完整，跳过此样本")

    return samples

def load_data(directory):
    """
    加载并组织loss目录中的文件，生成结构化的字典。
    Args:
        directory (str): loss所在目录。
    Returns:
        dict: loss字典，按照 {user_id: {gesture_code: [file1, file2, ...]}} 组织。
    """
    data_dict = defaultdict(lambda: defaultdict(list))

    # 定义正则表达式来匹配文件名
    pattern = re.compile(r"^(?P<user_id>user_id_\d+)_(?P<gesture_code>[a-zA-Z]+\d+)_\d+_(\d+|image)\.(csv|png)$")

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") or file.endswith(".png"):
                # 使用正则表达式匹配并提取信息
                match = pattern.match(file)
                if match:
                    user_id = match.group("user_id")  # 提取 user_id
                    gesture_code = match.group("gesture_code")  # 提取 gesture_code

                    # 将文件路径添加到 data_dict 中
                    data_dict[user_id][gesture_code].append(os.path.join(root, file))
                else:
                    print(f"文件 {file} 的命名不符合预期格式，已跳过。")

    return data_dict

def generate_same_person_same_gesture(data_dict):
    """
    生成同一人同一手势的样本对。
    Args:
        data_dict (dict): 包含loss文件路径的字典。
    Returns:
        list: 样本对列表，每个样本对为字典，包含 sample1 和 sample2。
    """
    sample_pairs = []

    for user_id, gestures in data_dict.items():
        for gesture_code, files in gestures.items():
            # 提取当前 gesture_code 下的三条loss
            samples = extract_samples(files)
            # 在三条loss间生成样本对
            for sample1, sample2 in itertools.combinations(samples, 2):
                sample_pair = {
                    "sample1": sample1,
                    "sample2": sample2
                }
                sample_pairs.append(sample_pair)

    return sample_pairs


def generate_different_person_different_gesture(data_dict, strick=True):
    """
    生成不同人不同手势的样本对。
    Args:
        data_dict (dict): 包含loss文件路径的字典。
    Returns:
        list: 样本对列表，每个样本对为字典，包含 sample1 和 sample2。
    """
    sample_pairs = []

    # 获取所有用户和手势的组合
    user_gesture_combinations = []

    # 遍历每个用户和手势，提取对应样本，并将每个用户的手势样本组合存入列表
    for user_id, gestures in data_dict.items():
        for gesture_code, files in gestures.items():
            samples = extract_samples(files)
            user_gesture_combinations.append((user_id, gesture_code, samples))

    # 使用 itertools.combinations 生成不同用户不同手势的组合
    for (user_id1, gesture_code1, samples1), (user_id2, gesture_code2, samples2) in itertools.combinations(user_gesture_combinations, 2):
        # 提取手势码的字母部分
        if strick:
            gesture_code1 = gesture_code1[0]
            gesture_code2 = gesture_code2[0]
        # 检查是否来自不同用户且手势不同
        if user_id1 != user_id2 and gesture_code1 != gesture_code2:
            # 在两个不同用户、不同手势的样本列表之间生成样本对
            for sample1 in samples1:
                for sample2 in samples2:
                    sample_pair = {
                        "sample1": sample1,
                        "sample2": sample2
                    }
                    sample_pairs.append(sample_pair)

    return sample_pairs

