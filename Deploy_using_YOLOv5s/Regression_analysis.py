import glob
import os
import numpy as np
from datetime import datetime, time as dtime
import time as T

REGRESSION = {
}


# for example:|| 27: (2.116703234653e-06, -1.697350720881e-03, 1.334891931835e+00),
# ------------------ 2. 工具函数 ------------------
def append_to_file(file_path, *items):
    """把任意字段追加到文件，逗号分隔"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(", ".join(str(i) for i in items) + "\n")


def remove_outliers(data, threshold=2):
    """硬阈值 + Z‒score 去离群"""
    data = [x for x in data if x <= 15000]
    if len(data) < 2:
        return data
    mean, std = np.mean(data), np.std(data)
    if std == 0:
        return data
    return [x for x in data if abs((x - mean) / std) < threshold]


def calculate_second_column_average(file_path):
    """
    读取 txt，取满足条件的第二列非零值的平均数。
    条件：第一列 > 4 才采用该行数据。
    """
    values = []
    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = [p.strip() for p in ln.strip().split(",")]
            if len(parts) >= 2:
                try:
                    first_val = float(parts[0])  # 第一列
                    second_val = float(parts[1])  # 第二列
                except ValueError:
                    # 列表里有非数字，直接忽略此行
                    continue

                # 第一列 ≤ 4 则跳过
                if first_val <= 4:
                    continue

                # 第二列非零才记录
                if second_val != 0:
                    values.append(second_val)

    if not values:
        return None
    return np.mean(remove_outliers(values))


def pfit(x, file_num):
    """
    二次拟合：根据 txt 文件名数字 -> (file_num − 27) -> REGRESSION，
    计算 y = a2·x^2 + a1·x + a0。
    """
    if x is None:
        return None
    key = file_num - 27
    coeff = REGRESSION.get(key)
    if not coeff:
        return None
    a2, a1, a0 = coeff
    return a2 * x * x + a1 * x + a0


# ------------------ 3. 主循环 ------------------
while True:
    now = datetime.now()
    start_window = dtime(16, 5, 0)
    end_window = dtime(16, 7, 0)
    weight_path = "./temp_data/weight.txt"

    if start_window <= now.time() < end_window:
        date_str = now.strftime("%Y-%m-%d")
        day_dir = os.path.join("./temp_data", date_str)

        # 在新日期开始前写一行分隔符
        append_to_file(weight_path, "#", date_str)

        for txtpath in glob.glob(os.path.join(day_dir, "*.txt")):
            avg_area = calculate_second_column_average(txtpath)

            fname = os.path.splitext(os.path.basename(txtpath))[0]
            try:
                file_num = int(fname)
            except ValueError:
                continue  # 非数字文件名跳过

            weight = pfit(avg_area, file_num)
            append_to_file(weight_path, file_num, weight, weight)

        T.sleep(121)  # 窗口内只跑一次
    else:
        T.sleep(1)  # 非窗口时间轻量休眠
