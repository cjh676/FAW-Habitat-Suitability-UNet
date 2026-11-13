import os
import numpy as np
import h5py  # 使用HDF5格式
import xarray as xr
import gc  # 导入gc模块

# 数据存储目录
data_path = "/root/autodl-tmp/25320/data/"

# 时间范围
years = range(2018, 2024)  # 2018-2023 年

# 保存切分后的数据
output_path = "/root/autodl-tmp/25320/data/split"

os.makedirs(output_path, exist_ok=True)


# 处理每年的数据
def process_year_data(year):
    file_path = os.path.join(data_path, f"deal_data_{year}.nc")
    ds = xr.open_dataset(file_path)

    # 提取环境因子和 NDVI 数据，并指定数据类型为 float32
    environment_data = ds["__xarray_dataarray_variable__"].values.astype(np.float32)  # 替换为实际环境因子变量名
    ndvi_data = ds["__xarray_dataarray_variable__"].sel(variable="NDVI").values.astype(np.float32)  # 替换为实际 NDVI 变量名

    # 确保 NDVI 数据维度一致
    if ndvi_data.ndim == 3:
        ndvi_data = np.expand_dims(ndvi_data, axis=0)  # 变成 (1, band, lat, lon)

    # 合并环境因子数据和 NDVI 数据
    environment_data = np.concatenate([environment_data, ndvi_data], axis=0)

    # 读取新的标签 combined_label
    labels = ds["combined_label"].values.astype(np.int8)
    if labels.ndim == 2:  # 如果标签没有时间维度
        labels = np.broadcast_to(labels, (environment_data.shape[0], *labels.shape))  # [time, lat, lon]
    return environment_data, labels


# 处理数据并保存为 HDF5 格式
def process_and_save_data(years):
    for year in years:
        print(f"Processing year {year}...")
        X_year, y_year = process_year_data(year)
        
        # 按年保存为单独的HDF5文件
        with h5py.File(os.path.join(output_path, f"data_{year}.h5"), "w") as f:
            f.create_dataset("X", data=X_year.astype(np.float32))
            f.create_dataset("y", data=y_year.astype(np.int8))
        
        # 手动调用垃圾回收
        gc.collect()


# 滑动窗口切分数据
def time_series_cross_validation(years, test_size=1):
    for start in range(len(years) - test_size):
        train_indices = slice(start, start + test_size)
        test_indices = slice(start + test_size, start + test_size + 1)

        X_train_all, y_train_all = [], []
        X_test_all, y_test_all = [], []

        # 读取训练数据
        for year in years[train_indices]:
            with h5py.File(os.path.join(output_path, f"data_{year}.h5"), "r") as f:
                X_train_all.append(f["X"][:])
                y_train_all.append(f["y"][:])

        # 读取测试数据
        for year in years[test_indices]:
            with h5py.File(os.path.join(output_path, f"data_{year}.h5"), "r") as f:
                X_test_all.append(f["X"][:])
                y_test_all.append(f["y"][:])

        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
        X_test = np.concatenate(X_test_all, axis=0)
        y_test = np.concatenate(y_test_all, axis=0)

        gc.collect()
        yield X_train, X_test, y_train, y_test


# 处理和切分数据
def process_and_split_data(years):
    for X_train, X_test, y_train, y_test in time_series_cross_validation(years, test_size=1):
        np.save(os.path.join(output_path, "X_train.npy"), X_train.astype(np.float32))
        np.save(os.path.join(output_path, "X_test.npy"), X_test.astype(np.float32))
        np.save(os.path.join(output_path, "y_train.npy"), y_train.astype(np.int8))
        np.save(os.path.join(output_path, "y_test.npy"), y_test.astype(np.int8))
        break


# 主程序
if __name__ == "__main__":
    print("Processing data...")
    process_and_save_data(years)
    print("Splitting and merging data...")
    process_and_split_data(years)
    print("Data processing completed.")
