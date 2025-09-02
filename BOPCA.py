import torch
import numpy as np
import os
from scipy.io import loadmat
import time
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

def pca_compress(X, k):
    """
    执行PCA压缩
    :param X: 输入数据
    :param k: 主成分数量
    :return: 压缩后的数据，重建误差，压缩比
    """
    n, m = X.shape
    k = min(k, n, m)

    # 开始计时压缩时间
    compress_start = time.time()
    # 数据中心化
    # X_mean = torch.mean(X, dim=0, keepdim=True)
    # X_centered = X - X_mean
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X_std = torch.std(X, dim=0, keepdim=True)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    X_centered = X_normalized

    # 对于小矩阵使用完整的特征分解
    if min(n, m) <= 50:
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (n - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        components = eigenvectors[:, :k]
    else:
        # 对于大矩阵使用随机SVD
        U, S, V = torch.svd_lowrank(X_centered, q=min(k + 10, min(n, m)))
        components = V[:, :k]

    # 投影和重建
    projected = torch.matmul(X_centered, components)
    compress_time = time.time() - compress_start

    # 开始计时解压缩时间
    decompress_start = time.time()
    X_approx_normalized = torch.matmul(projected, components.T)
    X_approx = X_approx_normalized * X_std + X_mean
    # X_approx = torch.matmul(projected, components.T) + X_mean
    decompress_time = time.time() - decompress_start

    # 计算指标
    reconstruction_error = torch.norm(X - X_approx, p='fro').pow(2) / (torch.norm(X, p='fro').pow(2))
    # Compression ratio calculation
    original_size = n * m
    compressed_size = k * (n + m)
    compression_ratio = max(1.0, original_size / compressed_size)

    return X_approx, reconstruction_error.item(), compression_ratio,compress_time, decompress_time


def load_mat_files(folder_path):
    """Load .mat files"""
    if not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist")
        return []

    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    print(f"Found {len(mat_files)} .mat files")

    data_list = []

    for i, mat_file in enumerate(mat_files):
        file_path = os.path.join(folder_path, mat_file)
        try:
            mat_data = loadmat(file_path)
            found_data = False
            for key in ['data', 'Data', 'matrix', 'image', 'img']:
                if key in mat_data:
                    data = mat_data[key]
                    if len(data.shape) == 2:
                        data_list.append(data)
                        found_data = True
                        print(f"  Successfully loaded {mat_file} (key='{key}', shape={data.shape})")
                    break
            if not found_data:
                print(f"  No suitable data found in {mat_file}")
        except Exception as e:
            print(f"  Error loading {mat_file}: {e}")

    print(f"Successfully loaded {len(data_list)} data matrices")
    return data_list


def calculate_ssim(original, reconstructed):
    """Calculate SSIM (Structural Similarity Index)"""
    data_range = original.max() - original.min()
    if data_range == 0:
        return 1.0
    return ssim(original, reconstructed, data_range=data_range)

def objective_function(p):
    """
    贝叶斯优化的目标函数
    :param p: 主成分数量
    :return: 压缩比（当SSIM大于0.96时）
    """
    p = int(round(p))  # 将p转换为整数
    X_approx, error, ratio, compress_time, decompress_time = pca_compress(I_tensor, p)
    ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())

    # 如果SSIM小于0.96，则返回一个大的负数，表示不希望这个解
    if ssim_val < 0.96:
        return -1000
    elif ratio > 6:
        return -1000

    return ratio

# Record program start time
program_start_time = time.time()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print('=' * 60)


# 加载数据
mat_folder = ('./solarha')  # 修改为你的路径
all_data = load_mat_files(mat_folder)
if len(all_data) == 0:
    print("No data loaded. Please check the data folder path.")
    exit(1)

# 数据集划分
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
print(f'Loaded {len(all_data)} samples, {len(train_data)} for training, {len(test_data)} for testing')

# 训练
train_results = []
best_ps = []

start_time = time.time()
# 限制训练样本数以加快速度
num_train_samples = min(60, len(train_data))
# num_train_samples = max(1, len(train_data))
for i, sample in enumerate(train_data[:num_train_samples]):
    print(f'\n{"=" * 60}')
    print(f'Training on sample {i + 1}/{num_train_samples}')
    print(f'{"=" * 60}')
    I_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
    print(f'Sample shape: {I_tensor.shape}')

    # 定义贝叶斯优化器
    pbounds = {'p': (1, min(I_tensor.shape))}  # p的范围是1到样本的最小维度
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # 运行优化
    optimizer.maximize(
        init_points=10,
        n_iter=50,
    )

    # 输出最优解
    best_p = int(round(optimizer.max['params']['p']))
    best_ps.append(best_p)
    print(f"Best p: {best_p}")

    # 使用最优的p进行压缩和重构
    X_approx, error, ratio, compress_time, decompress_time = pca_compress(I_tensor, best_p)
    ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())

    # PSNR
    mse = torch.mean((I_tensor - X_approx) ** 2)
    max_val = I_tensor.max()
    min_val = I_tensor.min()
    if max_val > min_val:
        psnr = 10 * torch.log10((max_val - min_val) ** 2 / mse)
    else:
        psnr = torch.tensor(0.0)

    train_metrics = {
        'best_p': best_p,
        'compression_ratio': ratio,
        'ssim': ssim_val,
        'psnr': psnr.item(),
        'compress_time': compress_time,
        'decompress_time': decompress_time
    }
    train_results.append(train_metrics)
    print(f"Compression ratio: {ratio}")
    print(f"SSIM: {ssim_val}")
    print(f"PSNR: {psnr.item()} dB")
    print(f"Compression time: {compress_time} s")
    print(f"Decompression time: {decompress_time} s")
train_time = time.time() - start_time

# 测试集评估
test_results = []
# avg_best_p = int(np.mean(best_ps))
avg_best_p = int(np.median(best_ps))

# 限制测试样本数以加快速度
num_test_samples = min(40, len(test_data))
# num_test_samples = max(1, len(test_data))
for i, sample in enumerate(test_data[:num_test_samples]):
    print(f'\nEvaluating test sample {i + 1}/{num_test_samples}')
    I_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
    X_approx, error, ratio, compress_time, decompress_time  = pca_compress(I_tensor, avg_best_p)
    ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())
    # PSNR
    mse = torch.mean((I_tensor - X_approx) ** 2)
    max_val = I_tensor.max()
    min_val = I_tensor.min()
    if max_val > min_val:
        psnr = 10 * torch.log10((max_val - min_val) ** 2 / mse)
    else:
        psnr = torch.tensor(0.0)

    test_metrics = {
        'compression_ratio': ratio,
        'ssim': ssim_val,
        'psnr': psnr.item(),
        'compress_time': compress_time,
        'decompress_time': decompress_time
    }
    test_results.append(test_metrics)
    print(f"Compression ratio: {ratio}")
    print(f"SSIM: {ssim_val}")
    print(f"Compression ratio: {ratio}")
    print(f"SSIM: {ssim_val}")
    print(f"PSNR: {psnr.item()} dB")
    print(f"Compression time: {compress_time} s")
    print(f"Decompression time: {decompress_time} s")


# 最终统计
print('\n' + '=' * 60)
print('FINAL RESULTS')
print('=' * 60)
# 训练集统计
train_ratios = [r['compression_ratio'] for r in train_results]
train_ssims = [r['ssim'] for r in train_results]
train_psnrs = [r['psnr'] for r in train_results]
train_compress_times = [r['compress_time'] for r in train_results]
train_decompress_times = [r['decompress_time'] for r in train_results]
print(f'\nTraining set averages:')
print(f'  Compression Ratio: {np.mean(train_ratios):.4f} ± {np.std(train_ratios):.2f}')
print(f'  SSIM: {np.mean(train_ssims):.4f} ± {np.std(train_ssims):.4f}')
print(f'  PSNR: {np.mean(train_psnrs):.2f} ± {np.std(train_psnrs):.2f} dB')
print(f'  Compression Time: {np.mean(train_compress_times) * 1000:.4f} ms')
print(f'  Decompression Time: {np.mean(train_decompress_times) * 1000:.4f} m')
print(f'\nTrain time: {train_time:.3g} seconds ({train_time / 60:.3g} minutes)')

# 测试集统计
test_ratios = [r['compression_ratio'] for r in test_results]
test_ssims = [r['ssim'] for r in test_results]
test_psnrs = [r['psnr'] for r in test_results]
test_compress_times = [r['compress_time'] for r in test_results]
test_decompress_times = [r['decompress_time'] for r in test_results]
print(f'\nTest set averages:')
print(f'  Compression Ratio: {np.mean(test_ratios):.4f} ± {np.std(test_ratios):.2f}')
print(f'  SSIM: {np.mean(test_ssims):.4f} ± {np.std(test_ssims):.4f}')
print(f'  PSNR: {np.mean(test_psnrs):.2f} ± {np.std(test_psnrs):.2f} dB')
print(f'  Compression Time: {np.mean(test_compress_times)*1000:.4f} ms')
print(f'  Decompression Time: {np.mean(test_decompress_times)*1000:.4f} ms')
print('\n' + '=' * 60)
print('Training and evaluation completed successfully!')
print('=' * 60)

# Total program time
total_time = time.time() - program_start_time
print(f'\nTotal program runtime: {total_time:.3g} seconds ({total_time / 60:.3g} minutes)')
print('=' * 60)
print('Bayesian optimization completed!')