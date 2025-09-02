#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import os
from collections import Counter, deque
import warnings
import random

warnings.filterwarnings('ignore')


class OptimizedRLPCA:
    """优化的RL-PCA实现，解决震荡和训练时间问题"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # PCA结果缓存
        self.pca_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'smoothed_rewards': [],
            'best_p_history': [],
            'ssim_history': [],
            'q_value_std': [],
            'action_distribution': [],
            'convergence_metric': []
        }

    def calculate_reward_smooth(self, ssim_value, reconstruction_error, compression_ratio, p, max_p):
        """平滑的多目标奖励函数，避免震荡"""
        # 1. SSIM奖励 - 使用sigmoid平滑过渡
        target_ssim = 0.96
        ssim_score = 1 / (1 + np.exp(-20 * (ssim_value - target_ssim)))  # sigmoid
        ssim_reward = 20 * ssim_score

        # 2. 重建误差奖励 - 使用指数衰减
        error_score = np.exp(-reconstruction_error * 1000)
        error_reward = 10 * error_score

        # 3. 压缩比奖励 - 使用高斯函数
        optimal_ratio = 5.0
        ratio_score = np.exp(-((compression_ratio - optimal_ratio) ** 2) / 8)
        # ratio_reward = 10 * ratio_score
        ratio_reward = 15 * ratio_score

        # 4. 效率奖励 - 鼓励使用较少的主成分
        efficiency_score = 1 - (p / max_p) ** 2
        # efficiency_reward = 5 * efficiency_score
        efficiency_reward = 10 * efficiency_score

        # 5. 总奖励（加权和）
        total_reward = (
                0.3 * ssim_reward +  # SSIM最重要
                0.1 * error_reward +  # 重建误差
                0.4 * ratio_reward +  # 压缩比
                0.2 * efficiency_reward  # 效率
        )

        # 添加小的噪声避免完全相同的奖励
        total_reward += np.random.normal(0, 0.01)

        return total_reward

    def pca_compress_cached(self, X, k):
        """带缓存的PCA压缩，大幅提升速度"""
        # 生成缓存键
        cache_key = f"{X.shape}_{k}_{X.sum().item():.4f}"

        # 检查缓存
        if cache_key in self.pca_cache:
            self.cache_hits += 1
            return self.pca_cache[cache_key]

        self.cache_misses += 1

        # 开始计时压缩时间
        compress_start = time.time()

        # 执行PCA压缩
        n, m = X.shape
        k = min(k, n, m)

        # 标准化
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X_std = torch.std(X, dim=0, keepdim=True)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        X_centered = X_normalized
        # 数据中心化
        # X_mean = torch.mean(X, dim=0, keepdim=True)
        # X_centered = X - X_mean

        try:
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
        except:
            # 备用方法
            components = torch.randn(m, k, device=self.device)
            components, _ = torch.qr(components)

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
        reconstruction_error = torch.norm(X - X_approx, p='fro').pow(2) / (torch.norm(X, p='fro').pow(2) + 1e-8)

        original_size = n * m
        compressed_size = k * (n + m)
        compression_ratio = max(1.0, original_size / compressed_size)

        # 缓存结果
        result = (X_approx, reconstruction_error.item(), compression_ratio, compress_time, decompress_time)

        # 限制缓存大小
        if len(self.pca_cache) > 100:
            # 删除最早的项
            first_key = next(iter(self.pca_cache))
            del self.pca_cache[first_key]

        self.pca_cache[cache_key] = result
        return result

    def train(self, I_tensor, max_episodes=200, verbose=True):
        """优化的训练过程"""
        max_p = min(I_tensor.shape)

        # 初始化Q表 - 使用更好的初始值
        Q = torch.ones((max_p, 3), device=self.device) * 0.1

        # 预计算一些状态以初始化Q表
        if verbose:
            print("Precomputing key states for initialization...")

        sample_ps = [1, max_p // 4, max_p // 2, 3 * max_p // 4, max_p]
        for p in sample_ps:
            if 1 <= p <= max_p:
                try:
                    X_approx, error, ratio = self.pca_compress_cached(I_tensor, p)
                    ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())
                    reward = self.calculate_reward_smooth(ssim_val, error, ratio, p, max_p)
                    # 初始化Q值
                    Q[p - 1, :] = reward * 0.5
                except:
                    pass

        # 训练参数
        alpha_start = 0.3
        alpha_end = 0.01
        alpha_decay = 0.995
        alpha = alpha_start

        # 自适应epsilon
        epsilon_start = 0.9
        epsilon_end = 0.05
        epsilon_decay = 0.98
        epsilon = epsilon_start

        gamma = 0.95

        # 奖励平滑
        reward_smoother = deque(maxlen=5)

        # 批量更新
        update_batch = []
        batch_size = 5

        # 主训练循环
        for episode in range(max_episodes):
            # 智能初始化
            if episode < 20:
                current_p = np.random.randint(1, max_p + 1)
            else:
                # 从Q值高的状态开始
                q_sums = torch.sum(Q, dim=1)
                probs = torch.softmax(q_sums * 2, dim=0).cpu().numpy()
                current_p = np.random.choice(range(1, max_p + 1), p=probs)

            episode_rewards = []
            episode_actions = []
            states_visited = []

            # Episode内循环
            for step in range(10):  # 减少步数加快训练
                # Epsilon-greedy with softmax探索
                if np.random.rand() < epsilon:
                    if np.random.rand() < 0.7:  # 70%概率随机
                        action = np.random.randint(0, 3)
                    else:  # 30%概率用softmax
                        q_values = Q[current_p - 1]
                        probs = torch.softmax(q_values * 5, dim=0).cpu().numpy()
                        action = np.random.choice(3, p=probs)
                else:
                    action = torch.argmax(Q[current_p - 1]).item()

                # 执行动作
                if action == 0 and current_p < max_p:
                    next_p = current_p + 1
                elif action == 1 and current_p > 1:
                    next_p = current_p - 1
                else:
                    next_p = current_p

                # PCA压缩（使用缓存）
                X_approx, recon_error, comp_ratio, compress_time, decompress_time = self.pca_compress_cached(I_tensor, next_p)

                # 计算SSIM
                ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())

                # 计算奖励
                reward = self.calculate_reward_smooth(ssim_val, recon_error, comp_ratio, next_p, max_p)
                episode_rewards.append(reward)
                episode_actions.append(action)
                states_visited.append((current_p, action, reward, next_p))

                # 批量更新Q值
                update_batch.append((current_p, action, reward, next_p))

                if len(update_batch) >= batch_size:
                    # 执行批量更新
                    for s, a, r, s_next in update_batch:
                        if s != s_next:
                            old_q = Q[s - 1, a]
                            max_next_q = torch.max(Q[s_next - 1])
                            new_q = old_q + alpha * (r + gamma * max_next_q - old_q)
                            Q[s - 1, a] = new_q
                    update_batch = []

                current_p = next_p

            # 清空剩余的批量更新
            for s, a, r, s_next in update_batch:
                if s != s_next:
                    old_q = Q[s - 1, a]
                    max_next_q = torch.max(Q[s_next - 1])
                    new_q = old_q + alpha * (r + gamma * max_next_q - old_q)
                    Q[s - 1, a] = new_q
            update_batch = []

            # Episode统计
            avg_reward = np.mean(episode_rewards)
            reward_smoother.append(avg_reward)
            smoothed_reward = np.mean(reward_smoother)

            # 找出最佳状态
            best_idx = np.argmax(episode_rewards)
            best_state = states_visited[best_idx] if states_visited else (current_p, 0, 0, current_p)
            best_p = best_state[3]

            # 记录历史
            self.training_history['episode_rewards'].append(avg_reward)
            self.training_history['smoothed_rewards'].append(smoothed_reward)
            self.training_history['best_p_history'].append(best_p)
            self.training_history['ssim_history'].append(ssim_val)
            self.training_history['q_value_std'].append(torch.std(Q).item())

            # 动作分布
            action_counts = [episode_actions.count(i) for i in range(3)]
            self.training_history['action_distribution'].append(action_counts)

            # 收敛度量
            if len(self.training_history['smoothed_rewards']) > 10:
                recent_std = np.std(self.training_history['smoothed_rewards'][-10:])
                self.training_history['convergence_metric'].append(recent_std)

            # 参数衰减
            alpha = max(alpha_end, alpha * alpha_decay)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # 打印进度
            if verbose and episode % 2 == 0:
                cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses + 1) * 100
                print(f"Episode {episode}: Reward={avg_reward:.2f} (smoothed={smoothed_reward:.2f}), "
                      f"Best p={best_p}, ε={epsilon:.3f}, α={alpha:.3f}, Cache={cache_rate:.1f}%")


        # 确定最佳p值
        # 方法1：使用最后20%的episodes中最频繁的p
        recent_ps = self.training_history['best_p_history'][
                    -max(10, len(self.training_history['best_p_history']) // 5):]
        p_counter = Counter(recent_ps)
        best_p = p_counter.most_common(1)[0][0]

        if verbose:
            print(f"\nTraining completed!")
            print(f"Cache hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses + 1) * 100:.1f}%")
            print(f"Best p: {best_p}")

        return best_p, Q, self.training_history


def load_mat_files(folder_path):
    """加载.mat文件"""
    if not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist")
        return np.array([])

    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    data_list = []

    for mat_file in mat_files:  # 限制数量
        file_path = os.path.join(folder_path, mat_file)
        try:
            mat_data = loadmat(file_path)
            for key in ['data', 'Data', 'matrix', 'image', 'img']:
                if key in mat_data:
                    data_list.append(mat_data[key])
                    break
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")

    return np.array(data_list) if data_list else np.array([])


def calculate_ssim(original, reconstructed):
    """计算SSIM"""
    data_range = original.max() - original.min()
    if data_range == 0:
        return 1.0
    return ssim(original, reconstructed, data_range=data_range)


def evaluate_compression(I_tensor, p, rl_pca):
    """评估压缩性能"""
    start_time = time.time()
    X_approx, reconstruction_error, compression_ratio , compress_time, decompress_time= rl_pca.pca_compress_cached(I_tensor, p)
    compression_time = time.time() - start_time

    # PSNR
    mse = torch.mean((I_tensor - X_approx) ** 2)
    max_val = I_tensor.max()
    min_val = I_tensor.min()
    if max_val > min_val:
        psnr = 10 * torch.log10((max_val - min_val) ** 2 / mse)
    else:
        psnr = torch.tensor(0.0)

    # SSIM
    ssim_value = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())

    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'ssim': ssim_value,
        'compression_ratio': compression_ratio,
        'reconstruction_error': reconstruction_error,
        'compression_time': compression_time,
        'compress_time': compress_time,
        'decompress_time': decompress_time
    }

def visualize_comprehensive_training(history, Q_table, save_prefix=''):
    """全面的训练可视化"""
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 12  # 设置字体大小
    plt.rcParams['xtick.direction'] = 'in'  # 刻度线向内
    plt.rcParams['ytick.direction'] = 'in'  # 刻度线向内

    fig= plt.figure()
    # 奖励曲线（原始和平滑）
    episodes = range(len(history['episode_rewards']))
    plt.plot(episodes, history['episode_rewards'],  linestyle='-.', color='#FD763F', label='Raw rewards')
    plt.plot(episodes, history['smoothed_rewards'], linestyle='--', color='#23BAC5', label='Smoothed rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(frameon=False)

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)  # 为底部标签留出空间
    if save_prefix:
        # plt.savefig(f'{save_prefix}_training_analysis.pdf', dpi=300, format='pdf', bbox_inches='tight', transparent=True)
        # print(f"Training analysis saved to: {save_prefix}_training_analysis.pdf")
        plt.savefig(f'{save_prefix}_training_analysis.png', dpi=300,  bbox_inches='tight',
                    )
        print(f"Training analysis saved to: {save_prefix}_training_analysis.png")
    else:
        plt.show()

    # 打印统计信息
    print("\n=== Training Statistics ===")
    print(f"Total episodes: {len(history['episode_rewards'])}")
    print(f"Final average reward: {history['smoothed_rewards'][-1]:.3f}")
    print(f"Reward std (last 20 eps): {np.std(history['episode_rewards'][-20:]):.3f}")
    print(f"Most common p values: {Counter(history['best_p_history']).most_common(3)}")

    # 震荡分析
    if len(history['episode_rewards']) > 20:
        early_std = np.std(history['episode_rewards'][:20])
        late_std = np.std(history['episode_rewards'][-20:])
        print(f"Oscillation reduction: {(1 - late_std / early_std) * 100:.1f}%")


def plot_performance_analysis(train_results, test_results, avg_best_p, save_prefix=''):
    """性能分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. PSNR vs Compression Ratio
    ax1 = axes[0, 0]
    train_psnrs = [r['psnr'] for r in train_results]
    train_crs = [r['compression_ratio'] for r in train_results]
    test_psnrs = [r['psnr'] for r in test_results]
    test_crs = [r['compression_ratio'] for r in test_results]

    ax1.scatter(train_crs, train_psnrs, c='blue', alpha=0.6, label='Train', s=50)
    ax1.scatter(test_crs, test_psnrs, c='red', alpha=0.6, label='Test', s=50)
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Compression Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. SSIM vs Compression Ratio
    ax2 = axes[0, 1]
    train_ssims = [r['ssim'] for r in train_results]
    test_ssims = [r['ssim'] for r in test_results]

    ax2.scatter(train_crs, train_ssims, c='blue', alpha=0.6, label='Train', s=50)
    ax2.scatter(test_crs, test_ssims, c='red', alpha=0.6, label='Test', s=50)
    ax2.axhline(y=0.96, color='green', linestyle='--', label='Target SSIM')
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs Compression Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 性能指标箱线图
    ax3 = axes[1, 0]
    metrics_data = [train_psnrs, test_psnrs, train_ssims, test_ssims]
    labels = ['Train PSNR', 'Test PSNR', 'Train SSIM', 'Test SSIM']
    positions = [1, 2, 4, 5]

    bp = ax3.boxplot(metrics_data, positions=positions, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax3.set_ylabel('Value')
    ax3.set_title('Performance Metrics Distribution')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 压缩时间分析
    ax4 = axes[1, 1]
    train_times = [r['compression_time'] for r in train_results]
    test_times = [r['compression_time'] for r in test_results]

    ax4.hist(train_times, bins=20, alpha=0.5, label='Train', color='blue')
    ax4.hist(test_times, bins=20, alpha=0.5, label='Test', color='red')
    ax4.set_xlabel('Compression Time (s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Compression Time Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_prefix:
        plt.savefig(f'{save_prefix}_performance_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Performance analysis saved to: {save_prefix}_performance_analysis.png")
    else:
        plt.show()


# 主程序
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Record program start time
    program_start_time = time.time()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    mat_folder = './solarha'  # 修改为你的路径
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
    all_histories = []
    all_Qs = []

    # 限制训练样本数以加快速度
    num_train_samples = min(60, len(train_data))
    # num_train_samples = max(1, len(train_data))

    start_time = time.time()
    # num_train_samples = 1
    for i, sample in enumerate(train_data[:num_train_samples]):
        print(f'\n{"=" * 60}')
        print(f'Training on sample {i + 1}/{num_train_samples}')
        print(f'{"=" * 60}')

        I_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
        print(f'Sample shape: {I_tensor.shape}')

        # 创建模型实例
        rl_pca = OptimizedRLPCA(device)

        # 训练
        best_p, Q, history = rl_pca.train(I_tensor, max_episodes=60, verbose=True)
        best_ps.append(best_p)
        all_histories.append(history)
        all_Qs.append(Q)

        # 评估
        train_metrics = evaluate_compression(I_tensor, best_p, rl_pca)
        train_results.append(train_metrics)

        print(f'\nSample {i + 1} Results:')
        print(f'  Best p: {best_p}')
        print(f'  PSNR: {train_metrics["psnr"]:.2f} dB')
        print(f'  SSIM: {train_metrics["ssim"]:.4f}')
        print(f'  compress time: {train_metrics["compress_time"]*1000:.4f}'+'ms')
        print(f'  decompress time: {train_metrics["decompress_time"]*1000:.4f}'+'ms')
        print(f'  Compression Ratio: {train_metrics["compression_ratio"]:.4f}')

        # 为第一个样本画详细图
        if i == 0:
            visualize_comprehensive_training(history, Q, save_prefix='sample1')
    train_time = time.time() - start_time
    # 平均最佳p值
    # avg_best_p = int(np.mean(best_ps))
    avg_best_p = int(np.median(best_ps))
    print(f'\n{"=" * 60}')
    print(f'Average best p across training samples: {avg_best_p}')
    print(f'{"=" * 60}')

    # 测试集评估
    test_results = []
    test_rl_pca = OptimizedRLPCA(device)  # 新实例用于测试

    num_test_samples = min(40, len(test_data))
    # num_test_samples = max(1, len(test_data))

    for i, sample in enumerate(test_data[:num_test_samples]):
        print(f'\nEvaluating test sample {i + 1}/{num_test_samples}')
        I_tensor = torch.tensor(sample, dtype=torch.float32).to(device)

        test_metrics = evaluate_compression(I_tensor, avg_best_p, test_rl_pca)
        test_results.append(test_metrics)

        print(f'  PSNR: {test_metrics["psnr"]:.2f} dB')
        print(f'  SSIM: {test_metrics["ssim"]:.4f}')
        print(f'  compress time: {test_metrics["compress_time"]*1000:.4f}'+'ms')
        print(f'  decompress time: {test_metrics["decompress_time"]*1000:.4f}'+'ms')
        print(f'  Compression Ratio: {test_metrics["compression_ratio"]:.4f}')

    # 画性能分析图
    if train_results and test_results:
        plot_performance_analysis(train_results, test_results, avg_best_p, save_prefix='final')

    # 最终统计
    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)

    # 训练集统计
    train_psnrs = [r['psnr'] for r in train_results]
    train_ssims = [r['ssim'] for r in train_results]
    train_crs = [r['compression_ratio'] for r in train_results]

    print(f'\nTraining set averages:')
    print(f'  PSNR: {np.mean(train_psnrs):.2f}dB')
    print(f'  SSIM: {np.mean(train_ssims):.4f}')
    print(f'  Compression Ratio: {np.mean(train_crs):.4f} ')
    print(f'\nTrain time: {train_time:.3g} seconds ({train_time / 60:.3g} minutes)')
    # 测试集统计
    test_psnrs = [r['psnr'] for r in test_results]
    test_ssims = [r['ssim'] for r in test_results]
    test_crs = [r['compression_ratio'] for r in test_results]

    print(f'\nTest set averages:')
    print(f'  PSNR: {np.mean(test_psnrs):.2f} dB')
    print(f'  SSIM: {np.mean(test_ssims):.4f}')
    print(f'  Compression Ratio: {np.mean(test_crs):.4f}')

    # Total program time
    total_time = time.time() - program_start_time
    print(f'\nTotal program runtime: {total_time:.3g} seconds ({total_time / 60:.3g} minutes)')
    print('\n' + '=' * 60)
    print('Training completed successfully!')
    print('=' * 60)