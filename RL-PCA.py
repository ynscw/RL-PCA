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
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pca_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

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
        target_ssim = 0.96
        ssim_score = 1 / (1 + np.exp(-20 * (ssim_value - target_ssim)))  # sigmoid
        ssim_reward = 20 * ssim_score

        error_score = np.exp(-reconstruction_error * 1000)
        error_reward = 10 * error_score

        optimal_ratio = 5.0
        ratio_score = np.exp(-((compression_ratio - optimal_ratio) ** 2) / 8)
        ratio_reward = 15 * ratio_score

        efficiency_score = 1 - (p / max_p) ** 2
        efficiency_reward = 10 * efficiency_score

        total_reward = (
                0.3 * ssim_reward +  
                0.1 * error_reward +  
                0.4 * ratio_reward +  
                0.2 * efficiency_reward  
        )

        total_reward += np.random.normal(0, 0.01)

        return total_reward

    def pca_compress_cached(self, X, k):
        cache_key = f"{X.shape}_{k}_{X.sum().item():.4f}"

        if cache_key in self.pca_cache:
            self.cache_hits += 1
            return self.pca_cache[cache_key]

        self.cache_misses += 1

        compress_start = time.time()

        n, m = X.shape
        k = min(k, n, m)

        X_mean = torch.mean(X, dim=0, keepdim=True)
        X_std = torch.std(X, dim=0, keepdim=True)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        X_centered = X_normalized

        try:
            if min(n, m) <= 50:
                cov_matrix = torch.matmul(X_centered.T, X_centered) / (n - 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
                sorted_indices = torch.argsort(eigenvalues, descending=True)
                eigenvectors = eigenvectors[:, sorted_indices]
                components = eigenvectors[:, :k]
            else:
                U, S, V = torch.svd_lowrank(X_centered, q=min(k + 10, min(n, m)))
                components = V[:, :k]
        except:
            components = torch.randn(m, k, device=self.device)
            components, _ = torch.qr(components)

        projected = torch.matmul(X_centered, components)
        compress_time = time.time() - compress_start

        decompress_start = time.time()
        X_approx_normalized = torch.matmul(projected, components.T)
        X_approx = X_approx_normalized * X_std + X_mean
        decompress_time = time.time() - decompress_start

        reconstruction_error = torch.norm(X - X_approx, p='fro').pow(2) / (torch.norm(X, p='fro').pow(2) + 1e-8)

        original_size = n * m
        compressed_size = k * (n + m)
        compression_ratio = max(1.0, original_size / compressed_size)

        result = (X_approx, reconstruction_error.item(), compression_ratio, compress_time, decompress_time)

        if len(self.pca_cache) > 100:
            first_key = next(iter(self.pca_cache))
            del self.pca_cache[first_key]

        self.pca_cache[cache_key] = result
        return result

    def train(self, I_tensor, max_episodes=200, verbose=True):
        max_p = min(I_tensor.shape)

        Q = torch.ones((max_p, 3), device=self.device) * 0.1

        if verbose:
            print("Precomputing key states for initialization...")

        sample_ps = [1, max_p // 4, max_p // 2, 3 * max_p // 4, max_p]
        for p in sample_ps:
            if 1 <= p <= max_p:
                try:
                    X_approx, error, ratio = self.pca_compress_cached(I_tensor, p)
                    ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())
                    reward = self.calculate_reward_smooth(ssim_val, error, ratio, p, max_p)
                    Q[p - 1, :] = reward * 0.5
                except:
                    pass

        alpha_start = 0.3
        alpha_end = 0.01
        alpha_decay = 0.995
        alpha = alpha_start

        epsilon_start = 0.9
        epsilon_end = 0.05
        epsilon_decay = 0.98
        epsilon = epsilon_start

        gamma = 0.95
        reward_smoother = deque(maxlen=5)

        update_batch = []
        batch_size = 5

        for episode in range(max_episodes):
            if episode < 20:
                current_p = np.random.randint(1, max_p + 1)
            else:
                q_sums = torch.sum(Q, dim=1)
                probs = torch.softmax(q_sums * 2, dim=0).cpu().numpy()
                current_p = np.random.choice(range(1, max_p + 1), p=probs)

            episode_rewards = []
            episode_actions = []
            states_visited = []

            for step in range(10): 
                if np.random.rand() < epsilon:
                    if np.random.rand() < 0.7: 
                        action = np.random.randint(0, 3)
                    else: 
                        q_values = Q[current_p - 1]
                        probs = torch.softmax(q_values * 5, dim=0).cpu().numpy()
                        action = np.random.choice(3, p=probs)
                else:
                    action = torch.argmax(Q[current_p - 1]).item()

                if action == 0 and current_p < max_p:
                    next_p = current_p + 1
                elif action == 1 and current_p > 1:
                    next_p = current_p - 1
                else:
                    next_p = current_p

                X_approx, recon_error, comp_ratio, compress_time, decompress_time = self.pca_compress_cached(I_tensor, next_p)

                ssim_val = calculate_ssim(I_tensor.cpu().numpy(), X_approx.cpu().numpy())

                reward = self.calculate_reward_smooth(ssim_val, recon_error, comp_ratio, next_p, max_p)
                episode_rewards.append(reward)
                episode_actions.append(action)
                states_visited.append((current_p, action, reward, next_p))
                update_batch.append((current_p, action, reward, next_p))

                if len(update_batch) >= batch_size:
                    for s, a, r, s_next in update_batch:
                        if s != s_next:
                            old_q = Q[s - 1, a]
                            max_next_q = torch.max(Q[s_next - 1])
                            new_q = old_q + alpha * (r + gamma * max_next_q - old_q)
                            Q[s - 1, a] = new_q
                    update_batch = []

                current_p = next_p

            for s, a, r, s_next in update_batch:
                if s != s_next:
                    old_q = Q[s - 1, a]
                    max_next_q = torch.max(Q[s_next - 1])
                    new_q = old_q + alpha * (r + gamma * max_next_q - old_q)
                    Q[s - 1, a] = new_q
            update_batch = []

            avg_reward = np.mean(episode_rewards)
            reward_smoother.append(avg_reward)
            smoothed_reward = np.mean(reward_smoother)

            best_idx = np.argmax(episode_rewards)
            best_state = states_visited[best_idx] if states_visited else (current_p, 0, 0, current_p)
            best_p = best_state[3]

            self.training_history['episode_rewards'].append(avg_reward)
            self.training_history['smoothed_rewards'].append(smoothed_reward)
            self.training_history['best_p_history'].append(best_p)
            self.training_history['ssim_history'].append(ssim_val)
            self.training_history['q_value_std'].append(torch.std(Q).item())
            action_counts = [episode_actions.count(i) for i in range(3)]
            self.training_history['action_distribution'].append(action_counts)

            if len(self.training_history['smoothed_rewards']) > 10:
                recent_std = np.std(self.training_history['smoothed_rewards'][-10:])
                self.training_history['convergence_metric'].append(recent_std)

            alpha = max(alpha_end, alpha * alpha_decay)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if verbose and episode % 2 == 0:
                cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses + 1) * 100
                print(f"Episode {episode}: Reward={avg_reward:.2f} (smoothed={smoothed_reward:.2f}), "
                      f"Best p={best_p}, ε={epsilon:.3f}, α={alpha:.3f}, Cache={cache_rate:.1f}%")

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
    if not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist")
        return np.array([])

    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    data_list = []

    for mat_file in mat_files:
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
    data_range = original.max() - original.min()
    if data_range == 0:
        return 1.0
    return ssim(original, reconstructed, data_range=data_range)


def evaluate_compression(I_tensor, p, rl_pca):
    start_time = time.time()
    X_approx, reconstruction_error, compression_ratio , compress_time, decompress_time= rl_pca.pca_compress_cached(I_tensor, p)
    compression_time = time.time() - start_time

    mse = torch.mean((I_tensor - X_approx) ** 2)
    max_val = I_tensor.max()
    min_val = I_tensor.min()
    if max_val > min_val:
        psnr = 10 * torch.log10((max_val - min_val) ** 2 / mse)
    else:
        psnr = torch.tensor(0.0)

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

if __name__ == "__main__":
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    program_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    mat_folder = './data/int8' 
    all_data = load_mat_files(mat_folder)

    if len(all_data) == 0:
        print("No data loaded. Please check the data folder path.")
        exit(1)

    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f'Loaded {len(all_data)} samples, {len(train_data)} for training, {len(test_data)} for testing')

    train_results = []
    best_ps = []
    all_histories = []
    all_Qs = []

    num_train_samples = min(60, len(train_data))

    start_time = time.time()
    for i, sample in enumerate(train_data[:num_train_samples]):
        print(f'\n{"=" * 60}')
        print(f'Training on sample {i + 1}/{num_train_samples}')
        print(f'{"=" * 60}')

        I_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
        print(f'Sample shape: {I_tensor.shape}')

        rl_pca = OptimizedRLPCA(device)

        best_p, Q, history = rl_pca.train(I_tensor, max_episodes=60, verbose=True)
        best_ps.append(best_p)
        all_histories.append(history)
        all_Qs.append(Q)

        train_metrics = evaluate_compression(I_tensor, best_p, rl_pca)
        train_results.append(train_metrics)

        print(f'\nSample {i + 1} Results:')
        print(f'  Best p: {best_p}')
        print(f'  PSNR: {train_metrics["psnr"]:.2f} dB')
        print(f'  SSIM: {train_metrics["ssim"]:.4f}')
        print(f'  compress time: {train_metrics["compress_time"]*1000:.4f}'+'ms')
        print(f'  decompress time: {train_metrics["decompress_time"]*1000:.4f}'+'ms')
        print(f'  Compression Ratio: {train_metrics["compression_ratio"]:.4f}')

    train_time = time.time() - start_time
    avg_best_p = int(np.median(best_ps))
    print(f'\n{"=" * 60}')
    print(f'Average best p across training samples: {avg_best_p}')
    print(f'{"=" * 60}')

    test_results = []
    test_rl_pca = OptimizedRLPCA(device)  

    num_test_samples = min(40, len(test_data))

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

    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)

    train_psnrs = [r['psnr'] for r in train_results]
    train_ssims = [r['ssim'] for r in train_results]
    train_crs = [r['compression_ratio'] for r in train_results]

    print(f'\nTraining set averages:')
    print(f'  PSNR: {np.mean(train_psnrs):.2f}dB')
    print(f'  SSIM: {np.mean(train_ssims):.4f}')
    print(f'  Compression Ratio: {np.mean(train_crs):.4f} ')
    print(f'\nTrain time: {train_time:.3g} seconds ({train_time / 60:.3g} minutes)')

    test_psnrs = [r['psnr'] for r in test_results]
    test_ssims = [r['ssim'] for r in test_results]
    test_crs = [r['compression_ratio'] for r in test_results]

    print(f'\nTest set averages:')
    print(f'  PSNR: {np.mean(test_psnrs):.2f} dB')
    print(f'  SSIM: {np.mean(test_ssims):.4f}')
    print(f'  Compression Ratio: {np.mean(test_crs):.4f}')

    total_time = time.time() - program_start_time
    print(f'\nTotal program runtime: {total_time:.3g} seconds ({total_time / 60:.3g} minutes)')
    print('\n' + '=' * 60)
    print('Training completed successfully!')
    print('=' * 60)