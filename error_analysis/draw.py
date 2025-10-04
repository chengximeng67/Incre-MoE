# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms as T
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm

def load_image_and_fft(image_path: str) -> np.ndarray:
    """
    加载、转换图像并计算其2D FFT，返回展平的实部和虚部。
    """
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    try:
        with Image.open(image_path) as img_source:
            image_source = img_source.convert("RGB")
        image_transformed = transform(image_source)
        fft_result = torch.fft.fft2(image_transformed)
        fft_flat = fft_result.view(-1).cpu()
        real_part = fft_flat.real.numpy()
        imag_part = fft_flat.imag.numpy()
        return np.concatenate([real_part, imag_part])
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def gather_all_samples(base_test_dirs: dict, error_dir_root: str) -> list:
    """
    收集所有测试样本，并标记哪些是错误分类的。
    """
    all_samples = []
    misclassified_set = set()
    print("Identifying all misclassified samples...")
    if os.path.exists(error_dir_root):
        for root, _, files in os.walk(error_dir_root):
            for name in files:
                misclassified_set.add(name)
    else:
        print(f"Warning: Error directory not found at {error_dir_root}")
    print(f"Found {len(misclassified_set)} unique misclassified filenames.")
    
    print("Gathering all test samples and their classification status...")
    for class_name, dir_path in base_test_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found, skipping: {dir_path}")
            continue
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for filename in image_files:
            is_misclassified = filename in misclassified_set
            full_path = os.path.join(dir_path, filename)
            all_samples.append({'path': full_path, 'class': class_name, 'misclassified': is_misclassified})
    return all_samples


if __name__ == "__main__":
    # --- 1. 配置路径 ---
    BASE_DATA_DIR = r"Q:\chxm\MBMoE\dataset"
    ERROR_DIR_ROOT = r"Q:\chxm\MBMoE\error_analysis_output\a_0.6\pt6"

    base_test_directories = {
        'Larch': os.path.join(BASE_DATA_DIR, 'larch_dataset', 'test'),
        'Smoke': os.path.join(BASE_DATA_DIR, 'smoke_dataset', 'test'),
        'Vehicle': os.path.join(BASE_DATA_DIR, 'vehicle_dataset', 'test'),
        'Car': os.path.join(BASE_DATA_DIR, 'flir_dataset', 'car_test'),
        'Person': os.path.join(BASE_DATA_DIR, 'flir_dataset', 'person_test'),
    }

    # --- 2. 数据加载与特征提取 ---
    all_samples_info = gather_all_samples(base_test_directories, ERROR_DIR_ROOT)
    
    print("\nExtracting FFT features for all samples...")
    features_list, info_list = [], []
    for sample in tqdm(all_samples_info, desc="Processing Images"):
        features = load_image_and_fft(sample['path'])
        if features is not None:
            features_list.append(features)
            info_list.append(sample)
    
    all_features = np.array(features_list)
    print(f"Successfully created feature matrix with shape: {all_features.shape}")
    
    # --- 3. t-SNE 降维 ---
    print("\nPerforming t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_features)
    
    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0],
        'tsne-2': tsne_results[:, 1],
        'class': [info['class'] for info in info_list],
        'misclassified': [info['misclassified'] for info in info_list]
    })

    # --- 4. 最终的大图可视化 (图例已调整到左上角并缩小) ---
    print("\nGenerating the final comprehensive plot...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix' 
    
    class_colors = {
        'Larch': '#2ca02c', 'Smoke': '#7f7f7f', 'Vehicle': '#ff7f0e',
        'Car': '#1f77b4', 'Person': '#9467bd'
    }   
    
    df_correct = df[~df['misclassified']]
    df_error = df[df['misclassified']]
    error_larch = df_error[df_error['class'] == 'Larch']
    error_sv = df_error[df_error['class'].isin(['Smoke', 'Vehicle'])]
    error_cp = df_error[df_error['class'].isin(['Car', 'Person'])]

    plt.figure(figsize=(18, 12))

    ax = sns.scatterplot(
        x='tsne-1', y='tsne-2', data=df_correct, hue='class', 
        palette=class_colors, alpha=0.8, s=60, zorder=1, legend=True
    )

    sns.scatterplot(
        x='tsne-1', y='tsne-2', data=error_larch, 
        color='red', marker='X', s=100, zorder=5,
        label='Misclassified Larch Samples'
    )
    sns.scatterplot(
        x='tsne-1', y='tsne-2', data=error_sv, 
        color='black', marker='X', s=100, zorder=5, 
        label='Misclassified Smoke/Vehicle Samples'
    )
    sns.scatterplot(
        x='tsne-1', y='tsne-2', data=error_cp, 
        color='yellow', marker='X', s=110, zorder=5, 
        edgecolor='black', linewidth=1,
        label='Misclassified Car/Person Samples'
    )

    # ==================== 核心修改：调整图例位置和大小 ====================
    handles, labels = ax.get_legend_handles_labels()

    n_classes = len(class_colors)
    reordered_handles = handles[n_classes:] + handles[:n_classes]
    reordered_labels = labels[n_classes:] + labels[:n_classes]
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.legend(
        handles=reordered_handles,
        labels=reordered_labels,
        title='Category / Error Type',
        fontsize=12,             # 减小字体
        title_fontsize=14,       # 减小标题字体
        loc='upper left',        # 移动到左上角
        frameon=True,
        facecolor='white',
        framealpha=0.9
    )
    # =====================================================================

    #plt.title('Comprehensive t-SNE Visualization of All Samples in Frequency Domain', fontsize=24, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout(pad=1.0)
    
    output_filename = "t-SNE_Visualization_Legend_TopLeft.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {output_filename}")
    plt.show()