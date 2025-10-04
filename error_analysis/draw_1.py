import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from PIL import Image
from torchvision import transforms as T
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# --- 1. 图像处理和FFT函数 (无变化) ---
def load_image_and_fft(image_path: str) -> np.ndarray:
    """加载、转换图像并计算其2D FFT的实部和虚部。"""
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

# --- 2. 数据收集 (无变化) ---
def gather_all_samples(base_test_dirs: dict, error_dir_root: str) -> list:
    """收集所有测试样本，并标记哪些是错误分类的。"""
    all_samples = []
    misclassified_set = set()
    print("Identifying all misclassified samples...")
    # 遍历错误文件夹，记录所有错误分类的文件名
    for root, _, files in os.walk(error_dir_root):
        for name in files:
            misclassified_set.add(name)
    print(f"Found {len(misclassified_set)} unique misclassified filenames.")
    
    print("Gathering all test samples and their classification status...")
    # 遍历所有测试集目录
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

# --- 3. 主执行流程 ---
if __name__ == "__main__":
    # 请确保这里的路径是您本机的正确路径
    BASE_DATA_DIR = r"Q:\chxm\MBMoE\dataset"
    ERROR_DIR_ROOT = r"Q:\chxm\MBMoE\error_analysis_output\a_0.6\pt6"

    base_test_directories = {
        'Larch': os.path.join(BASE_DATA_DIR, 'larch_dataset', 'test'),
        'Smoke': os.path.join(BASE_DATA_DIR, 'smoke_dataset', 'test'),
        'Vehicle': os.path.join(BASE_DATA_DIR, 'vehicle_dataset', 'test'),
        'Car': os.path.join(BASE_DATA_DIR, 'flir_dataset', 'car_test'),
        'Person': os.path.join(BASE_DATA_DIR, 'flir_dataset', 'person_test'),
    }

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
    
    print("\nPerforming t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_features)
    
    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0],
        'tsne-2': tsne_results[:, 1],
        'class': [info['class'] for info in info_list],
        'misclassified': [info['misclassified'] for info in info_list]
    })
    
    # --- 4. 最终的大图可视化 (已根据您的要求修改) ---
    print("\nGenerating the final comprehensive plot...")
    
    # 设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 定义颜色和标记
    class_colors = {
        'Larch': 'green', 'Smoke': 'grey', 'Vehicle': 'orange', 
        'Car': 'blue', 'Person': 'purple'
    }
    
    # 定义错误样本分组
    df_error = df[df['misclassified']]
    error_larch = df_error[df_error['class'] == 'Larch']
    error_sv = df_error[df_error['class'].isin(['Smoke', 'vehicle'])]
    error_cp = df_error[df_error['class'].isin(['Car', 'person'])]

    # <<< 修改点 1: 调整图形尺寸为 1.5:1 的比例 >>>
    plt.figure(figsize=(18, 12))

    # 第一层：绘制所有正确分类的样本
    df_correct = df[~df['misclassified']]
    # 将绘图对象赋值给 ax，方便后续获取图例句柄
    ax = sns.scatterplot(
        x='tsne-1', y='tsne-2', data=df_correct, hue='class', 
        palette=class_colors, alpha=0.6, s=50, zorder=1, legend=True
    )

    # 第二层：逐层绘制不同类型的错误样本
    sns.scatterplot(x='tsne-1', y='tsne-2', data=error_larch, color='hotpink', marker='X', s=120, zorder=5)
    sns.scatterplot(x='tsne-1', y='tsne-2', data=error_sv, color='black', marker='X', s=120, zorder=5)
    sns.scatterplot(x='tsne-1', y='tsne-2', data=error_cp, color='red', marker='X', s=120, zorder=5)
        
    # --- 创建一个清晰的、合并的图例 ---
    # 从 sns.scatterplot 获取自动生成的图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()
    # 移除 seaborn 自动生成的旧图例，以便我们创建自定义的新图例
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    class_handles = handles[:len(class_colors)]
    class_labels = labels[:len(class_colors)]

    # 手动创建错误样本的图例句柄
    error_handles = [
        mlines.Line2D([], [], color='hotpink', marker='X', linestyle='None', markersize=10, label='Misclassified Larch Samples'),
        mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Misclassified Smoke/Vehicle Samples'),
        mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label='Misclassified Car/Person Samples')
    ]
    
    # <<< 修改点 2: 调整图例参数，使其悬浮在图内右上角 >>>
    plt.legend(
        handles=class_handles + error_handles,
        title='Category / Error Type',
        fontsize=12,
        title_fontsize=14,
        loc='upper right',  # 将图例放置在图表内部的右上角
        frameon=True,       # 显示图例边框
        facecolor='white',  # 设置图例背景色为白色
        framealpha=0.8      # 设置背景为半透明，避免完全遮挡数据点
    )

    plt.title('Comprehensive t-SNE Visualization of All Samples in Frequency Domain', fontsize=20, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    # <<< 修改点 3: 使用标准的 tight_layout 自动调整布局 >>>
    plt.tight_layout(pad=1.0)
    
    # 保存图像
    plt.savefig("plot_final_floating_legend_1.5_ratio.png", dpi=300)
    
    # 显示图像
    plt.show()