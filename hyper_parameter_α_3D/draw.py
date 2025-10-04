import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gaussian_kde

# =============================================================================
# 1. 初始化设置 (与之前相同)
# =============================================================================
mpl.rcParams['font.size'] = 14
try:
    mpl.rcParams['font.family'] = 'Times New Roman'
except:
    print("Times New Roman not found, using default font.")
base_path = r"d:\OneDrive\大模型相关论文\持续学习\router实部虚部超参" # 请确保此路径存在且包含数据

# =============================================================================
# 2. 数据读取与处理 (与之前相同)
# =============================================================================
print("开始读取和处理数据...")
data_list = []
folder_names = [f"{i/10.0:.1f}" for i in range(11)] 
for folder_name in folder_names:
    folder_path = os.path.join(base_path, folder_name)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv') and 'result' in file_name:
                file_path = os.path.join(folder_path, file_name)
                try:
                    data = pd.read_csv(file_path)
                    if 'AP' in data.columns:
                        for v in data['AP'].tolist():
                            data_list.append({'hyperparameter': folder_name, 'Metric': '$AP_{50:95}$', 'value': v})
                    if 'AP50' in data.columns:
                        for v in data['AP50'].tolist():
                            data_list.append({'hyperparameter': folder_name, 'Metric': '$AP_{50}$', 'value': v})
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
df_long = pd.DataFrame(data_list)

# 确保 hyperparameter 列是数值类型，以便在3D图中作为连续轴进行排序和定位
df_long['hyperparameter_numeric'] = df_long['hyperparameter'].astype(float)
# 按照数值进行排序
df_long = df_long.sort_values(by='hyperparameter_numeric')

print("数据准备完毕。")

if df_long.empty:
    print("错误：未能加载任何数据，无法生成图表。")
else:
    # =============================================================================
    # 3. 绘图：3D 密度图 (3D Ridge Plot) - 增强版 (移除Peak，添加Median)
    # =============================================================================
    
    # 筛选出两种指标的数据
    df_ap = df_long[df_long['Metric'] == '$AP_{50:95}$']
    df_ap50 = df_long[df_long['Metric'] == '$AP_{50}$']

    # --- 获取所有 AP 值的共同范围，用于 KDE 的 X 轴 ---
    min_ap_val = df_long['value'].min()
    max_ap_val = df_long['value'].max()
    ap_range_padding = (max_ap_val - min_ap_val) * 0.1 
    x_kde_grid = np.linspace(min_ap_val - ap_range_padding, max_ap_val + ap_range_padding, 200) 

    # --- 设置颜色映射，用于区分不同的超参数 ---
    colormap = plt.cm.plasma_r 
    norm = plt.Normalize(df_long['hyperparameter_numeric'].min(), df_long['hyperparameter_numeric'].max())

    # --- 辅助函数：绘制3D KDE 曲线和填充 (增强版，移除Peak，添加Median) ---
    def plot_3d_kde_enhanced_median_only(ax, df_metric, x_grid, colormap_func, norm_func, metric_name):
        unique_hyperparameters = df_metric['hyperparameter_numeric'].unique()
        
        # 按照超参数从小到大排序，这样在3D图中会从“后”到“前”绘制，避免遮挡
        sorted_hps = np.sort(unique_hyperparameters) 

        # 用于存储均值和中位数，以便绘制趋势线
        mean_points = [] 
        median_points = [] 

        # ====== 核心修改点：调整文本Z轴偏移量，并设置字体大小和背景框透明度 ======
        mean_text_z_offset = 0.55 
        median_text_z_offset = 0.85 # 增加偏移量，确保中位数标签在均值标签上方且有足够空间
        label_font_size = 14 # 增大字体
        bbox_alpha = 0.7 # 增加背景框透明度，使其更不透明
        # ====================================================================

      
        chosen_best_hp_val = 0.6 
        if chosen_best_hp_val not in unique_hyperparameters:
            print(f"警告: 指定的最佳超参数 {chosen_best_hp_val} 不在数据中，将不会高亮显示。")
            best_hp_val = None 
        else:
            best_hp_val = chosen_best_hp_val

        for hp_val in sorted_hps:
            subset = df_metric[df_metric['hyperparameter_numeric'] == hp_val]
            data_points = subset['value'].dropna().values

            if len(data_points) < 2: 
                continue

            kde = gaussian_kde(data_points)
            y_kde = kde(x_grid) 

            # 计算均值和中位数
            mean_ap_val = np.mean(data_points)
            median_ap_val = np.median(data_points) 

            color = colormap_func(norm_func(hp_val))
            
            # 高亮指定超参数曲线和点
            is_best_hp = (hp_val == best_hp_val)
            line_color = 'red' if is_best_hp else color
            line_width = 2.5 if is_best_hp else 1.5
            alpha_val = 0.9 if is_best_hp else 0.7
            
            # 绘制 KDE 曲线本身
            ax.plot(x_grid, hp_val * np.ones_like(x_grid), y_kde, 
                    color=line_color, alpha=alpha_val, linewidth=line_width)
            
            # 构建填充区域的多边形
            verts = [list(zip(x_grid, hp_val * np.ones_like(x_grid), y_kde)) + \
                     list(zip(x_grid[::-1], hp_val * np.ones_like(x_grid[::-1]), np.zeros_like(y_kde[::-1])))]
            poly = Poly3DCollection(verts, alpha=0.1, facecolor=color)
            ax.add_collection3d(poly)

            # 绘制均值点 (在Z=0平面上)
            ax.scatter(mean_ap_val, hp_val, 0, 
                       color='blue' if not is_best_hp else 'darkblue', 
                       s=80 if is_best_hp else 40, 
                       marker='o', 
                       edgecolors='black', linewidth=0.5,
                       label='Mean AP' if hp_val == sorted_hps[0] else None) 
            
            # 绘制中位数点 (在Z=0平面上)
            ax.scatter(median_ap_val, hp_val, 0, 
                       color='purple' if not is_best_hp else 'darkmagenta', 
                       s=80 if is_best_hp else 40, 
                       marker='s', 
                       edgecolors='black', linewidth=0.5,
                       label='Median AP' if hp_val == sorted_hps[0] else None) 

            mean_points.append((mean_ap_val, hp_val, 0)) 
            median_points.append((median_ap_val, hp_val, 0)) 

            # 为指定超参数添加文本标注并加粗
            if is_best_hp:
                # ====== 核心修改点：使用新的字体大小和背景框透明度 ======
                ax.text(mean_ap_val, hp_val, mean_text_z_offset, f'Mean: {mean_ap_val:.2f}', 
                        color='darkblue', fontsize=label_font_size, ha='center', va='bottom',
                        fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=bbox_alpha, edgecolor='none', boxstyle='round,pad=0.2')) 
                ax.text(median_ap_val, hp_val, median_text_z_offset, f'Median: {median_ap_val:.2f}', 
                        color='darkmagenta', fontsize=label_font_size, ha='center', va='bottom',
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=bbox_alpha, edgecolor='none', boxstyle='round,pad=0.2')) 
                # ======================================================
        
        # 绘制均值趋势线
        if mean_points:
            mean_points = np.array(mean_points)
            ax.plot(mean_points[:, 0], mean_points[:, 1], mean_points[:, 2], 
                    color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Mean Trend')
        
        # 绘制中位数趋势线
        if median_points:
            median_points = np.array(median_points)
            ax.plot(median_points[:, 0], median_points[:, 1], median_points[:, 2], 
                    color='purple', linestyle=':', linewidth=2, alpha=0.7, label='Median Trend') 

        # 放置图例
        from matplotlib.collections import PathCollection
        from matplotlib.lines import Line2D
        
        custom_handles = [
            Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=8, label='Mean AP'),
            Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Mean Trend'),
            Line2D([0], [0], marker='s', color='purple', linestyle='None', markersize=8, label='Median AP'),
            Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Median Trend')
        ]
        
        ax.legend(handles=custom_handles, loc='upper left', bbox_to_anchor=(0.05, 0.95))


    # --- 创建两个子图，每个子图是一个独立的3D图 ---
    fig = plt.figure(figsize=(26, 12)) 

    # --- 左子图: AP50:95 的 3D 密度图 ---
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_kde_enhanced_median_only(ax1, df_ap, x_kde_grid, colormap, norm, '$AP_{50:95}$')
    
    ax1.set_title('$AP_{50:95}$ Distribution (Mean & Median Trends)', fontsize=18, weight='bold')
    ax1.set_xlabel('AP Score', fontsize=14)
    ax1.set_ylabel('FDR Hyper-parameters α', fontsize=14)
    ax1.set_zlabel('Density', fontsize=14)
    
    ax1.view_init(elev=25, azim=-55) 
    ax1.set_yticks(df_ap['hyperparameter_numeric'].unique())
    ax1.set_yticklabels([f'{val:.1f}' for val in df_ap['hyperparameter_numeric'].unique()])
    ax1.set_zlim(0, None) 

    # --- 右子图: AP50 的 3D 密度图 ---
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_kde_enhanced_median_only(ax2, df_ap50, x_kde_grid, colormap, norm, '$AP_{50}$')
    
    ax2.set_title('$AP_{50}$ Distribution (Mean & Median Trends)', fontsize=18, weight='bold')
    ax2.set_xlabel('AP Score', fontsize=14)
    ax2.set_ylabel('FDR Hyper-parameters α', fontsize=14)
    ax2.set_zlabel('Density', fontsize=14)
    ax2.view_init(elev=25, azim=-55) 
    ax2.set_yticks(df_ap50['hyperparameter_numeric'].unique())
    ax2.set_yticklabels([f'{val:.1f}' for val in df_ap50['hyperparameter_numeric'].unique()])
    ax2.set_zlim(0, None) 


    # # --- 添加颜色条 (可选，但推荐，以显示超参数的颜色映射) ---
    # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7]) 
    # cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=colormap, norm=norm,
    #                                   orientation='vertical')
    # cbar.set_label('Hyper-parameter Value α', fontsize=12)

    # --- 全局设置 ---
    # fig.suptitle('3D Density Plots of AP Score Distributions by Hyper-parameter (Mean & Median Quantiles)', fontsize=22, weight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.94]) 

    # --- 保存与显示 ---
    output_path_3d_median_only_fixed_hp_bbox_larger_text = os.path.join(base_path, 'Figure_3D_Density_Subplots_MedianOnly_HP06_LargerLabels.png') 
    plt.savefig(output_path_3d_median_only_fixed_hp_bbox_larger_text, dpi=300)
    print(f"增强版3D密度图 (标签更大更清晰, 0.6高亮) 已成功保存至: {output_path_3d_median_only_fixed_hp_bbox_larger_text}")
    plt.show()