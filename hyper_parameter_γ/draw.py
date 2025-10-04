import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import matplotlib.ticker as mticker # 导入 ticker 模块

# --- 全局设置 ---
# 设置 Matplotlib 字体和字号
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'Times New Roman'

# 定义基础路径和颜色映射
BASE_PATH = Path(r'D:\OneDrive\大模型相关论文\持续学习\SLERP插值超参')
# 修复 DeprecationWarning
CMAP_OURS = plt.get_cmap('Blues')     # 用于 "Ours" (slerp)
CMAP_WISEFT = plt.get_cmap('Oranges') # 用于 "WiSE-FT" (weighted)

# --- 辅助函数 ---
def load_data_for_task(task_name: str):
    """根据任务名称加载 slerp 和 weighted 的数据"""
    slerp_path = BASE_PATH / f'{task_name}_results.csv'
    weighted_path = BASE_PATH / 'duibi' / f'{task_name}_results.csv'
    
    slerp_df = pd.read_csv(slerp_path)
    weighted_df = pd.read_csv(weighted_path)
    
    data = {
        'slerp_ap': slerp_df['AP'].tolist(),
        'slerp_ap50': slerp_df['AP50'].tolist(),
        'weighted_ap': weighted_df['AP'].tolist(),
        'weighted_ap50': weighted_df['AP50'].tolist()
    }
    return data

def plot_comparison_subplot(ax, x_slerp, y_slerp, x_weighted, y_weighted, title, xlabel, ylabel):
    """在指定的子图(ax)上绘制两种方法的对比散点图，并添加图例"""
    ax.set_title(title, fontsize=14)
    
    # 确保列表非空，避免除以零
    len_slerp = len(x_slerp) if len(x_slerp) > 1 else 2
    len_weighted = len(x_weighted) if len(x_weighted) > 1 else 2

    # 绘制 "Ours" (slerp) 的散点图
    for i, (x, y) in enumerate(zip(x_slerp, y_slerp)):
        ax.scatter(x, y, color=CMAP_OURS(i / (len_slerp - 1)), edgecolor='black', zorder=3)
        
    # 绘制 "WiSE-FT" (weighted) 的散点图
    for i, (x, y) in enumerate(zip(x_weighted, y_weighted)):
        ax.scatter(x, y, color=CMAP_WISEFT(i / (len_weighted - 1)), edgecolor='black', zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)

    # === 新增代码：设置坐标轴刻度标签的精度 ===
    # 使用 FormatStrFormatter 来限制小数点后的位数
    # '%.3f' 表示浮点数，保留小数点后3位
    formatter = mticker.FormatStrFormatter('%.3f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # ==========================================

    # === 保留您原来的图例生成方式 ===
    # 添加用于图例的“隐藏”点
    ax.scatter([], [], color='blue', label='Ours', edgecolor='black')
    ax.scatter([], [], color='orange', label='WiSE-FT', edgecolor='black')
    ax.legend()
    # ===============================

# --- 主逻辑 ---
# 1. 加载所有数据
smoke_data = load_data_for_task('smoke')
vehicle_data = load_data_for_task('vehicle')
car_data = load_data_for_task('car')
person_data = load_data_for_task('person')

# 2. 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(9, 8))

# 3. 绘制每个子图
# 子图 [0, 0]: Smoke-Vehicle AP (AP at IoU=0.50:0.95)
plot_comparison_subplot(
    axs[0, 0],
    smoke_data['slerp_ap'], vehicle_data['slerp_ap'],
    smoke_data['weighted_ap'], vehicle_data['weighted_ap'],
    title='Smoke-Vehicle (AP 50:95)',
    xlabel='Smoke AP50:95',
    ylabel='Vehicle AP50:95'
)

# 子图 [0, 1]: Smoke-Vehicle AP50
plot_comparison_subplot(
    axs[0, 1],
    smoke_data['slerp_ap50'], vehicle_data['slerp_ap50'],
    smoke_data['weighted_ap50'], vehicle_data['weighted_ap50'],
    title='Smoke-Vehicle (AP 50)',
    xlabel='Smoke AP50',
    ylabel='Vehicle AP50'
)

# 子图 [1, 0]: Car-Person AP
plot_comparison_subplot(
    axs[1, 0],
    car_data['slerp_ap'], person_data['slerp_ap'],
    car_data['weighted_ap'], person_data['weighted_ap'],
    title='Car-Person (AP 50:95)',
    xlabel='Car AP50:95',
    ylabel='Person AP50:95'
)

# 子图 [1, 1]: Car-Person AP50
plot_comparison_subplot(
    axs[1, 1],
    car_data['slerp_ap50'], person_data['slerp_ap50'],
    car_data['weighted_ap50'], person_data['weighted_ap50'],
    title='Car-Person (AP 50)',
    xlabel='Car AP50',
    ylabel='Person AP50'
)

# 4. 调整布局并保存/显示
plt.tight_layout()
output_path = BASE_PATH.parent / 'Figure_1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()