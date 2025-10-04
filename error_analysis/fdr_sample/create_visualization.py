import os
from PIL import Image, ImageDraw, ImageFont

# --- 1. 配置区域 ---

# 存放样本图片的根目录
BASE_PATH = r'Q:\CHXM\MBMOE\ERROR_ANALYSIS_OUTPUT\FDR_SAMPLE' 

# 生成的可视化图片的保存路径和文件名
OUTPUT_IMAGE_PATH = 'confusion_matrix_visualization_times_new_roman.jpg'

# 定义表格的行列标签
TRUE_LABELS = ['True Larch', 'True Smoke', 'True Car', 'True Vehicle', 'True Person']
PREDICTED_LABELS = ['Base', 'Larch', 'Smoke/Vehicle', 'Car/Person']

# 可视化参数
IMG_SIZE = (300, 300)
GRID_SPACING = 10
# GLOBAL_LEFT_TITLE_WIDTH 已移除，因为它不再需要
LEFT_MARGIN_WIDTH = 150       # 用于放置具体预测类别的宽度（旋转90度）
TOP_MARGIN_HEIGHT = 100       # 顶部标签的高度
BACKGROUND_COLOR = 'white'
HEADER_FONT_SIZE = 40
HEADER_FONT_COLOR = 'black'

# --- 字体配置 ---
# 在 Windows 系统上，Times New Roman 字体文件通常在此路径
FONT_PATH = 'C:/Windows/Fonts/times.ttf'     # 常规字体
BOLD_FONT_PATH = 'C:/Windows/Fonts/timesbd.ttf' # 粗体字体文件，可能需要根据您的系统调整

try:
    FONT = ImageFont.truetype(FONT_PATH, HEADER_FONT_SIZE)
except IOError:
    print(f"警告: 常规字体文件 '{FONT_PATH}' 未找到。请确认路径是否正确。")
    print("将使用 Pillow 的默认字体。")
    FONT = ImageFont.load_default()

try:
    BOLD_FONT = ImageFont.truetype(BOLD_FONT_PATH, HEADER_FONT_SIZE)
except IOError:
    print(f"警告: 粗体字体文件 '{BOLD_FONT_PATH}' 未找到。请确认路径是否正确。")
    print("True Labels 将使用常规字体。")
    BOLD_FONT = FONT # 如果粗体字体找不到，则使用常规字体代替

# --- 2. 脚本主逻辑 ---

def create_visualization():
    """根据配置生成最终版的可视化图片"""
    num_rows = len(PREDICTED_LABELS)
    num_cols = len(TRUE_LABELS)

    # 计算总画布尺寸
    # 总宽度 = 左侧预测类别标签宽度 + 所有图片宽度 + 所有图片间距
    # 注意：GLOBAL_LEFT_TITLE_WIDTH 已从这里移除
    total_width = LEFT_MARGIN_WIDTH + (num_cols * IMG_SIZE[0]) + (max(0, num_cols - 1) * GRID_SPACING)
    # 总高度 = 顶部标签高度 + 所有图片高度 + 所有图片间距
    total_height = TOP_MARGIN_HEIGHT + (num_rows * IMG_SIZE[1]) + (max(0, num_rows - 1) * GRID_SPACING)

    # 创建主画布
    canvas = Image.new('RGB', (total_width, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    # --- 绘制标签 ---
    
    # 绘制顶部标签 (True labels) - 加粗
    for i, label in enumerate(TRUE_LABELS):
        # 顶部标签的X坐标现在仅需加上 LEFT_MARGIN_WIDTH
        x = LEFT_MARGIN_WIDTH + (i * (IMG_SIZE[0] + GRID_SPACING)) + (IMG_SIZE[0] / 2)
        y = TOP_MARGIN_HEIGHT / 2
        # 使用 BOLD_FONT 绘制 TRUE_LABELS
        draw.text((x, y), label, font=BOLD_FONT, fill=HEADER_FONT_COLOR, anchor='mm')


    # 绘制左侧旋转90度的具体Predicted Expert分类标签 (coco, larch等)
    for i, label in enumerate(PREDICTED_LABELS):
        bbox = draw.textbbox((0, 0), label, font=FONT)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        
        text_draw.text((-bbox[0], -bbox[1]), label, font=FONT, fill=HEADER_FONT_COLOR)

        rotated_text_img = text_img.rotate(90, expand=True)

        # 左侧预测标签的X坐标现在仅需居中于 LEFT_MARGIN_WIDTH
        center_x = (LEFT_MARGIN_WIDTH / 2)
        center_y = TOP_MARGIN_HEIGHT + (i * (IMG_SIZE[1] + GRID_SPACING)) + (IMG_SIZE[1] / 2)
        paste_x = int(center_x - rotated_text_img.width / 2)
        paste_y = int(center_y - rotated_text_img.height / 2)
        canvas.paste(rotated_text_img, (paste_x, paste_y), rotated_text_img)

    # --- 填充图片网格 ---
    print("开始处理图片...")
    for row_idx, pred_label in enumerate(PREDICTED_LABELS):
        for col_idx, true_label in enumerate(TRUE_LABELS):
            # 注意：实际图片文件名可能不包含 "True " 前缀，这里需要去除它来构建路径
            # 例如 'True larch' -> 'larch'
            # 如果您的文件夹结构已经包含 'True '，请删除 .replace('True ', '')
            sanitized_true_label = true_label.replace('True ', '') 
            
            # 文件名：将 'smoke/vehicle' 转换为 'smoke_vehicle.jpg'
            filename = pred_label.replace('/', '_') + '.jpg'
            image_path = os.path.join(BASE_PATH, sanitized_true_label, filename)

            try:
                with Image.open(image_path) as img:
                    cell_image = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            except FileNotFoundError:
                print(f"警告: 未找到图片 {image_path}，将用空白图片填充。")
                cell_image = Image.new('RGB', IMG_SIZE, BACKGROUND_COLOR)

            # 图片网格的X坐标现在仅需加上 LEFT_MARGIN_WIDTH
            paste_x = LEFT_MARGIN_WIDTH + col_idx * (IMG_SIZE[0] + GRID_SPACING)
            paste_y = TOP_MARGIN_HEIGHT + row_idx * (IMG_SIZE[1] + GRID_SPACING)
            canvas.paste(cell_image, (paste_x, paste_y))

    # 保存最终的图片
    canvas.save(OUTPUT_IMAGE_PATH, quality=95)
    print(f"\n可视化图片已成功保存到: {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    create_visualization()