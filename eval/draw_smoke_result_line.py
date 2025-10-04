import pandas as pd
import matplotlib.pyplot as plt
#csv_name = "larch_backbone_encoder"
import os


def plot_png(path,save_path):

    # 读取CSV内容为DataFrame
    data = pd.read_csv(path)


    # 定义需要绘制折线图的列
    # columns_to_plot = [
    #     "AP@[IoU=0.50:0.95|area=all|maxDets=100]",
    #     "AP@[IoU=0.50|area=all|maxDets=100]",
    #     "AP@[IoU=0.75|area=all|maxDets=100]",
    #     "AR@[IoU=0.50:0.95|area=all|maxDets=1]",
    #     "AR@[IoU=0.50:0.95|area=all|maxDets=10]",
    #     "AR@[IoU=0.50:0.95|area=all|maxDets=100]"
    # ]
    columns_to_plot = [
        "AP@[IoU=0.50:0.95|area=all|maxDets=100]",
        "AP@[IoU=0.50|area=all|maxDets=100]",
        "AP@[IoU=0.75|area=all|maxDets=100]"
    ]
    # 绘制每个列的折线图并保存
    plt.figure(figsize=(14, 10))
    for column in columns_to_plot:
        plt.plot(data['Epoch'], data[column], marker='o', label=column)
    # 添加标题和标签
    plt.title('Performance Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    # 保存图像到当前目录
    plt.savefig(save_path)

def check_and_create_directory(directory):
    try:
        # 检查目录是否存在
        if not os.path.exists(directory):
            # 如果目录不存在，则创建目录
            os.makedirs(directory)
            #print(f"Directory '{directory}' created.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    csv_name_list = ["smoke_backbone_and_encoder"]
    save_path = "/home/enbo/chxm/MBMoE/smoke_result/"
    check_and_create_directory(save_path)
    for csv_name in csv_name_list:
        path = "/home/enbo/chxm/MBMoE/weights/"+csv_name+"_all/"+csv_name+"_result.csv"
        plot_png(path,save_path+csv_name)