import os
import shutil
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import torch



# 1. 数据集转换函数（将VOC格式转为YOLO格式）
def convert_voc_to_yolo(xml_path, txt_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in classes:
                continue
            
            cls_id = classes.index(cls_name)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # 转换为YOLO格式并确保有效性
          
            x_center = max(0.0, min(1.0, ((xmin + xmax) / 2) / img_width))
            y_center = max(0.0, min(1.0, ((ymin + ymax) / 2 )/ img_height))
            width = max(0, min(1, (xmax - xmin) / img_width))
            height = max(0, min(1, (ymax - ymin) / img_height))
            
            # 过滤无效标注（宽高为0的情况）
            if width < 1e-6 or height < 1e-6:
                continue
                
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 2. 准备数据集（仅训练集）
dataset_path = "NEU-DET"  # 确保该目录包含IMAGES和ANNOTATIONS子目录
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# 创建YOLO格式目录结构
yolo_dataset = "yolo_neudet"
os.makedirs(os.path.join(yolo_dataset, "images/train"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset, "labels/train"), exist_ok=True)

# 获取所有样本
all_files = [f for f in os.listdir(os.path.join(dataset_path, "IMAGES")) 
            if f.endswith(('.jpg','.png'))]

# 转换并复制所有文件到训练集
for file in all_files:
    # 复制图像
    src_img = os.path.join(dataset_path, "IMAGES", file)
    dst_img = os.path.join(yolo_dataset, "images/train", file)
    shutil.copy(src_img, dst_img)
    
    # 转换并保存标签
    xml_file = os.path.splitext(file)[0] + ".xml"
    src_xml = os.path.join(dataset_path, "ANNOTATIONS", xml_file)
    dst_txt = os.path.join(yolo_dataset, "labels/train", os.path.splitext(file)[0] + ".txt")
    
    if os.path.exists(src_xml):
        convert_voc_to_yolo(src_xml, dst_txt, classes)
    else:
        open(dst_txt, 'a').close()  # 创建空标签文件

# 3. 创建数据集配置文件
# 创建类别权重调整的数据集配置文件
data_yaml = f"""
path: {os.path.abspath(yolo_dataset)}
train: images/train
val: images/train

names:
  0: crazing     # 为稀有类别分配更高权重
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches

# 添加类别权重（稀有类别权重更大）
class_weights:
  0: 3.0  # crazing类别权重
  1: 1.0
  2: 1.0
  3: 1.0
  4: 1.0
  5: 1.0
"""

with open("neu-det.yaml", "w") as f:
    f.write(data_yaml)


# 修改 Detect 层的分类头
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.nn.modules import Detect

# 查看模型结构，找到 Detect 层的输入通道数

# 4. 加载预训练模型并修改分类头
model = YOLO('yolov8n.pt')  # 官方预训练模型
# model.model.nc = len(classes)  # 修改类别数量


# 假设 Detect 层的输入通道数为 256（这个值需要根据实际模型结构确定）
# 你可以根据打印的模型结构找到正确的通道数
detect_input_channels = 256

# 创建一个新的 Detect 层，设置新的类别数量
new_detect_layer = Detect(nc=len(classes), ch=[detect_input_channels])

# 替换原来的 Detect 层
model.model.model[-1] = new_detect_layer
print(model.model)
model.model.nc = len(classes)

# 5. 配置训练参数（禁用验证）
# train_args = {
#     'data': 'neu-det.yaml',
#     'epochs': 100,
#     'batch': 8,  # 减小batch size
#     'imgsz': 640,  # 可尝试减小到416
#     'device': '0' if torch.cuda.is_available() else 'cpu',
#     'optimizer': 'AdamW',
#     'lr0': 1e-4,
#     'save': True,  # 启用模型保存
#     'save_period': 1,  # 每个epoch都保存
#     'val': False,  # 禁用验证
#     'project': 'yolov8_neudet',
#     'name': 'no_val_exp',
#     'exist_ok': True
# }

train_args = {
    'data': 'neu-det.yaml',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': 'cpu',  # 设置设备为 CPU
    'optimizer': 'AdamW',
    'lr0': 5e-5,
    'lrf': 0.01,
    'save': True,
    'save_period': 10,
    'val': False,
    'project': 'yolov8_neudet',
    'name': 'crazing_optimized',
    'exist_ok': True,
    'freeze': [0, 1, 2],
    'cos_lr': True,
    'flipud': 0.5,
    'fliplr': 0.5
}

# 6. 开始训练
results = model.train(**train_args)

# 7. 显式保存最终模型
final_model_path = os.path.join(train_args['project'], train_args['name'], 'weights/last.pt')
print(f"训练完成，最终模型已保存至：{final_model_path}")