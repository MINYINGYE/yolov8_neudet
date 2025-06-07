import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PySide6.QtCore import Qt
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("钢材缺陷检测系统")
        self.setGeometry(100, 100, 900, 650)

        # 设置应用程序字体为支持中文的字体
        font = QFont("Microsoft YaHei", 10)  # 使用微软雅黑字体
        self.setFont(font)

        # 初始化模型（修改为6类别，加载NEU-DET训练的权重）
        self.model = YOLO("yolov8_neudet/no_val_exp/weights/best.pt")  # 加载训练好的 YOLOv8 模型
        self.model.to('cpu')  # 将模型移动到 CPU 上

        # 创建主窗口
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建布局
        layout = QVBoxLayout()
        layout.setSpacing(10)  # 减少组件之间的间距
        main_widget.setLayout(layout)

        # 创建图像显示区域
        self.image_layout = QHBoxLayout()
        self.image_layout.setSpacing(10)  # 减少图像区域之间的间距

        # 原始图像显示区域
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(400, 300)
        self.original_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.image_layout.addWidget(self.original_label)

        # 检测结果图像显示区域
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(400, 300)
        self.result_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.image_layout.addWidget(self.result_label)

        layout.addLayout(self.image_layout)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # 减少按钮之间的间距

        # 创建按钮
        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet("padding: 8px 15px; border-radius: 4px; background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.load_button)

        self.detect_button = QPushButton("开始检测")
        self.detect_button.clicked.connect(self.detect_defects)
        self.detect_button.setStyleSheet("padding: 8px 15px; border-radius: 4px; background-color: #2196F3; color: white;")
        button_layout.addWidget(self.detect_button)

        layout.addLayout(button_layout)

        # 添加描述标签
        self.description_label = QLabel("原始图像                      检测结果")
        self.description_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.description_label)

        # 创建结果显示区域
        self.result_text = QLabel()
        self.result_text.setAlignment(Qt.AlignCenter)
        self.result_text.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.result_text)

        # 创建详细结果区域
        self.detail_results = QTextEdit()
        self.detail_results.setReadOnly(True)
        self.detail_results.setStyleSheet("border: 1px solid #ddd; border-radius: 4px; padding: 10px;")
        self.detail_results.setMinimumHeight(150)
        self.detail_results.setFont(QFont("Microsoft YaHei", 9))
        layout.addWidget(self.detail_results)

        # 添加状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

        # 初始化变量
        self.current_image_path = None
        self.current_image_bgr = None  # 存储 BGR 格式图像（OpenCV 格式）
        self.detection_performed = False  # 添加标志变量

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.current_image_path = file_paths[0]
                # 处理中文路径问题：使用二进制模式读取文件
                try:
                    with open(self.current_image_path, 'rb') as f:
                        img_data = f.read()
                    nparr = np.frombuffer(img_data, np.uint8)
                    self.current_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if self.current_image_bgr is None:
                        self.status_bar.showMessage(f"无法解码图像: {self.current_image_path}")
                        return
                    
                    # 调整图像大小以适应显示区域
                    height, width = self.current_image_bgr.shape[:2]
                    max_size = max(width, height)
                    scale = 300 / max_size
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    self.current_image_bgr = cv2.resize(self.current_image_bgr, (new_width, new_height))
                    
                    self.display_image(self.current_image_bgr, self.original_label)
                    self.result_label.clear()  # 清空检测结果显示区域
                    self.result_text.setText("加载成功！点击“开始检测”按钮进行检测。")
                    self.detail_results.setText("")
                    self.detection_performed = False  # 重置标志
                    self.status_bar.showMessage(f"图像加载成功: {os.path.basename(self.current_image_path)}")
                except Exception as e:
                    self.status_bar.showMessage(f"加载图像时出错: {str(e)}")

    def display_image(self, image_bgr, label):
        if image_bgr is None:
            label.clear()
            return
        # BGR 转 RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QPixmap.fromImage(
            QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        )
        label.setPixmap(q_image.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_defects(self):
        if self.current_image_bgr is None:
            self.result_text.setText("请先加载图像!")
            self.status_bar.showMessage("错误：请先加载图像")
            return

        if self.detection_performed:
            self.result_text.setText("检测已完成！请重新加载图像以进行新的检测。")
            self.status_bar.showMessage("提示：请重新加载图像进行新的检测")
            return

        # 检测
        with torch.no_grad():
            results = self.model(self.current_image_bgr)  # 直接使用模型进行检测
        
        # 后处理输出
        boxes, labels, scores, defect_info = self.postprocess_output(results[0])

        # 在原始图像上绘制检测结果
        result_image = self.current_image_bgr.copy()
        class_names = [
            "裂纹", "夹杂", "斑块", "麻点", "氧化皮", "划痕"  # NEU-DET的6个类别（与训练索引0-5对应）
        ]

        # 绘制边界框和中文文本
        result_image = self.draw_results(result_image, boxes, labels, scores, class_names)

        # 显示检测结果图像
        self.display_image(result_image, self.result_label)

        # 更新结果显示文本
        if len(boxes) == 0:
            self.result_text.setText("未检测到任何缺陷。")
            self.detail_results.setText("检测结果：未发现钢材缺陷")
        else:
            self.result_text.setText(f"检测到 {len(boxes)} 处缺陷")
            self.detail_results.setText(defect_info)

        self.detection_performed = True  # 设置标志为 True
        self.status_bar.showMessage("检测完成")

    def postprocess_output(self, outputs, score_threshold=0.5):
        boxes = []
        labels = []
        scores = []
        defect_info = "检测结果：\n\n"
        
        if outputs is None or len(outputs) == 0:
            return boxes, labels, scores, "未检测到任何缺陷"

        class_names = ["裂纹", "夹杂", "斑块", "麻点", "氧化皮", "划痕"]
        
        # 遍历检测结果
        for result in [outputs]:
            # 获取检测到的边界框、类别和置信度
            for i in range(len(result.boxes)):
                # 跳过背景类别（如果有）
                cls_id = int(result.boxes.cls[i].item())
                score = result.boxes.conf[i].item()
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = result.boxes.xyxy[i].int().tolist()
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)
                scores.append(score)
                
                # 构建详细信息文本
                defect_info += f"缺陷类型: {class_names[cls_id]}\n"
                defect_info += f"置信度: {score:.2f}\n"
                defect_info += f"位置: ({x1}, {y1}) - ({x2}, {y2})\n\n"
        
        if not defect_info.strip():
            defect_info = "未检测到任何缺陷"
        else:
            defect_info = defect_info.strip()
        
        return boxes, labels, scores, defect_info

    def draw_results(self, image_bgr, boxes, labels, scores, class_names):
        # 绘制边界框
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # 阈值可根据需要调整
                x1, y1, x2, y2 = box
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 使用 PIL 绘制中文文本
        result_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(result_image)
        font = ImageFont.truetype("msyh.ttc", 20)  # 使用微软雅黑字体

        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # 阈值可根据需要调整
                x1, y1, x2, y2 = box
                text = f"{class_names[label]}: {score:.2f}"
                text_width = draw.textlength(text, font=font)
                
                # 绘制文本背景
                draw.rectangle([x1, y1, x1 + text_width + 10, y1 + 30], fill=(0, 0, 0, 128))
                
                # 绘制文本
                draw.text((x1 + 5, y1 + 5), text, fill=(255, 255, 255), font=font)

        return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))  # 设置全局字体为微软雅黑
    window = MainWindow()
    window.show()
    sys.exit(app.exec())