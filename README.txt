# 基于YOLO+多目标跟踪的智能视频监控系统
YOLO + Multi-Object Tracking + Intelligent Monitoring System

## 项目简介
本项目基于YOLO目标检测与多目标跟踪算法，实现视频流中的智能分析功能，适用于安防监控、人流统计、区域入侵检测等场景。

## 实现功能
 目标检测（行人/车辆检测）
 多目标实时跟踪
 禁区闯入报警（越界提醒）
 分方向进出统计（进入/离开计数）
 追踪效果评估


## 技术栈
- Python
- YOLOv8 / Ultralytics
- 多目标跟踪算法
- OpenCV 图像处理
- NumPy

## 项目亮点
1. 完整实现检测+跟踪+统计+报警一体化系统
2. 支持自定义禁区、统计方向线
3. 输出可视化结果，实用性强
4. 可直接部署到监控摄像头/视频文件

## 使用说明
1. 安装依赖：pip install -r requirements.txt
2. 运行主程序：python track_and_save.py
3. 配置禁区与统计线可在代码中修改


