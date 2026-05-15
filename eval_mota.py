import motmetrics as mm
import numpy as np
import os


def evaluate_mota(gt_dir, res_dir):
    # 1. 创建累加器
    acc = mm.MOTAccumulator(auto_id=True)

    # 2. 遍历结果文件夹中的所有文件
    for res_file in os.listdir(res_dir):
        if not res_file.endswith('.txt'):
            continue

        # 寻找对应的真值文件
        gt_file = os.path.join(gt_dir, res_file)
        res_path = os.path.join(res_dir, res_file)

        if not os.path.exists(gt_file):
            print(f"跳过: 找不到对应的真值文件 {gt_file}")
            continue

        print(f"正在评测序列: {res_file}")

        # 加载数据 (MOT Challenge格式: frame, id, x1, y1, w, h, ...)
        gt_data = np.loadtxt(gt_file, delimiter=',')
        res_data = np.loadtxt(res_path, delimiter=',')

        # 获取帧数范围
        frames = np.unique(gt_data[:, 0])

        for f in frames:
            # 提取当前帧的 gt 和 res
            gt_objs = gt_data[gt_data[:, 0] == f]
            res_objs = res_data[res_data[:, 0] == f]

            # 提取坐标 (x1, y1, w, h)
            gt_boxes = gt_objs[:, 2:6]
            res_boxes = res_objs[:, 2:6]

            # 提取 ID
            gt_ids = gt_objs[:, 1].astype(int)
            res_ids = res_objs[:, 1].astype(int)

            # 计算物体之间的距离矩阵（通常使用 IOU，这里转为距离）
            # 这里的 0.5 是 IOU 阈值，只有 IOU > 0.5 且距离最近才算匹配
            distances = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)

            # 更新累加器
            acc.update(gt_ids, res_ids, distances)

    # 3. 计算并结果
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota', 'motp', 'idf1', 'num_switches', 'precision', 'recall'],
        name='Summary'
    )

    # 导出为 CSV (可以用 Excel 直接打开)
    csv_path = "tracking_report.csv"
    summary.to_csv(csv_path)


    print(f"\n--- 评估完成 ---")
    print(f"表格已导出至 CSV: {os.path.abspath(csv_path)}")
    print(summary)


if __name__ == "__main__":
    EVAL_GT_DIR = "F:/CVproject/VisDrone-MOT/val/annotations/"
    EVAL_RES_DIR = "F:/CVproject/report/mot_results/"  # 这是之前脚本生成 txt 的地方

    evaluate_mota(EVAL_GT_DIR, EVAL_RES_DIR)