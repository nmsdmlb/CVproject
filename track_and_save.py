
import os

from ultralytics import YOLO

unique_ids = set()
enter_ids = set()
cross_ids = set()

# 方向计数统计
in_count = 0  # 左 → 右（进入）
out_count = 0 # 右 → 左（离开）
track_history = dict() # 记录目标历史位置，判断方向
has_counted = dict()
def track_visdrone(model_path, seq_dir, save_dir="mot_results"):
    """
    运行VisDrone图片序列跟踪，并保存MOT标准格式结果
    :param model_path: 训练好的best.pt路径
    :param seq_dir: 图片序列文件夹路径
    :param save_dir: 结果保存目录
    """
    # 初始化模型
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)

    # 遍历所有图片序列
    for seq_name in os.listdir(seq_dir):
        seq_path = os.path.join(seq_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue

        print(f"正在跟踪序列：{seq_name}")
        # 运行ByteTrack跟踪
        results = model.track(
            source=seq_path,
            tracker="bytetrack.yaml",
            conf=0.1,  # 调高置信度，减少误检
            imgsz=1280,  # 重要：大幅提升分辨率（须为32的倍数，如960, 1088, 1280）
            persist=True,  # 持续跟踪，减少ID跳变
            iou=0.45,  # 解决密集目标重叠
            workers=0,  # 避免Windows多进程问题
            save=True  # 保存跟踪后的图片/视频
        )

        # 保存MOT Challenge标准格式结果
        save_file = os.path.join(save_dir, f"{seq_name}.txt")
        with open(save_file, "w", encoding="utf-8") as f:
            frame_id = 1
            for r in results:
                if r.boxes.id is None:
                    frame_id += 1
                    continue
                # 解析每帧的跟踪结果
                boxes = r.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
                track_ids = r.boxes.id.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()

                # 按MOT格式写入：帧号,ID,x1,y1,w,h,score,-1,-1,-1
                for box, tid, score in zip(boxes, track_ids, scores):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    #工业功能 1：目标计数 + 禁区报警
                    # ==========================
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    tid = int(tid)

                    unique_ids.add(tid)
                    if tid not in enter_ids:
                        enter_ids.add(tid)
                        print(f"[统计] 目标 {tid} 进入画面")

                    # 禁区（可自己改坐标）
                    danger_zone = [150, 150, 500, 500]
                    if danger_zone[0] < cx < danger_zone[2] and danger_zone[1] < cy < danger_zone[2]:
                        if tid not in cross_ids:
                            cross_ids.add(tid)
                            print(f"[⚠️ 报警] 目标 {tid} 闯入禁区！")
                    # 工业功能 2：分方向进出统计（新功能！）
                    # ==========================
                    if tid not in track_history:
                        track_history[tid] = []
                    track_history[tid].append(cx)

                    if len(track_history[tid]) > 10:
                        track_history[tid].pop(0)

                    if len(track_history[tid]) >= 5:
                        first_x = track_history[tid][0]
                        last_x = track_history[tid][-1]
                        move_x = last_x - first_x

                        if move_x > 40:
                            if not has_counted.get(tid, False):
                                in_count += 1
                                has_counted[tid] = True
                                print(f"[✅ 进入] 目标 {tid} | 总进入：{in_count}")
                        elif move_x < -40:
                            if not has_counted.get(tid, False):
                                out_count += 1
                                has_counted[tid] = True
                                print(f"[❌ 离开] 目标 {tid} | 总离开：{out_count}")
                    #工业功能结束
                    # ==========================
                    f.write(f"{frame_id},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.2f},-1,-1,-1\n")

                frame_id += 1

    print(f"所有跟踪结果已保存到：{os.path.abspath(save_dir)}")


# 训练完成后，替换成你的实际路径再运行
if __name__ == "__main__":
    track_visdrone(
        model_path="F:/CVproject/ultralytics_old/train/runs/detect/train5/weights/best.pt",  # 训练完后改这里的X
        seq_dir="F:/CVproject/VisDrone-MOT/val/sequences/"
    )
#该代码的目的是追踪整个验证集的sequences，给里面的每个序列的每一张图片都打上ID