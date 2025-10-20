import cv2
import os

# ================== 配置参数 ==================
LABEL_FILE = './output_labels/test_correct_positive_labels.txt'
VIDEO_DIR = './data/AVE/AVE_Dataset/AVE_Dataset/new80video'      # 视频所在目录
OUTPUT_DIR = './extracted_frames'                               # 输出帧保存路径
TARGET_FRAMES_PER_CLIP = 16                                     # 每个片段提取多少帧
# =============================================

def extract_frames_from_video(video_path, start_sec, end_sec, output_folder, target_frames=16):
    """
    从视频的 [start_sec, end_sec] 区间均匀提取指定数量的帧
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # 边界检查
    if start_frame >= total_frames:
        print(f"[警告] 起始时间超出范围: {video_path} ({start_sec}s)")
        cap.release()
        return
    if end_frame < 0:
        print(f"[警告] 终止时间无效: {video_path} ({end_sec}s)")
        cap.release()
        return

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames - 1)

    if start_frame >= end_frame:
        print(f"[跳过] 无效时间区间: {start_frame} >= {end_frame}")
        cap.release()
        return

    # 均匀采样帧索引
    frame_indices = [
        int(start_frame + (end_frame - start_frame) * i / max(target_frames - 1, 1))
        for i in range(target_frames)
    ]
    frame_indices = sorted(set(frame_indices))  # 去重

    count = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            fname = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_folder, fname), frame)
            count += 1
        else:
            print(f"[警告] 读取帧失败: 第 {frame_idx} 帧 from {video_path}")

    cap.release()
    print(f"[完成] 从 {os.path.basename(video_path)} ({start_sec:.2f}s~{end_sec:.2f}s) 提取 {count} 帧 → {output_folder}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split('&')
        if len(parts) < 5:
            print(f"[格式错误] 行格式不完整: {line}")
            continue

        category = parts[0]
        video_id = parts[1]          # 完整视频ID，如 PBjKI8lrRws_0050
        # parts[2] == 'good' （可忽略）
        try:
            start_sec = float(parts[3])  # 起始时间（秒）
            end_sec = float(parts[4])    # 终止时间（秒）
        except ValueError:
            print(f"[解析失败] 起止时间非数字: {parts[3]}, {parts[4]} ← {line}")
            continue

        if start_sec >= end_sec:
            print(f"[跳过] 起始时间 >= 终止时间: {start_sec} >= {end_sec}")
            continue

        # 构造视频文件路径
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)

        if not os.path.exists(video_path):
            print(f"[缺失] 视频文件不存在: {video_path}")
            continue

        # 输出文件夹名：类别_视频ID_起始_终止
        clip_name = f"{category}_{video_id}_{start_sec:.2f}_{end_sec:.2f}"
        output_folder = os.path.join(
            OUTPUT_DIR,
            clip_name.replace(' ', '_').replace('/', '_').replace(',', '_')
        )
        os.makedirs(output_folder, exist_ok=True)

        # 执行抽帧
        extract_frames_from_video(
            video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            output_folder=output_folder,
            target_frames=TARGET_FRAMES_PER_CLIP
        )

    print(f"\n✅ 所有视频片段帧提取完成！结果保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()