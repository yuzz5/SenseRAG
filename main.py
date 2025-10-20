import os
import re
import json
import glob

PROJECT_ROOT = "/root/autodl-tmp/Multimodal_AIGC"
COPIED_FRAMES_ROOT = os.path.join(PROJECT_ROOT, "chroma_db", "copied_frames")
GENERATED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "generated_from_last_frames_final")
QUERY_OUTPUT = os.path.join(PROJECT_ROOT, "batch_answers.jsonl")
FINAL_SUMMARY = os.path.join(PROJECT_ROOT, "final_summary.jsonl")

def extract_timestamp_from_frames(video_id):
    """
    在 copied_frames 目录下找到与 video_id 匹配的目录名或文件名，解析时间戳
    """
    for d in os.listdir(COPIED_FRAMES_ROOT):
        if video_id in d:
            parts = d.split("_")
            if len(parts) >= 4:
                start, end = parts[-2], parts[-1]
                return start, end
    return None, None


def collect_images(video_id):
    safe_video_id = "".join(c if c.isalnum() or c in "-._" else "_" for c in video_id)

    candidates = []
    for d in os.listdir(GENERATED_IMAGES_DIR):
        if not d.startswith("gen_"):
            continue
        core = d[4:]
        if safe_video_id in core or core in safe_video_id:
            candidates.append(d)

    if not candidates:
        print(f"[WARN] 没有找到 SD 目录匹配 video_id={video_id}")
        return None, [], None, None

    candidates.sort(key=len)
    video_dir = os.path.join(GENERATED_IMAGES_DIR, candidates[0])
    files = glob.glob(os.path.join(video_dir, "*"))

    original_frame = None
    sd_images = []
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            if re.search(r"_\d+\.png$", f):
                sd_images.append(f)
            else:
                original_frame = f

    # 解析时间戳
    start_time, end_time = extract_timestamp_from_frames(video_id)

    print(f"[INFO] video_id={video_id} 使用目录 {video_dir}, start={start_time}, end={end_time}")
    return original_frame, sd_images, start_time, end_time


def main():
    if not os.path.exists(QUERY_OUTPUT):
        print(f"[ERROR] 没有找到 query 输出文件: {QUERY_OUTPUT}")
        return

    results = []
    with open(QUERY_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            video_id = data.get("video_id")
            answers = data.get("answers", [])

            sd_answer = None
            for ans in answers:
                if ans.get("question_index") == 3:
                    sd_answer = ans.get("answer")
                    break

            original_frame, sd_images, start_time, end_time = collect_images(video_id)

            result_entry = {
                "video_id": video_id,
                "label": data.get("label"),
                "sentence_answers": [
                    {"q": ans.get("question"), "a": ans.get("answer")}
                    for ans in answers
                ],
                "sd_answer": sd_answer,
                "original_frame": original_frame,
                "sd_images": sd_images,
                "start_time": start_time,
                "end_time": end_time
            }
            results.append(result_entry)

    with open(FINAL_SUMMARY, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 汇总完成，结果保存在 {FINAL_SUMMARY}")


if __name__ == "__main__":
    main()
