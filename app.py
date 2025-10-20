import os
import json
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SD 生成图像目录（保持不变）
SD_IMAGES_DIR = os.path.join(BASE_DIR, "generated_from_last_frames_final")

# ✅ 修正：原始帧在项目根目录下的 chroma_db/last_frames/
ORIGINAL_FRAMES_DIR = os.path.join(BASE_DIR, "chroma_db", "last_frames")

# 最终结果文件
FINAL_SUMMARY_PATH = os.path.join(BASE_DIR, "final_summary.jsonl")

summary_cache = {}

def load_summary():
    global summary_cache
    summary_cache = {}
    if os.path.exists(FINAL_SUMMARY_PATH):
        with open(FINAL_SUMMARY_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        vid = item.get("video_id")
                        if vid:
                            summary_cache[vid] = item
                    except Exception as e:
                        print(f"⚠️ JSONL 解析错误: {e}")
    print(f"✅ 加载 {len(summary_cache)} 条问答记录")

# ===== 路由 =====
@app.route('/')
def index():
    index_path = os.path.join(BASE_DIR, 'index.html')
    if not os.path.exists(index_path):
        return "❌ index.html 未找到", 500
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/result')
def get_result():
    video_id = request.args.get('video_id', '').strip()
    if not video_id:
        return jsonify({"error": "video_id 不能为空"}), 400

    record = summary_cache.get(video_id)
    if not record:
        return jsonify({"error": "未找到该 video_id"}), 404

    answers = [{"question": qa.get("q", ""), "answer": qa.get("a", "")} for qa in record.get("sentence_answers", [])]

    # SD 图像
    sd_image_urls = []
    gen_dir = os.path.join(SD_IMAGES_DIR, f"gen_{video_id}")
    if os.path.exists(gen_dir):
        for file in sorted(os.listdir(gen_dir)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sd_image_urls.append(f"/sd_images/gen_{video_id}/{file}")

    # 原始帧：在 chroma_db/last_frames/ 下查找
    original_frame_url = None
    for ext in ['.jpg', '.jpeg', '.png']:
        if os.path.exists(os.path.join(ORIGINAL_FRAMES_DIR, f"{video_id}{ext}")):
            original_frame_url = f"/original_frame/{video_id}{ext}"
            break

    return jsonify({
        "video_id": video_id,
        "label": record.get("label", "Unknown"),
        "answers": answers,
        "sd_prompt": record.get("sd_answer", ""),
        "sd_image_urls": sd_image_urls,
        "original_frame_url": original_frame_url
    })

@app.route('/sd_images/<path:filename>')
def serve_sd_image(filename):
    if ".." in filename or filename.startswith("/"):
        return "非法路径", 403
    full_path = os.path.join(SD_IMAGES_DIR, filename)
    if not os.path.exists(full_path):
        return "SD 图像未找到", 404
    return send_from_directory(SD_IMAGES_DIR, filename)

@app.route('/original_frame/<filename>')
def serve_original_frame(filename):
    if ".." in filename or filename.startswith("/"):
        return "非法路径", 403
    full_path = os.path.join(ORIGINAL_FRAMES_DIR, filename)
    if not os.path.exists(full_path):
        print(f"❌ 原始帧不存在: {full_path}")
        return "原始帧未找到", 404
    return send_from_directory(ORIGINAL_FRAMES_DIR, filename)

# ===== 启动 =====
if __name__ == '__main__':
    print("🔧 项目根目录:", BASE_DIR)
    print("📄 index.html 存在?", os.path.exists(os.path.join(BASE_DIR, 'index.html')))
    print("📁 SD 图像目录存在?", os.path.exists(SD_IMAGES_DIR))
    print("📁 原始帧目录存在?", os.path.exists(ORIGINAL_FRAMES_DIR))
    print("📄 final_summary.jsonl 存在?", os.path.exists(FINAL_SUMMARY_PATH))

    load_summary()

    print("\n✅ 启动成功！请访问: http://localhost:7860\n")
    app.run(host='0.0.0.0', port=7860, debug=True)