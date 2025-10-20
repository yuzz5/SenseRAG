# frame.py - 构建视频帧向量数据库（以 clip_id 为主键）

import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import chromadb
import re

# --- 配置 ---
root_frame_dir = "/root/autodl-tmp/Multimodal_AIGC/frames"
output_dir = "/root/autodl-tmp/Multimodal_AIGC/chroma_db"
label_file_path = "/root/autodl-tmp/Multimodal_AIGC/labels.txt"
os.makedirs(output_dir, exist_ok=True)

# 输出子目录
copied_frames_base_dir = os.path.join(output_dir, "copied_frames")
os.makedirs(copied_frames_base_dir, exist_ok=True)
last_frames_dir = os.path.join(output_dir, "last_frames")
os.makedirs(last_frames_dir, exist_ok=True)
collection_name = "video_frames"

# --- 初始化模型 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/root/autodl-tmp/Multimodal_AIGC/models/openai/clip-vit-base-patch32"
print("🚀 正在加载 CLIP 模型...")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print("✅ 模型加载完成")

# --- 初始化 ChromaDB ---
chroma_client = chromadb.PersistentClient(path=output_dir)

# 👇 安全删除旧集合
try:
    existing_collections = chroma_client.list_collections()
    if collection_name in [c.name for c in existing_collections]:
        chroma_client.delete_collection(name=collection_name)
        print(f"🗑️ 已删除旧集合 '{collection_name}'")
    else:
        print(f"ℹ️ 集合 '{collection_name}' 不存在，将新建")
except Exception as e:
    print(f"⚠️ 删除集合时出错（可忽略）: {e}")

# 创建新集合
try:
    chroma_collection = chroma_client.create_collection(name=collection_name)
    print(f"✅ 成功创建新集合 '{collection_name}'")
except Exception as e:
    print(f"❌ 创建集合失败: {e}")
    raise

# --- 存储列表 ---
all_embeddings = []
all_frame_ids = []
all_metadatas = []

# --- 辅助函数：从文件名提取帧序号 ---
def extract_frame_info(filename):
    try:
        name_part, ext = os.path.splitext(filename)
        if name_part.startswith("frame_"):
            number_str = name_part[6:]
            if number_str.isdigit():
                return int(number_str)
        return None
    except:
        return None

# --- 辅助函数：解析子目录名（支持两种格式）---
def extract_subfolder_info(subfolder_name):
    """
    支持两种格式：
    1. {label}_{video_id}_{start}_{end}
       → Chainsaw_0M3AYnUPk4g_0.00_10.00
    2. {label}_{video_id}_suffix_{start}_{end}
       → Traffic_accident_LQTQt2-QzGU_0208_5.00_10.00

    返回: (label, clip_id, start, end, original_video_id)
    """
    # 模式1: label_videoId_start_end
    pattern_simple = r'^([a-zA-Z_]+)_([a-zA-Z0-9_-]{11})_(\d+\.?\d*)_([\d\.]+)$'

    # 模式2: label_videoId_suffix_start_end
    pattern_complex = r'^([a-zA-Z_]+)_([a-zA-Z0-9_-]+)_(\d+)_(\d+\.?\d*)_(\d+\.?\d*)$'

    match_simple = re.match(pattern_simple, subfolder_name)
    if match_simple:
        label = match_simple.group(1)
        video_id = match_simple.group(2)  # 纯视频ID
        start = float(match_simple.group(3))
        end = float(match_simple.group(4))
        clip_id = video_id  # 片段ID = 视频ID（无后缀）
        return label, clip_id, start, end, video_id

    match_complex = re.match(pattern_complex, subfolder_name)
    if match_complex:
        label = match_complex.group(1)
        original_video_id = match_complex.group(2)
        suffix_num = match_complex.group(3)
        start = float(match_complex.group(4))
        end = float(match_complex.group(5))
        clip_id = f"{original_video_id}_{suffix_num}"  # 完整片段ID
        return label, clip_id, start, end, original_video_id

    print(f"❌ 子目录格式不匹配: {subfolder_name}")
    return None

# --- 清理元数据 ---
def clean_metadata(metadata_dict):
    return {k: v for k, v in metadata_dict.items() if v is not None}

# --- 加载标签文件 ---
def load_labels_from_txt(label_file):
    labels = {}
    if not os.path.exists(label_file):
        print(f"🟡 标签文件不存在: {label_file}")
        return labels

    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '&' in line:
                    parts = line.split('&')
                    if len(parts) >= 3:
                        frame_filename = parts[0].strip()
                        video_id = parts[1].strip()  # 注意：这里可能不准确，我们以 clip_id 为准
                        label = parts[2].strip()
                        start_second = None
                        end_second = None
                        if len(parts) >= 5:
                            try: start_second = float(parts[3].strip())
                            except: pass
                            try: end_second = float(parts[4].strip())
                            except: pass
                        labels[frame_filename] = {
                            "label": label,
                            "video_id": video_id,
                            "start_second": start_second,
                            "end_second": end_second
                        }
        print(f"✅ 成功加载 {len(labels)} 条标签数据")
    except Exception as e:
        print(f"❌ 读取标签文件失败: {e}")
    return labels

# --- 主处理逻辑 ---
labels_data = load_labels_from_txt(label_file_path)
last_frame_per_clip = {}  # 按 clip_id 保存最后一帧

if not os.path.exists(root_frame_dir):
    print(f"❌ 帧目录不存在: {root_frame_dir}")
else:
    subfolders = sorted([
        f for f in os.listdir(root_frame_dir)
        if os.path.isdir(os.path.join(root_frame_dir, f))
    ])

    print(f"🔍 发现 {len(subfolders)} 个子目录")

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_frame_dir, subfolder)
        subfolder_copied_dir = os.path.join(copied_frames_base_dir, subfolder)
        os.makedirs(subfolder_copied_dir, exist_ok=True)

        # 解析子目录
        result = extract_subfolder_info(subfolder)
        if result is None:
            continue
        label, clip_id, start, end, original_video_id = result

        print(f"\n📁 处理子目录: {subfolder}")
        print(f"   → label={label}, clip_id={clip_id}, original_video_id={original_video_id}, start={start}, end={end}")

        # 获取帧文件
        frame_files = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg")) and f.startswith("frame_")
        ])
        if not frame_files:
            print(f"   ⚠️ 无有效帧文件，跳过")
            continue

        print(f"   🖼️  共 {len(frame_files)} 帧")

        for filename in frame_files:
            path = os.path.join(subfolder_path, filename)
            try:
                # 复制帧
                copied_frame_path = os.path.join(subfolder_copied_dir, filename)
                if not os.path.exists(copied_frame_path):
                    shutil.copy2(path, copied_frame_path)

                # 解析帧索引
                frame_index = extract_frame_info(filename)
                if frame_index is None:
                    continue

                # 获取标签
                label_info = labels_data.get(filename, {"label": "unknown"})
                label_from_file = label_info["label"]

                # 图像加载与特征提取
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_feature = model.get_image_features(**inputs)
                image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)

                text_inputs = processor(text=label_from_file, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    text_feature = model.get_text_features(**text_inputs)
                text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

                # 融合特征
                combined_feature = (image_feature.cpu().numpy().flatten() + text_feature.cpu().numpy().flatten()) / 2.0
                all_embeddings.append(combined_feature)

                # 构建元数据：使用 clip_id 作为 video_id
                raw_metadata = {
                    "source_dir": subfolder,
                    "original_frame_file_path": path,
                    "frame_file_path": copied_frame_path,
                    "frame_index": frame_index,
                    "label": label_from_file,
                    "video_id": clip_id,                    # ✅ 用 clip_id 作为 video_id
                    "original_video_id": original_video_id, # 🔽 保留原始视频ID
                    "start_time": start,
                    "end_time": end,
                    "parsed_label": label,
                    "clip_id": clip_id                     # ✅ 冗余存储，便于查询
                }
                cleaned_metadata = clean_metadata(raw_metadata)

                # 生成唯一 ID
                unique_frame_id = f"{clip_id}_{filename}"
                all_frame_ids.append(unique_frame_id)
                all_metadatas.append(cleaned_metadata)

                # 更新最后一帧（按 clip_id 分组）
                if (clip_id not in last_frame_per_clip or 
                    frame_index > last_frame_per_clip[clip_id]['frame_index']):
                    last_frame_per_clip[clip_id] = {
                        'path': path,
                        'copied_path': copied_frame_path,
                        'frame_index': frame_index,
                        'metadata': cleaned_metadata
                    }

            except Exception as e:
                print(f"❌ 处理帧失败 {path}: {e}")

# --- 写入 ChromaDB ---
if all_embeddings:
    print(f"\n📦 正在批量写入 ChromaDB ({len(all_embeddings)} 条数据)...")

    batch_size = 100
    total_count = len(all_frame_ids)
    try:
        for i in range(0, total_count, batch_size):
            end_idx = min(i + batch_size, total_count)
            chroma_collection.add(
                embeddings=all_embeddings[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_frame_ids[i:end_idx]
            )
        print(f"✅ 成功写入 {total_count} 条数据到 ChromaDB")
    except Exception as e:
        print(f"❌ 写入 ChromaDB 失败: {e}")
        raise

    # 🔍 验证数据
    try:
        sample = chroma_collection.get(limit=3, include=["metadatas"])
        print("\n🔍 验证前3条数据:")
        for idx, meta in enumerate(sample['metadatas']):
            vid = meta.get("video_id", "MISSING")        # ← 这就是 clip_id
            orig = meta.get("original_video_id", "MISSING")
            print(f"  [{idx}] video_id={vid}, original_video_id={orig}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
else:
    print("❌ 没有生成任何 embedding。")

# --- 保存每个片段的最后一帧 ---
print(f"\n💾 保存每个片段的最后一帧到 {last_frames_dir}...")
for clip_id, info in last_frame_per_clip.items():
    src_path = info['path']
    dst_path = os.path.join(last_frames_dir, f"{clip_id}_last_frame.jpg")
    try:
        img = Image.open(src_path).convert("RGB")
        img.save(dst_path, "JPEG")
        print(f"  ✅ {clip_id} -> {dst_path}")
    except Exception as e:
        print(f"  ❌ 保存失败 {clip_id}: {e}")

print("\n🎉 ✅ 数据库构建完成！")