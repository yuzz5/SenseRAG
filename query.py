import os
import glob
import json
from typing import List, Dict, Any
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
import chromadb

# ========== 配置 ==========
PROJECT_ROOT = "/root/autodl-tmp/Multimodal_AIGC"
QUERY_FILE_PATH = os.path.join(PROJECT_ROOT, "questions_qwen_vl.jsonl")
LABELS_TXT_PATH = os.path.join(PROJECT_ROOT, "labels.txt")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
CHROMA_COLLECTION_NAME = "video_frames"
QWEN_VL_MODEL_PATH = "/root/autodl-tmp/models/qwen-vl-chat-int4"
CLIP_MODEL_NAME = "/root/autodl-tmp/Multimodal_AIGC/models/openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3
FRAME_BASE_DIR = os.path.join(PROJECT_ROOT, "frames")
BATCH_OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "batch_answers.jsonl")
SD_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sd_prompts")
os.makedirs(SD_OUTPUT_DIR, exist_ok=True)
# =========================


# ================= MultimodalRAG =================
class MultimodalRAG:
    def __init__(self,
                 chroma_db_path: str,
                 chroma_collection_name: str,
                 qwen_vl_model_path: str,
                 clip_model_name: str,
                 device: str,
                 top_k: int = TOP_K):
        self.device = device
        self.top_k = top_k

        print("加载 Qwen-VL tokenizer & model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_vl_model_path, trust_remote_code=True)
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token or self.qwen_tokenizer.bos_token or ""
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_vl_model_path, trust_remote_code=True, device_map="auto"
        ).eval()
        for p in self.qwen_model.parameters():
            p.requires_grad_(False)
        print("Qwen-VL 加载完成。")

        print("加载 CLIP processor & model...")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        print("CLIP 加载完成。")

        print("连接 Chroma 数据库...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        try:
            self.chroma_collection = self.chroma_client.get_collection(name=chroma_collection_name)
        except Exception:
            self.chroma_collection = self.chroma_client.create_collection(name=chroma_collection_name)
        print("Chroma 连接成功。")

    # 文本嵌入
    def embed_text(self, text: str) -> List[float]:
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.get_text_features(**inputs)
        feats_norm = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats_norm.cpu().numpy().flatten().tolist()

    # 检索相似帧（限定当前 video_id）
    def search_similar_frames(self, query_embedding: List[float], video_id: str) -> List[Dict[str, Any]]:
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            where={"video_id": video_id},   # ✅ 限制只查当前视频
            include=['metadatas', 'distances']
        )
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        out = []
        for i in range(len(ids)):
            md = metadatas[i] if i < len(metadatas) else {}
            out.append({
                'id': ids[i],
                'metadata': md,
                'distance': distances[i] if i < len(distances) else None,
                'frame_path': md.get('frame_file_path', 'N/A'),
                'labels': md.get('label', md.get('labels', 'N/A'))
            })
        return out

    # 构建上下文字符串（调试用）
    def build_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        parts = []
        for i, item in enumerate(retrieved_items):
            parts.append(
                f"检索到的相关视频帧 {i+1}:\n  - 标签: {item.get('labels','N/A')}\n  - 相似度距离: {item.get('distance', 'N/A'):.4f}"
            )
        return "\n".join(parts)

    # 处理一条记录
    def run_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        video_id = record.get('video_id', 'unknown')
        label = record.get('label', '')

        # 搜索包含 video_id 的帧目录
        frame_dirs = glob.glob(os.path.join(FRAME_BASE_DIR, f"*_{video_id}_*"))
        all_frames = []
        if frame_dirs:
            frame_dir = frame_dirs[0]
            all_frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

        questions = record.get('questions', [])
        answers = []

        # ✅ 记录实际用于视觉输入的帧路径（去重）
        used_frames_for_vqa = set()

        for q_idx, qobj in enumerate(questions):
            question_text = qobj.get('question') if isinstance(qobj, dict) else str(qobj)
            query_emb = self.embed_text(question_text)
            retrieved_items = self.search_similar_frames(query_emb, video_id)   # ✅ 限定当前视频
            context_text = self.build_context(retrieved_items)

            if q_idx in [0, 1, 2]:
                if retrieved_items:  # ✅ 使用检索到的帧
                    try:
                        input_frames = []
                        for item in retrieved_items:
                            frame_path = item['frame_path']
                            if frame_path and os.path.exists(frame_path):
                                input_frames.append(frame_path)
                                used_frames_for_vqa.add(frame_path)

                        query_list = []
                        for f in input_frames:
                            query_list.append({'image': f})
                        query_list.append({'text': f"问题: {question_text}\n请基于这些检索到的帧进行详细回答。"})

                        query_formatted = self.qwen_tokenizer.from_list_format(query_list)
                        frame_ans = self.qwen_model.chat(self.qwen_tokenizer, query=query_formatted, history=None)[0]
                        answer_text = frame_ans
                    except Exception as e:
                        print(f"[WARN] Qwen 回答失败: {e}")
                        answer_text = "回答失败。"
                else:
                    answer_text = "未找到相关检索帧，无法回答。"
            else:
                # 第四题：生成 SD Prompt，使用最后一帧
                last_frame = all_frames[-1] if all_frames else None
                if last_frame and os.path.exists(last_frame):
                    used_frames_for_vqa.add(last_frame)

                    instruction_and_question = f"""You are a professional Stable Diffusion prompt engineer.
Your task is to generate a photorealistic **dangerous scene description** based **only on the last frame of the video**.

Context:
- Scene Label: {label}
- Retrieved Labels: {', '.join(set([it.get('labels','') for it in retrieved_items if it.get('labels')])) or label}
- Last Frame: {os.path.basename(last_frame)}

Rules:
1. **Do NOT change the identity or appearance of any person or object**.
2. Only change the **action or physical state** to a dangerous one (e.g., slipping, falling, crashing).
3. Environment, objects, and characters remain unchanged.
4. Include cinematic effects like motion blur, sparks, dust, shadows if relevant.
5. Generate only the **Stable Diffusion prompt** starting with 'A photorealistic ...'.

Output format:
A photorealistic ..."""

                    query_list = [{'image': last_frame}, {'text': instruction_and_question.strip()}]
                    query_formatted = self.qwen_tokenizer.from_list_format(query_list)
                    try:
                        answer_text = self.qwen_model.chat(self.qwen_tokenizer, query=query_formatted, history=None)[0]

                        # 保存 SD Prompt 到文件
                        sd_prompt_path = os.path.join(SD_OUTPUT_DIR, f"{video_id}_sd_prompt.txt")
                        with open(sd_prompt_path, 'w', encoding='utf-8') as f:
                            f.write(answer_text)
                    except Exception as e:
                        print(f"[WARN] 生成或保存 SD prompt 失败: {e}")
                        answer_text = "Failed to generate SD prompt."
                else:
                    answer_text = "No frame available for SD prompt generation."

            answers.append({
                'question_index': q_idx,
                'question': question_text,
                'answer': answer_text
            })

        used_frame_basenames = sorted([os.path.basename(f) for f in used_frames_for_vqa])

        return {
            'video_id': video_id,
            'label': label,
            'used_frames_for_vqa': used_frame_basenames,
            'answers': answers
        }

# ================= 工具函数 =================
def load_records_from_json(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"错误: 找不到文件 {path}")
        return records
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"警告: 无法解析行（跳过）: {line[:200]}")
    except Exception as e:
        print(f"读取 {path} 时出错: {e}")
    print(f"[INFO] 从 {path} 加载到 {len(records)} 条记录。")
    return records

def load_labels(labels_txt_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.exists(labels_txt_path):
        print(f"[INFO] labels 文件不存在: {labels_txt_path}")
        return mapping
    try:
        with open(labels_txt_path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split('&')
                first = parts[0].strip() if parts else ""
                vid = parts[1].strip() if len(parts)>=2 else ""
                label_main = first.split(',')[0].strip() if first else ""
                if vid:
                    mapping[vid] = label_main or first
    except Exception as e:
        print(f"[WARN] 读取 labels.txt 失败: {e}")
    print(f"[INFO] 解析 labels.txt，得到 {len(mapping)} 个 video_id->label 映射。")
    return mapping

# ================= main =================
def main():
    print("初始化 Multimodal RAG（批处理版本）...")
    try:
        rag = MultimodalRAG(
            chroma_db_path=CHROMA_DB_PATH,
            chroma_collection_name=CHROMA_COLLECTION_NAME,
            qwen_vl_model_path=QWEN_VL_MODEL_PATH,
            clip_model_name=CLIP_MODEL_NAME,
            device=DEVICE,
            top_k=TOP_K
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    label_map = load_labels(LABELS_TXT_PATH)
    records = load_records_from_json(QUERY_FILE_PATH)
    if not records:
        print("没有要处理的记录，退出。")
        return

    with open(BATCH_OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for idx, rec in enumerate(records):
            video_id = rec.get('video_id', '')
            if video_id and video_id in label_map:
                rec['label'] = label_map[video_id]
            print(f"\n--- 处理记录 {idx+1}/{len(records)}: video_id={video_id} ---")
            try:
                result = rag.run_record(rec)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"[ERROR] 处理 record 时发生异常: {e}")

    print(f"\n批处理完成，输出保存在: {BATCH_OUTPUT_JSONL}")
    print(f"第四题（SD prompts）单文件保存在目录: {SD_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
