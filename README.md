# 多模态 AIGC 感知与生成系统

## 📌 项目概述

本项目实现了一个端到端的**多模态感知与生成系统**，能够处理视频与音频数据，完成以下任务：

- 提取并索引视频帧与音频片段  
- 基于**视觉语言模型（Qwen-VL）** 对视频片段进行语义问答  
- 根据场景理解生成 **Stable Diffusion 提示词与图像**  
- 通过 **Flask Web API** 提供可视化结果服务  

系统整合了 **CLIP、Qwen-VL、ChromaDB 和 Stable Diffusion**，构建了一条从感知到生成的完整多模态智能分析流水线。

---

## Framework


![Framework](https://raw.githubusercontent.com/yuzz5/SenseRAG/main/framework/framework.png)

## 🧠 核心模块说明

### 1. **帧特征索引（`frame.py`）**
- 从 `/frames/{标签}_{视频ID}_...` 目录读取视频帧
- 使用 **CLIP-ViT** 提取图像-文本联合嵌入向量
- 将嵌入向量与元数据（包括 `clip_id`、时间戳、标签等）存入 **ChromaDB 向量数据库**
- 为每个视频片段保存**最后一帧**至 `chroma_db/last_frames/`，供后续使用

> 支持两种目录命名格式：
> - `Chainsaw_0M3AYnUPk4g_0.00_10.00`
> - `Traffic_accident_LQTQt2-QzGU_0208_5.00_10.00`

### 2. **音频索引（`index_audio_files.py`）**
- 扫描 `audio/clips_wav/` 目录下的 `.wav`、`.mp3` 等音频文件
- 使用 **torchaudio** 提取时长、采样率、文件大小等元数据
- 将信息存入 **SQLite 数据库**（`audio_index.db`），并建立索引以支持高效查询

### 3. **多模态问答与推理（`query.py`）**
- 从 `questions_qwen_vl.jsonl` 加载问题列表
- 对每个视频片段（`video_id`）：
  - 使用 **CLIP 文本编码器** 对问题进行嵌入
  - 在 ChromaDB 中**仅检索当前视频片段内的相关帧**
  - 将检索到的帧与问题输入 **Qwen-VL 模型** 进行视觉问答
- 输出结构化答案至 `batch_answers.jsonl`

### 4. **结果聚合（`main.py`）**
- 读取 `batch_answers.jsonl`
- 为每个 `video_id` 匹配：
  - 原始最后一帧（`chroma_db/last_frames/{video_id}.jpg`）
  - 生成的 SD 图像（`generated_from_last_frames_final/gen_{video_id}/`）
- 从帧目录名中解析时间戳（起止时间）
- 生成最终汇总文件 `final_summary.jsonl`

### 5. **Web API 服务（`app.py`）**
- 通过 Flask 启动 Web 服务（默认端口 `7860`）
- 提供以下接口：
  - `GET /` → 返回前端页面 `index.html`
  - `GET /api/result?video_id=xxx` → 返回 JSON 结果，包含：
    - 问答对
    - SD 提示词（来自第3个问题）
    - 原始帧与生成图像的 URL
  - `/sd_images/...` 与 `/original_frame/...` → 静态图像资源服务

---

## 📂 项目结构

```
Multimodal_AIGC/
├── frames/                          # 输入：视频帧（按片段组织）
├── audio/clips_wav/                 # 输入：音频片段
├── chroma_db/
│   ├── copied_frames/               # 处理后的帧副本
│   └── last_frames/                 # 每个片段的最后一帧（如 `xxx_last_frame.jpg`）
├── generated_from_last_frames_final/ # 输出：Stable Diffusion 生成图像
├── models/
│   ├── openai/clip-vit-base-patch32/
├── labels.txt                       # 帧级标签（可选）
├── questions_qwen_vl.jsonl          # 输入：每个 video_id 的问题列表
├── batch_answers.jsonl              # 中间结果：Qwen-VL 问答输出
├── final_summary.jsonl              # 最终结果：聚合数据
├── audio_index.db                   # 音频元数据索引
├── frame.py                         # 帧索引与 ChromaDB 构建
├── index_audio_files.py             # 音频索引
├── query.py                         # 多模态问答
├── main.py                          # 结果聚合
└── app.py                           # Web API 服务
```

---

## ⚙️ 部署与使用

### 环境要求
- Python ≥ 3.8
- CUDA（推荐 GPU 加速）
- 预下载模型至指定路径

### 安装依赖
```bash
pip install flask torch torchvision transformers chromadb torchaudio pillow modelscope
```

### 运行流程

1. **构建帧向量库**
   ```bash
   python frame.py
   ```

2. **构建音频索引（可选）**
   ```bash
   python index_audio_files.py
   ```

3. **执行多模态问答**
   ```bash
   python query.py
   ```

4. **聚合最终结果**
   ```bash
   python main.py
   ```

5. **启动 Web 服务**
   ```bash
   python app.py
   ```
   然后访问：`http://localhost:7860?video_id=你的视频ID`

---

## 

---

## 📝 注意事项

- 系统假设每个 `video_id` 对应一个**唯一的视频片段**（可能带后缀，如 `_0208`）
- `questions_qwen_vl.jsonl` 中的**第3个问题**专用于生成 **Stable Diffusion 提示词**
- 所有路径默认为 `/root/autodl-tmp/Multimodal_AIGC/`，如需修改请调整 `PROJECT_ROOT`
- 生成的图像需手动或通过外部流程放入 `generated_from_last_frames_final/gen_{video_id}/` 目录



## 🙌 致谢

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 阿里通义千问团队  
- [CLIP](https://github.com/openai/clip) — OpenAI  
- [ChromaDB](https://www.trychroma.com/) — 向量数据库  
- [ModelScope](https://modelscope.cn/) — 魔搭模型开放平台  

---


> ✨ 本系统实现了 **感知 → 推理 → 生成** 的多模态闭环，适用于视频理解、内容创作、AI辅助分析等场景。
