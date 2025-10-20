import os
import re
import requests
import base64
import shutil

# --- 配置 ---
SD_PROMPTS_DIR = "/root/autodl-tmp/Multimodal_AIGC/sd_prompts"
LAST_FRAMES_DIR = "/root/autodl-tmp/Multimodal_AIGC/chroma_db/last_frames"  # 直接使用存放最后一帧的目录
GENERATED_IMAGES_DIR = "/root/autodl-tmp/Multimodal_AIGC/generated_from_last_frames_final"
SD_API_URL = "http://127.0.0.1:7860/"  # Stable Diffusion WebUI API 地址

os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROMPT_FILE_PATTERN = re.compile(r"^(.+)_sd_prompt\.txt$")


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
    return f"data:image/{mime};base64,{img_b64}"


def find_last_frame(video_id: str):
    """
    在 LAST_FRAMES_DIR 中查找名为 {video_id}_last_frame.jpg 的文件
    """
    # 构造预期的文件名
    expected_filename = f"{video_id}_last_frame.jpg"
    expected_path = os.path.join(LAST_FRAMES_DIR, expected_filename)
    
    # 检查是否存在精确匹配的文件
    if os.path.exists(expected_path):
        print(f"[DEBUG] 找到精确匹配的最后帧: {expected_path}")
        return expected_path
    else:
        print(f"[WARN] 找不到 video_id={video_id} 的最后一帧文件 (期望: {expected_path})")
        return None


def call_sd_img2img_api(init_image_b64: str, prompt: str, output_dir: str, output_filename_base: str):
    url = f"{SD_API_URL}/sdapi/v1/img2img"
    negative_prompt = (
        "safe riding, normal riding, standing still, blurry, low resolution, "
        "cartoon, anime, painting, illustration, abstract, unrealistic, "
        "extra limbs, missing limbs, broken anatomy, disfigured face, "
        "mutated hands, bad proportions, bad anatomy, out of frame, "
        "duplicate rider, duplicate motorcycle, distorted helmet, "
        "distorted road, calm scene, peaceful, smiling"
    )

    controlnet_units = [{
        "input_image": init_image_b64,
        "mask": None,
        "module": "openpose_full",
        "model": "control_v11p_sd15_openpose [cab727d4]",
        "weight": 1.3,
        "resize_mode": "Crop and Resize",
        "lowvram": False,
        "processor_res": 512,
        "threshold_a": 100,
        "threshold_b": 200,
        "guidance_start": 1.0,
        "guidance_end": 1.0,
        "control_mode": "Balanced",
        "pixel_perfect": True,
    }]

    payload = {
        "init_images": [init_image_b64],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 25,
        "cfg_scale": 14,
        "denoising_strength": 0.55,
        "seed": -1,
        "sampler_name": "DPM++ 2M",
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "n_iter": 5,
        "alwayson_scripts": {"ControlNet": {"args": controlnet_units}},
    }

    print(f"  调用 SD API 生成图片 ({output_filename_base})...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            r = response.json()
            for i, img_str in enumerate(r["images"]):
                img_data_b64 = img_str.split(",", 1)[1] if img_str.startswith("data:image") else img_str
                image_data = base64.b64decode(img_data_b64)
                output_path = os.path.join(output_dir, f"{output_filename_base}_{i}.png")
                with open(output_path, "wb") as f:
                    f.write(image_data)
                print(f"  ✅ 图片已保存至: {output_path}")
        else:
            print(f"[ERROR] SD API 调用失败，状态码: {response.status_code}")
            print(f"  错误信息: {response.text}")
    except Exception as e:
        print(f"[ERROR] 调用 SD img2img API 时出错: {e}")


def main():
    print("--- 批量生成开始：使用每个 prompt 对应视频的最后一帧 ---")

    for fname in os.listdir(SD_PROMPTS_DIR):
        if not fname.endswith(".txt"):
            continue
        match = PROMPT_FILE_PATTERN.match(fname)
        if not match:
            continue

        video_id = match.group(1)
        prompt_file_path = os.path.join(SD_PROMPTS_DIR, fname)
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"[ERROR] 读取 {fname} 时出错: {e}")
            continue

        last_frame_path = find_last_frame(video_id)
        if not last_frame_path:
            continue

        print(f"\n--- 处理 video_id={video_id} ---")
        print(f"  选取最后一帧: {last_frame_path}")

        init_image_b64 = encode_image_to_base64(last_frame_path)
        if not init_image_b64:
            print(f"  [WARN] 无法编码图片，跳过")
            continue

        safe_video_id = re.sub(r"[^\w\-_.]", "_", video_id)
        output_dir = os.path.join(GENERATED_IMAGES_DIR, f"gen_{safe_video_id}")
        os.makedirs(output_dir, exist_ok=True)

        shutil.copy(last_frame_path, os.path.join(output_dir, os.path.basename(last_frame_path)))

        call_sd_img2img_api(init_image_b64, prompt, output_dir, safe_video_id)

    print("\n✅ 所有 prompt 已处理完成！")


if __name__ == "__main__":
    main()



