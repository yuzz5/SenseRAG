import numpy as np
import resampy
from scipy.io import wavfile
import tensorflow as tf
import vggish_params
import vggish_slim
import os
from vggish_input import waveform_to_examples


# --- 修正后的正确实现 ---
# 正确实现：将音频文件转换为VGGish模型所需的log-mel频谱特征，使用标准0.96秒分帧
def correct_wavfile_to_examples(wav_file):
    """正确的分帧实现：按 0.96s 自动分帧"""
    sr, wav_data = wavfile.read(wav_file)  # 读取WAV文件并验证数据类型为16位整数
    assert wav_data.dtype == np.int16, f"Bad sample type: {wav_data.dtype}"
    wav_data = wav_data / 32768.0  # 归一化到 [-1.0, +1.0]将音频数据归一化到[-1.0, +1.0]范围

    # 强制采样率为 16kHz
    if sr != vggish_params.SAMPLE_RATE:  # 如果采样率不是16kHz，使用resampy库进行重采样
        wav_data = resampy.resample(wav_data, sr, vggish_params.SAMPLE_RATE)
        sr = vggish_params.SAMPLE_RATE

    # 调用标准分帧函数，调用VGGish的标准分帧函数waveform_to_examples，自动按0.96秒分帧
    return waveform_to_examples(wav_data, sr)


# --- 原始错误实现（硬编码 1秒分段）---
def incorrect_wavfile_to_examples(wav_file):
    """错误的分帧实现：硬编码 1秒分段"""
    sr, wav_data = wavfile.read(wav_file)
    wav_data = wav_data / 32768.0
    T = 10  # 硬编码 10 段
    log_mel = np.zeros([T, 96, 64])
    for i in range(T):
        s = i * sr  # 按 1秒分段
        e = (i + 1) * sr
        data = wav_data[s:e]
        log_mel[i] = waveform_to_examples(data, sr)
    return log_mel


# --- 测试用例 ---
def test_audio_framing():
    # 生成一个 10秒的测试音频（静音）
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz 正弦波
    audio = (audio * 32767).astype(np.int16)
    wavfile.write("test.wav", sr, audio)

    # 测试1：输出形状验证
    # 生成10秒测试音频(440Hz正弦波)
    # 比较正确和错误实现的输出形状
    # 预期形状都是(10, 96, 64)
    log_mel_correct = correct_wavfile_to_examples("test.wav")
    log_mel_incorrect = incorrect_wavfile_to_examples("test.wav")
    print(f"修正后输出形状: {log_mel_correct.shape}")  # 预期: (10, 96, 64)
    print(f"错误实现输出形状: {log_mel_incorrect.shape}")  # 预期: (10, 96, 64)

    # 测试2：分帧连续性验证
    # 检查分帧时间是否连续
    # 计算每帧的开始和结束时间，确保误差在1%以内
    frame_rate = 1 / vggish_params.STFT_HOP_LENGTH_SECONDS  # 100Hz
    time_per_example = vggish_params.NUM_FRAMES / frame_rate  # 0.96s

    for i in range(log_mel_correct.shape[0] - 1):
        # 修正变量名和计算逻辑
        end_time_i = (i + 1) * time_per_example
        start_time_next = (i + 1) * time_per_example  # 下一个片段的开始时间

        # 允许1%的误差容忍
        if not np.isclose(end_time_i, start_time_next, atol=0.01):
            print(f"片段{i}结束时间: {end_time_i:.3f}s, 片段{i + 1}开始时间: {start_time_next:.3f}s")
            raise AssertionError("分帧不连续")

    print("分帧连续性测试通过")

    # 测试3：频谱内容对比
    assert not np.allclose(log_mel_correct, log_mel_incorrect), "频谱内容不应相同"
    print("频谱内容对比测试通过")

    # 测试4：模型兼容性测试
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, "vggish_model.ckpt")
        features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
        embedding_tensor = sess.graph.get_tensor_by_name("vggish/embedding:0")
        embeddings = sess.run(embedding_tensor, feed_dict={features_tensor: log_mel_correct})
        print(f"嵌入形状: {embeddings.shape}")  # 预期: (10, 128)
        assert embeddings.shape == (log_mel_correct.shape[0], 128), "模型兼容性失败"
    print("模型兼容性测试通过")

    # 测试5：边缘案例测试（短音频）
    short_audio = np.zeros(int(sr * 0.5), dtype=np.int16)  # 0.5秒静音
    wavfile.write("short.wav", sr, short_audio)
    short_log_mel = correct_wavfile_to_examples("short.wav")
    print(f"短音频输出形状: {short_log_mel.shape}")  # 预期: (0, 96, 64)
    assert short_log_mel.shape[0] == 0, "短音频处理错误"

    print("所有测试通过！")


if __name__ == "__main__":
    test_audio_framing()