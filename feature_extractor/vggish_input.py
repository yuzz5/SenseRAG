# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy
from scipy.io import wavfile

import mel_features
import vggish_params


def waveform_to_examples(data, sample_rate): # 这行代码定义了一个名为 waveform_to_examples 的函数，它接受两个参数：
                                             # data：一个 numpy 数组，代表音频波形数据。sample_rate：音频数据的采样率，即每秒采集的样本数。
  """Converts audio waveform into an array of examples for VGGish.描述函数的功能。这里说明该函数的作用是将音频波形转换为适用于 VGGish 模型的示例数组。

  Args:详细描述了 data 参数：它是一个 numpy 数组，可以是一维（单声道）或二维（多声道）。通常，每个样本的值应该在 [-1.0, +1.0] 范围内，但这不是强制要求。
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:这部分文档字符串开始描述函数的返回值。一个三维的 numpy 数组，形状为 [num_examples, num_frames, num_bands]。这个数组表示一系列的示例，每个示例包含一个对数梅尔频谱图的片段，覆盖了 num_frames 帧的音频和 num_bands 个梅尔频率带。帧的长度由 vggish_params.STFT_HOP_LENGTH_SECONDS 决定。
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.把多声道音频转换为单声道音频。
  if len(data.shape) > 1: # 这是一个条件判断语句，用于检查输入的音频数据 data 是否为多声道音频。data.shape 返回一个元组，该元组表示 data 数组的维度信息。
                          # 例如，如果 data 是一个单声道音频，其形状可能是 (N,)，其中 N 是音频样本的数量；如果是多声道音频，其形状可能是 (N, C)，其中 C 是声道的数量。len(data.shape) 用于获取维度的数量，当 len(data.shape) > 1 时，说明 data 是多声道音频，此时需要进行转换。
    data = np.mean(data, axis=1) # 如果 data 是多声道音频，这行代码将对每个时间点上的所有声道的值求平均值，从而将多声道音频转换为单声道音频。np.mean 是 numpy 库中的函数，用于计算数组的平均值。axis=1 表示沿着第二个维度（即声道维度）进行求平均操作。
                                 # 例如，如果 data 的形状是 (N, C)，经过 np.mean(data, axis=1) 操作后，返回的数组形状将变为 (N,)，即单声道音频数据。
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE: # 这是一个条件判断语句，用于检查输入音频数据的采样率 sample_rate 是否与 VGGish 模型所假定的采样率 vggish_params.SAMPLE_RATE 相同。vggish_params 是一个自定义的参数模块，其中定义了 VGGish 模型在处理音频时所使用的各种参数，SAMPLE_RATE 即为该模型假定的采样率。如果输入音频的采样率与模型假定的采样率不一致，就需要进行重采样操作。
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE) # 如果上述条件判断为 True，则执行这行代码。resampy.resample 是 resampy 库中的一个函数，用于对音频数据进行重采样。该函数接受三个主要参数：
                                                                          # data：表示输入的音频数据，是一个 numpy 数组。
                                                                          # sample_rate：输入音频数据的原始采样率。
                                                                          # vggish_params.SAMPLE_RATE：目标采样率，即 VGGish 模型所假定的采样率。
                                                                          # 函数会根据原始采样率和目标采样率，对输入的音频数据进行插值或抽取操作，从而将音频数据的采样率转换为目标采样率。最后，将重采样后的音频数据重新赋值给 data 变量。

  # Compute log mel spectrogram features.计算音频数据的对数梅尔频谱（log mel spectrogram）特征
  log_mel = mel_features.log_mel_spectrogram( # 调用 mel_features 模块中的 log_mel_spectrogram 函数，并将函数的返回值赋值给变量 log_mel。mel_features 模块应该是自定义的一个用于音频特征提取的模块。
      data, # data 是输入的音频数据，是一个 numpy 数组。在调用这个函数之前，代码已经对音频数据进行了处理，如将多声道音频转换为单声道，以及重采样到 VGGish 模型所假定的采样率。
      audio_sample_rate=vggish_params.SAMPLE_RATE, # audio_sample_rate 是音频数据的采样率，这里使用了 vggish_params 模块中定义的 SAMPLE_RATE，即 VGGish 模型所假定的采样率。
      log_offset=vggish_params.LOG_OFFSET, # log_offset 是一个偏移量，用于在计算对数频谱时避免对零值取对数。这个值同样是从 vggish_params 模块中获取的。
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS, # window_length_secs 是短时傅里叶变换（STFT）中每个窗口的长度（以秒为单位）。在计算频谱时，需要将音频数据分割成多个窗口进行处理，这个参数决定了每个窗口的时长。
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS, # hop_length_secs 是相邻两个窗口之间的跳跃长度（以秒为单位）。它决定了窗口在音频数据上滑动的步长。
      num_mel_bins=vggish_params.NUM_MEL_BINS, # num_mel_bins 是梅尔滤波器组中的滤波器数量，也就是梅尔频谱中的频率带数量。
      lower_edge_hertz=vggish_params.MEL_MIN_HZ, # lower_edge_hertz 是梅尔滤波器组的最低频率（以赫兹为单位）。
      upper_edge_hertz=vggish_params.MEL_MAX_HZ) # upper_edge_hertz 是梅尔滤波器组的最高频率（以赫兹为单位）。

  # Frame features into examples.这段代码的主要目的是将之前计算得到的对数梅尔频谱特征 log_mel 划分为多个固定长度的窗口，形成一个个独立的示例（examples），以满足 VGGish 模型的输入要求。
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS # vggish_params.STFT_HOP_LENGTH_SECONDS 是短时傅里叶变换（STFT）中相邻两个窗口之间的跳跃长度（以秒为单位）。
                                                                     #features_sample_rate 表示特征的采样率，通过取 STFT_HOP_LENGTH_SECONDS 的倒数得到。这个采样率反映了特征在时间上的分辨率，即每秒有多少个特征帧。
                                                                     #1/0.01=100
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate)) # vggish_params.EXAMPLE_WINDOW_SECONDS 是每个示例窗口的时长（以秒为单位）。
                                                                    # 通过将 EXAMPLE_WINDOW_SECONDS 乘以 features_sample_rate，可以得到每个示例窗口包含的特征帧数。
                                                                    # round 函数用于对结果进行四舍五入，int 函数将结果转换为整数类型。最终得到的 example_window_length 就是每个示例窗口的长度（以特征帧数为单位）。
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate)) # vggish_params.EXAMPLE_HOP_SECONDS 是相邻两个示例窗口之间的跳跃时长（以秒为单位）。
                                                                 # 同样地，将 EXAMPLE_HOP_SECONDS 乘以 features_sample_rate，并经过四舍五入和整数转换，得到 example_hop_length，即相邻两个示例窗口之间的跳跃长度（以特征帧数为单位）。

#函数的返回值 log_mel_examples 是一个三维的 numpy 数组，形状为 [num_examples, num_frames, num_bands]，其中 num_examples 是示例的数量，num_frames 是每个示例包含的特征帧数，num_bands 是梅尔频谱中的频率带数量。
  log_mel_examples = mel_features.frame(
      log_mel, # 调用 mel_features 模块中的 frame 函数，将对数梅尔频谱特征 log_mel 划分为多个示例。
      window_length=example_window_length, # window_length 参数指定每个示例窗口的长度，即每个示例包含的特征帧数。
      hop_length=example_hop_length) # hop_length 参数指定相邻两个示例窗口之间的跳跃长度，即每隔多少个特征帧取一个新的示例。
  return log_mel_examples


def wavfile_to_examples(wav_file): # 这行代码定义了一个名为 wavfile_to_examples 的函数，它接受一个参数 wav_file。wav_file 可以是一个表示文件路径的字符串，也可以是一个类似文件对象（file-like object），用于指向一个 WAV 格式的音频文件。
  """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:详细描述了 wav_file 参数：它可以是文件的字符串路径，也可以是类似文件对象。同时，该文件被假定包含有符号 16 位脉冲编码调制（PCM）样本的 WAV 音频数据。
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.

  Returns:说明该函数的返回值与 waveform_to_examples 函数的返回值相同。waveform_to_examples 函数返回一个三维的 numpy 数组，形状为 [num_examples, num_frames, num_bands]，表示一系列的示例，每个示例包含一个对数梅尔频谱图的片段。
    See waveform_to_examples.
  """
  sr, wav_data = wavfile.read(wav_file) # wavfile.read 是 scipy.io.wavfile 模块中的函数，用于读取 WAV 格式的音频文件。
                                        # wav_file 是输入的 WAV 文件路径或类似文件对象。
                                        # 该函数返回两个值：sr 表示音频的采样率（每秒采样的样本数），wav_data 是一个 numpy 数组，包含音频的样本数据。如果音频是单声道，wav_data 是一维数组；如果是多声道，wav_data 是二维数组，其中每个声道的数据位于不同的列。
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype # assert 是 Python 中的断言语句，用于检查某个条件是否为真。如果条件为假，则会抛出 AssertionError 异常，并显示指定的错误信息。
                                                                            # 这里检查 wav_data 的数据类型是否为 np.int16（有符号 16 位整数），因为该函数假定输入的 WAV 文件包含的是有符号 16 位 PCM 样本。如果数据类型不是 np.int16，则会抛出异常并显示错误信息，指出样本类型不正确。
  wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0] 这行代码将音频数据的取值范围从 [-32768, 32767]（有符号 16 位整数的取值范围）归一化到 [-1.0, +1.0]。
                                 # 通过将 wav_data 数组中的每个元素除以 32768.0 来实现归一化，这样可以确保音频数据的取值范围符合后续处理的要求。
  T = 10 #定义一个变量 T 并赋值为 10。这个变量表示要处理的音频片段的数量，后续代码会将音频数据分割成 T 个片段进行处理。
  L = wav_data.shape[0] # wav_data.shape 返回一个元组，表示 wav_data 数组的形状。如果是单声道音频，wav_data 是一维数组，shape[0] 表示音频的样本数量；如果是多声道音频，shape[0] 表示音频的帧数（每个声道的样本数量相同）。
                        # 这里将音频的帧数（或样本数量）赋值给变量 L。
  log_mel = np.zeros([T, 96, 64]) # 这里创建一个形状为 [T, 96, 64] 的三维 numpy 数组 log_mel，用于存储 T 个音频片段的对数梅尔频谱特征。其中，96 可能表示每个音频片段的帧数，64 可能表示每个帧的梅尔频率带数量。
  for i in range(T): # T 在前面的代码中被定义为 10，这意味着循环会执行 10 次。每一次循环处理一个音频片段。
      s = i * sr # sr 是音频的采样率，即每秒采样的样本数。
                 # s 表示当前音频片段的起始样本索引，通过 i * sr 计算得到，意味着从第 i 秒开始。
      e = (i + 1) * sr # e 表示当前音频片段的结束样本索引，通过 (i + 1) * sr 计算得到，意味着到第 i + 1 秒结束。
      if len(wav_data.shape) > 1: # 这里判断音频数据 wav_data 的维度。如果 wav_data 的维度大于 1，说明音频是多声道的，
          data = wav_data[s:e, :]  # 此时使用 wav_data[s:e, :] 提取从 s 到 e 之间所有声道的样本数据。
      else:                        # 如果 wav_data 的维度等于 1，说明音频是单声道的
          data = wav_data[s:e]     # 直接使用 wav_data[s:e] 提取从 s 到 e 之间的样本数据。
      log_mel[i, :, :] = waveform_to_examples(data, sr) # 调用 waveform_to_examples 函数，将提取的音频片段 data 和采样率 sr 作为参数传入，该函数会将音频片段转换为对数梅尔频谱特征。
                                                        # 将得到的对数梅尔频谱特征存储在 log_mel 数组的第 i 个位置，即 log_mel[i, :, :]。
  return log_mel # 循环结束后，返回存储了所有音频片段对数梅尔频谱特征的 log_mel 数组。
