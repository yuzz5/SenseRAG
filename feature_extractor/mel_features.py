#这部分是版权声明和许可证信息，表明该代码由 TensorFlow 团队版权所有，并遵循 Apache 2.0 许可证。
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

#文档字符串描述了该模块的主要功能，即从音频波形中计算梅尔频谱特征。
"""Defines routines to compute mel spectrogram features from audio waveform."""

import numpy as np # 导入 numpy 库，用于数值计算。


def frame(data, window_length, hop_length): # data：这是一个至少一维的 numpy 数组，通常代表音频信号的时间序列数据，其第一个维度表示样本数量。
                                            # window_length：每个帧所包含的样本数量。在音频处理中，这个参数决定了每个分析窗口的大小。
                                            # hop_length：相邻两个帧之间的样本偏移量。它决定了帧与帧之间的重叠程度，如果 hop_length 小于 window_length，则帧会重叠。
  """Convert array into a sequence of successive possibly overlapping frames.

  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.

  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.

  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.

  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
  """
  num_samples = data.shape[0] #这行代码的作用是获取输入音频数据 data 的样本数量。data 是一个至少一维的 numpy 数组，data.shape 返回一个元组，元组的第一个元素 data.shape[0] 代表数组的第一个维度的大小，也就是音频数据的样本数量。
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length)) # num_samples - window_length：计算除去第一个完整帧后剩余的样本数量。
                                                                             # (num_samples - window_length) / hop_length：计算剩余样本可以容纳的帧数。
                                                                             # np.floor(...)：对上述结果向下取整，确保得到的是整数帧数。
                                                                             # 1 + ...：加上第一个完整帧，得到总的完整帧数。
  shape = (num_frames, window_length) + data.shape[1:] # (num_frames, window_length)：表示输出数组的前两个维度，其中 num_frames 是帧数，window_length 是每个帧的样本数量。
                                                       # data.shape[1:]：表示输入数组 data 除第一个维度外的其他维度，将其添加到 (num_frames, window_length) 后面，确保输出数组在其他维度上与输入数组保持一致。
  strides = (data.strides[0] * hop_length,) + data.strides # 这行代码定义了输出数组的步长（strides）。步长表示在数组中移动一个元素所需跳过的字节数。
                                                           # data.strides[0] * hop_length：表示在输出数组中从一个帧移动到下一个帧所需跳过的字节数，即相邻帧之间的偏移量。
                                                           # data.strides：表示输入数组 data 的步长，将其添加到 (data.strides[0] * hop_length,) 后面，确保输出数组在其他维度上的步长与输入数组保持一致。
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides) # 这行代码使用 numpy 的 stride_tricks.as_strided 函数创建一个新的数组视图，该视图的形状和步长由前面计算得到的 shape 和 strides 决定。这个函数通过修改数组的元数据（形状和步长）来实现数据的重新组织，而不需要复制数据，因此可以高效地完成帧分割操作。


def periodic_hann(window_length): # def periodic_hann(window_length): 定义了一个名为 periodic_hann 的函数，它接受一个参数 window_length，表示要生成的窗函数的点数。
#详细解释了经典汉宁窗和周期性汉宁窗的区别。经典汉宁窗在 numpy 中可以通过 np.hanning() 函数得到，它是一个周期为 N - 1 的余弦函数，在傅里叶分析中可能会有一些局限性。而周期性汉宁窗是一个周期为 N 的余弦函数，更适合在长度为 N 的傅里叶基上进行分析。
  """Calculate a "periodic" Hann window.

  The classic Hann window is defined as a raised cosine that starts and
  ends on zero, and where every value appears twice, except the middle
  point for an odd-length window.  Matlab calls this a "symmetric" window
  and np.hanning() returns it.  However, for Fourier analysis, this
  actually represents just over one cycle of a period N-1 cosine, and
  thus is not compactly expressed on a length-N Fourier basis.  Instead,
  it's better to use a raised cosine that ends just before the final
  zero value - i.e. a complete cycle of a period-N cosine.  Matlab
  calls this a "periodic" window. This routine calculates it.

  Args:
    window_length: The number of points in the returned window.

  Returns:
    A 1D np.array containing the periodic hann window.
  """
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length))) # np.arange(window_length)：生成一个从 0 到 window_length - 1 的一维数组，代表窗函数中每个点的索引。
                                                        # 2 * np.pi / window_length * np.arange(window_length)：计算余弦函数的相位。2 * np.pi / window_length 是每个点之间的相位增量，乘以索引数组得到每个点的相位值。
                                                        # np.cos(...)：对相位值数组应用余弦函数，得到余弦值数组。
                                                        # 0.5 * np.cos(...)：将余弦值数组乘以 0.5。
                                                        # 0.5 - (...)：用 0.5 减去上述结果，得到周期性汉宁窗的值。


def stft_magnitude(signal, fft_length,   # stft_magnitude，表明该函数的主要功能是计算 STFT 的幅度。
                   hop_length=None,      # signal：一维的 numpy 数组，代表输入的时域信号。例如，在音频处理中，这可能是一段音频的采样数据。
                   window_length=None):  # fft_length：整数类型，指定要应用的快速傅里叶变换（FFT）的大小。FFT 是一种高效计算离散傅里叶变换（DFT）的算法，fft_length 决定了频谱的分辨率。
                                         # hop_length：整数类型，可选参数，默认值为 None。它表示相邻两个分析窗口之间的样本偏移量，即每个窗口向前移动的样本数。在实际使用时，如果不指定，可能会在函数内部进行默认值的设置。
                                         # window_length：整数类型，可选参数，默认值为 None。它表示每个输入到 FFT 的样本块的长度，也就是分析窗口的长度。同样，如果不指定，函数内部可能会有默认的处理。
#明确指出该函数用于计算输入时域信号的短时傅里叶变换的幅度。
  """Calculate the short-time Fourier transform magnitude.

  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.

  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
  """
  frames = frame(signal, window_length, hop_length) # 调用之前定义的 frame 函数，将输入的一维时域信号 signal 分割成多个帧。
                                                    # signal：输入的一维时域信号。
                                                    # window_length：每个帧的长度，即每个分析窗口包含的样本数。
                                                    # hop_length：相邻帧之间的偏移量，控制帧与帧之间的重叠程度。
                                                    # frames 是一个二维的 numpy 数组，其形状为 (num_frames, window_length)，其中 num_frames 是分割得到的帧数。
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length) # 调用 periodic_hann 函数生成一个周期性的汉宁（Hann）窗。汉宁窗是一种常用的窗函数，在信号处理中用于减少频谱泄漏问题。
                                        # window_length 是每个分析窗口包含的样本数，该参数决定了窗函数的长度。
                                        # window 是一个一维的 numpy 数组，其长度为 window_length，包含了周期性汉宁窗的数值。
  windowed_frames = frames * window # 将分帧后的信号 frames 与生成的窗函数 window 逐元素相乘。这样做的目的是对每一帧信号应用窗函数，使得信号在窗口两端逐渐衰减到零，从而减少频谱泄漏。
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length))) # 对经过窗函数处理后的帧 windowed_frames 应用实值快速傅里叶变换（np.fft.rfft），并取其幅度。由于输入信号是实值的，因此使用 np.fft.rfft 可以只计算正频率部分的频谱，从而减少计算量。
                                                               # windowed_frames 是一个二维的 numpy 数组，代表经过窗函数处理后的帧。
                                                               # int(fft_length) 是要应用的快速傅里叶变换的大小，通常是一个 2 的幂次方，以提高计算效率。
                                                               # np.fft.rfft(windowed_frames, int(fft_length)) 返回一个二维的 numpy 数组，包含了每个帧的复数频谱。np.abs(...) 对复数频谱取模，得到频谱的幅度。最终返回的二维数组形状为 (num_frames, fft_length/2 + 1)，其中每一行包含了对应帧的 FFT 的 fft_length/2 + 1 个唯一值的幅度。


# Mel spectrum constants and functions.接下来的代码部分与梅尔频谱的常量和相关函数有关
_MEL_BREAK_FREQUENCY_HERTZ = 700.0 # 量名以单下划线开头，按照 Python 的约定，这表示它是一个私有变量，通常用于模块内部使用。_MEL_BREAK_FREQUENCY_HERTZ 代表梅尔频率转换中的一个关键频率点，单位是赫兹（Hz）。
                                   # 在将频率从赫兹转换到梅尔的公式中，这个频率点是一个转折点，区分了线性和对数转换的区域。在低于这个频率时，梅尔频率和赫兹频率近似成线性关系；而在高于这个频率时，两者呈对数关系。
_MEL_HIGH_FREQUENCY_Q = 1127.0 # _MEL_HIGH_FREQUENCY_Q 是一个与高频相关的常量，在将频率从赫兹转换到梅尔的公式中起到缩放因子的作用。
                               # 它参与了将赫兹频率转换为梅尔频率的具体计算，帮助调整转换的比例关系，使得转换后的梅尔频率更符合人类听觉系统的特性。


def hertz_to_mel(frequencies_hertz): # hertz_to_mel，清晰地表明了该函数的功能是将赫兹频率转换为梅尔频率。
                                     # frequencies_hertz，它可以是一个标量（单个频率值），也可以是一个 numpy 数组（包含多个频率值），单位为赫兹。
                                     # 返回一个与输入 frequencies_hertz 大小相同的对象（标量或 numpy 数组），其中包含对应的梅尔频率值。
  """Convert frequencies to mel scale using HTK formula.

  Args:
    frequencies_hertz: Scalar or np.array of frequencies in hertz.

  Returns:
    Object of same size as frequencies_hertz containing corresponding values
    on the mel scale.
  """
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)) # 该函数使用了 HTK（Hidden Markov Model Toolkit）公式来进行频率转换。具体公式为：\(M(f) = Q *ln(1 + f/fb)其中，M(f) 是频率 f 对应的梅尔频率，Q 是 _MEL_HIGH_FREQUENCY_Q（值为 1127.0），\(f_b\) 是 _MEL_BREAK_FREQUENCY_HERTZ（值为 700.0）。


def spectrogram_to_mel_matrix(num_mel_bins=20,           # 指定最终生成的梅尔频谱图中的频段数量，也就是生成的梅尔转换矩阵的列数。梅尔频谱图是将音频信号的频谱从线性频率转换到梅尔频率上，num_mel_bins 决定了在梅尔频率尺度上划分的频段数量。
                              num_spectrogram_bins=129,  # 表示输入频谱图中的频段数量。通常情况下，频谱图是通过快速傅里叶变换（FFT）得到的，其频段数量为 fft_size / 2 + 1，因为对于实值信号的 FFT 结果具有对称性，只需要保留一半加上直流分量的结果即可。
                              audio_sample_rate=8000,    # 表示输入音频信号的采样率，单位是赫兹（Hz）。采样率决定了音频信号在时间上的分辨率，在将频谱图转换为梅尔频谱图的过程中，需要根据采样率来确定每个频谱图频段对应的实际频率。
                              lower_edge_hertz=125.0,    # 指定梅尔频谱图中最低频段的起始频率，单位是赫兹（Hz）。这个参数决定了在频率转换过程中，哪些低频部分的频率会被包含在梅尔频谱图中。
                              upper_edge_hertz=3800.0):  # 指定梅尔频谱图中最高频段的截止频率，单位是赫兹（Hz）。它决定了在频率转换过程中，哪些高频部分的频率会被包含在梅尔频谱图中。
  """Return a matrix that can post-multiply spectrogram rows to make mel.

  Returns a np.array matrix A that can be used to post-multiply a matrix S of
  spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
  "mel spectrogram" M of frames x num_mel_bins.  M = S A.

  The classic HTK algorithm exploits the complementarity of adjacent mel bands
  to multiply each FFT bin by only one mel weight, then add it, with positive
  and negative signs, to the two adjacent mel bands to which that bin
  contributes.  Here, by expressing this operation as a matrix multiply, we go
  from num_fft multiplies per frame (plus around 2*num_fft adds) to around
  num_fft^2 multiplies and adds.  However, because these are all presumably
  accomplished in a single call to np.dot(), it's not clear which approach is
  faster in Python.  The matrix multiplication has the attraction of being more
  general and flexible, and much easier to read.

  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.  This is
      the number of columns in the output matrix.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    audio_sample_rate: Samples per second of the audio at the input to the
      spectrogram. We need this to figure out the actual frequencies for
      each spectrogram bin, which dictates how they are mapped into mel.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum.  This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.

  Returns:
    An np.array with shape (num_spectrogram_bins, num_mel_bins).

  Raises:
    ValueError: if frequency edges are incorrectly ordered or out of range.
  """
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz < 0.0:
    raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > nyquist_hertz:
    raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
  # The i'th mel band (starting from i=1) has center frequency
  # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
  # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
  # the band_edges_mel arrays.
  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the *mel* domain, not hertz.
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  # HTK excludes the spectrogram DC bin; make sure it always gets a zero
  # coefficient.
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix


def log_mel_spectrogram(data, # 代表音频波形数据。音频信号在时域上是一系列的采样点，这个数组存储了这些采样点的值。
                        audio_sample_rate=8000, # 音频数据的采样率，单位是赫兹（Hz）。采样率表示每秒采集的音频样本数量，它决定了音频信号在时间上的分辨率。
                        log_offset=0.0, # 取对数时添加到数值上的偏移量。这是为了避免在计算对数时出现负无穷大（-Inf）的情况，因为对数函数的定义域是正实数，当输入为 0 时会得到负无穷大。
                        window_length_secs=0.025, # 每个分析窗口的持续时间，单位是秒。在计算频谱图时，通常会将音频信号分成多个短的窗口，分别对每个窗口进行傅里叶变换，这个参数决定了每个窗口的时长。
                        hop_length_secs=0.010, # 相邻两个分析窗口之间的间隔时间，单位是秒。它决定了窗口在音频信号上滑动的步长，通过设置合适的步长可以控制分析的时间分辨率。
                        **kwargs): # 用于传递额外的参数给 spectrogram_to_mel_matrix 函数。这些参数可以用于自定义梅尔频谱转换矩阵的生成，例如 num_mel_bins、lower_edge_hertz 等。
  """Convert waveform to a log magnitude mel-frequency spectrogram.

  Args:
    data: 1D np.array of waveform data.
    audio_sample_rate: The sampling rate of data.
    log_offset: Add this to values when taking log to avoid -Infs.
    window_length_secs: Duration of each window to analyze.
    hop_length_secs: Advance between successive analysis windows.
    **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.

  Returns:
    2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
    magnitudes for successive frames.
  """
  window_length_samples = int(round(audio_sample_rate * window_length_secs)) # 计算窗口长度的采样点数
                                                                             # audio_sample_rate：音频数据的采样率，单位是赫兹（Hz），表示每秒采集的音频样本数量。
                                                                             # window_length_secs：每个分析窗口的持续时间，单位是秒
                                                                             # window_length_samples：每个分析窗口所包含的音频采样点数。
                                                                             # 计算过程：audio_sample_rate * window_length_secs：将采样率与窗口持续时间相乘，得到理论上窗口包含的采样点数，这个结果可能是一个浮点数。
                                                                                     # round(...)：对上述结果进行四舍五入，得到一个最接近的整数。int(...)：将四舍五入后的结果转换为整数类型。
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs)) # hop_length_secs：相邻两个分析窗口之间的间隔时间，单位是秒。
                                                                       # hop_length_samples：相邻两个分析窗口之间的采样点数间隔。
                                                                       # audio_sample_rate * hop_length_secs：将采样率与窗口间隔时间相乘，得到理论上的采样点数间隔，可能是浮点数。
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0))) # fft_length：进行快速傅里叶变换（FFT）时使用的长度，通常为 2 的幂次方，这样可以提高 FFT 算法的计算效率。
                                                                              # np.log(window_length_samples) / np.log(2.0)：计算 window_length_samples 以 2 为底的对数，即求出一个指数值，使得 2 的该指数次幂接近 window_length_samples。
                                                                              # np.ceil(...)：对上述对数结果进行向上取整，确保得到的指数值不小于实际计算得到的对数。
                                                                              # 2 ** ...：将 2 提升到向上取整后的指数次幂，得到最终的 FFT 长度，该长度是大于或等于 window_length_samples 的最小的 2 的幂次方。
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)
  mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
      num_spectrogram_bins=spectrogram.shape[1],
      audio_sample_rate=audio_sample_rate, **kwargs))
  return np.log(mel_spectrogram + log_offset)
