import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import pdb

#init_layers 函数用于初始化神经网络层的参数，确保模型训练的稳定性和收敛性。
def init_layers(layers): #定义一个名为 init_layers 的函数，用于初始化神经网络层的参数。layers，一个包含 PyTorch 神经网络层（如 nn.Linear等）的列表。
    for layer in layers: #遍历输入的每一层。
        nn.init.xavier_uniform(layer.weight) #使用 Xavier 均匀分布初始化当前层的权重参数。Xavier 初始化通过保持输入和输出的方差一致，减少了梯度消失和爆炸的风险。具体来说，权重 w 从均匀分布 [-a, a] 中采样，（其中 a = gain * sqrt(6 / (fan_in + fan_out))。fan_in 是输入神经元的数量；fan_out 是输出神经元的数量；gain 是一个缩放因子，对于 ReLU 激活函数通常为 sqrt(2)平方根，对于 Sigmoid 为 1。）
        layer.bias.data.fill_(0) #将当前层的偏置参数全部初始化为 0。偏置的初始值通常设为 0，因为非零初始值可能会引入不必要的偏差。模型可以通过学习来调整偏置值。

#这段代码定义了一个名为 SelfAttention 的 PyTorch 神经网络模块，它是一个自注意力机制的实现，这里以音频自注意力为例。
class SelfAttention(nn.Module):# 自定义一个selfattention，它是nn.module的一个子类
    # Take audio self-attention for example.  init函数包括了整个网络全部要的层都在这个里面
    def __init__(self, audio_emb_dim, hidden_dim=64): #这行代码定义了 __init__ 方法，它是类的构造函数，构造我们的模型，在创建 SelfAttention 类的实例时会自动调用。
                                                      #audio_emb_dim 是一个必需的参数，表示输入音频特征的嵌入维度，即每个音频特征向量的长度。
                                                      #hidden_dim 是一个可选参数，默认值为 64，它表示线性层输出的隐藏维度，也就是经过线性变换后特征向量的长度。
        super(SelfAttention, self).__init__() #super(SelfAttention, self).__init__() 调用了父类 nn.Module 的构造函数。由于 SelfAttention 类继承自 nn.Module，通过调用父类的构造函数，可以确保 SelfAttention 类的实例正确继承并初始化 nn.Module 的所有属性和方法。这是在自定义 PyTorch 模块时的标准做法，用于确保模块的正确初始化。
        #接下来调用三个全连接层
        self.phi = nn.Linear(audio_emb_dim, hidden_dim) #nn.Linear 是 PyTorch 中的线性层（全连接层），它执行一个线性变换，将输入特征映射到输出特征。然后把它存在一个类的成员变量里面self.phi
                                                        #audio_emb_dim 是输入音频特征的嵌入维度，即每个音频特征向量的长度。
                                                        #hidden_dim 是输出的隐藏维度，也就是经过线性变换后特征向量的长度。
                                                        #self.phi 是一个线性层，用于将输入的音频特征转换为查询（query）向量。
        self.theta = nn.Linear(audio_emb_dim, hidden_dim) #同样使用 nn.Linear 定义一个线性层。self.theta 用于将输入的音频特征转换为键（key）向量。
        self.g = nn.Linear(audio_emb_dim, hidden_dim) #还是使用 nn.Linear 定义线性层。self.g 用于将输入的音频特征转换为值（value）向量。
        layers = [self.phi, self.theta, self.g] #创建一个列表 layers，将之前定义的三个线性层 self.phi、self.theta 和 self.g 放入列表中。
        init_layers(layers) #调用自定义函数 init_layers，传入 layers 列表作为参数。

#这段代码是 SelfAttention 类中的 forward 方法，其核心功能是实现音频特征的自注意力机制。自注意力机制可以让模型在处理输入序列时，自动关注序列中不同位置元素之间的关系。
    #接下来定义前向函数forward,输入是audio_feature
    def forward(self, audio_feature):
        # audio_feature: [bs, seg_num=10, 128]
        bs, seg_num, audio_emb_dim = audio_feature.shape #这行代码获取输入音频特征张量的形状信息，分别将批量大小、音频片段数量和每个片段的嵌入维度赋值给 bs、seg_num 和 audio_emb_dim。
        phi_a = self.phi(audio_feature) #将输入的音频特征 audio_feature 转换为查询（Query）向量 phi_a。self.phi 是在类的初始化中定义的线性层（nn.Linear(audio_emb_dim, hidden_dim)）。线性变换公式：phi_a = audio_feature × W_phi + b_phi，其中 W_phi 和 b_phi 是线性层的权重和偏置。
                                        #输入 / 输出形状：输入 audio_feature 形状：[batch_size, seq_len, audio_emb_dim]（例如 [32, 10, 128]）。输出 phi_a 形状：[batch_size, seq_len, hidden_dim]（例如 [32, 10, 64]）。
        theta_a = self.theta(audio_feature) #将输入的音频特征 audio_feature 转换为键（Key）向量 theta_a。self.theta 是与 self.phi 类似的线性层（nn.Linear(audio_emb_dim, hidden_dim)）。线性变换公式：theta_a = audio_feature × W_theta + b_theta。
                                            #输入 / 输出形状：输入 audio_feature 形状：[batch_size, seq_len, audio_emb_dim]；输出 theta_a 形状：[batch_size, seq_len, hidden_dim]。
        g_a = self.g(audio_feature) #将输入的音频特征 audio_feature 转换为值（Value）向量 g_a。self.g 是与 self.phi、self.theta 类似的线性层（nn.Linear(audio_emb_dim, hidden_dim)）。线性变换公式：g_a = audio_feature × W_g + b_g。
                                    #输入 / 输出形状：输入 audio_feature 形状：[batch_size, seq_len, audio_emb_dim]；输出 g_a 形状：[batch_size, seq_len, hidden_dim]。
        a_seg_rel = torch.bmm(phi_a, theta_a.permute(0, 2, 1)) # [bs, seg_num, seg_num] # torch.bmm 是 PyTorch 中的批量矩阵乘法函数，用于在批量维度上进行矩阵乘法。这里输入的 phi_a 是查询（query）向量，theta_a 是键（key）向量。
                                                                                        # theta_a.permute(0, 2, 1) 对 theta_a 的最后两个维度进行转置，使得矩阵形状符合矩阵乘法的要求。第一个元素 0 表示将原张量的第 0 个维度（也就是批量大小 bs 这一维度）保持在新张量的第 0 个位置；第二个元素 2 表示把原张量的第 2 个维度（即嵌入维度 audio_emb_dim）放到新张量的第 1 个位置。第三个元素 1 表示将原张量的第 1 个维度（即音频片段数量 seg_num）置于新张量的第 2 个位置。
                                                                                        # 经过矩阵乘法后，得到音频片段之间的相关性矩阵 a_seg_rel，其形状为 [bs, seg_num, seg_num]，其中 bs 是批量大小，seg_num 是音频片段的数量。
        a_seg_rel = a_seg_rel / torch.sqrt(torch.FloatTensor([audio_emb_dim]).cuda()) #为了缓解梯度消失和梯度爆炸问题，对相关性矩阵 a_seg_rel 进行缩放操作。
                                                                                      #torch.FloatTensor([audio_emb_dim]) 创建一个包含输入音频特征嵌入维度的单元素张量。
                                                                                      #torch.sqrt(...) 计算该张量的平方根。
                                                                                      #.cuda() 将该张量移动到 GPU 上，确保与 a_seg_rel 在同一设备上进行计算。
                                                                                      #最后将相关性矩阵 a_seg_rel 除以该平方根。
        a_seg_rel = F.relu(a_seg_rel) #F.relu 是 PyTorch 中的 ReLU（Rectified Linear Unit）激活函数，用于对相关性矩阵进行非线性变换。ReLU 函数将负值置为 0，保留正值，引入了非线性因素，有助于模型学习更复杂的特征表示。
        a_seg_rel = (a_seg_rel + a_seg_rel.permute(0, 2, 1)) / 2 #对相关性矩阵进行对称化处理，将其与转置后的矩阵相加并除以 2。a_seg_rel.permute(0, 2, 1) 对 a_seg_rel 的最后两个维度进行转置。
        sum_a_seg_rel = torch.sum(a_seg_rel, dim=-1, keepdim=True) #torch.sum(a_seg_rel, dim=-1, keepdim=True)：对 a_seg_rel 张量的最后一个维度（即 seg_num 维度）进行求和操作，keepdim=True 表示保持求和后的维度不变。a_seg_rel 是经过缩放、ReLU 激活和对称化处理后的相关性矩阵，形状为 [bs, seg_num, seg_num]，求和后 sum_a_seg_rel 的形状为 [bs, seg_num, 1]。
        a_seg_rel = a_seg_rel / (sum_a_seg_rel + 1e-8) #a_seg_rel = a_seg_rel / (sum_a_seg_rel + 1e-8)：将 a_seg_rel 矩阵的每一行元素除以该行元素的和（加上一个小的常数 1e-8 是为了避免除零错误），使得每一行元素的和为 1，从而完成归一化操作。归一化后的 a_seg_rel 可以看作是注意力权重矩阵。
        a_att = torch.bmm(a_seg_rel, g_a) #torch.bmm 是 PyTorch 中的批量矩阵乘法函数，用于在批量维度上进行矩阵乘法。a_seg_rel 是归一化后的注意力权重矩阵，形状为 [bs, seg_num, seg_num]；g_a 是对输入音频特征经过线性变换 self.g 得到的值（value）向量，形状为 [bs, seg_num, hidden_dim]。通过矩阵乘法 torch.bmm(a_seg_rel, g_a)，得到加权后的音频特征 a_att，形状为 [bs, seg_num, hidden_dim]。
        a_att_plus_ori = a_att + audio_feature #将加权后的音频特征 a_att 与原始输入的音频特征 audio_feature 相加，实现残差连接。残差连接有助于缓解梯度消失问题，使得模型在训练过程中更容易学习到有效的特征表示。
        return a_att_plus_ori, a_seg_rel #返回经过注意力机制处理并加上残差连接后的音频特征 a_att_plus_ori，以及归一化的注意力权重矩阵 a_seg_rel。


class AVGA(nn.Module): #class AVGA(nn.Module)：定义了一个名为 AVGA 的类，它继承自 nn.Module。在 PyTorch 中，nn.Module 是所有神经网络模块的基类，自定义的神经网络模块需要继承这个类，并实现 __init__ 和 forward 方法
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, a_dim=128, v_dim=512, hidden_size=512, map_size=49): #定义类的初始化方法，设置默认参数。a_dim：音频特征的维度，默认 128；v_dim：视频特征的维度，默认 512；hidden_size：隐藏层的维度，默认 512；map_size：注意力图的大小，默认 49（对应 7×7 的特征图）。
        super(AVGA, self).__init__() #调用父类 nn.Module 的构造函数，确保正确初始化基类。所有继承自 nn.Module 的类必须调用此方法。
        self.relu = nn.ReLU() #创建 ReLU 激活函数实例，用于后续的非线性变换。数学表达式：ReLU(x) = max(0, x)。
        self.affine_audio = nn.Linear(a_dim, hidden_size) #将音频特征从 a_dim 维度映射到 hidden_size 维度。输入：[batch_size, seq_len, a_dim（时间步特征维度）]；输出：[batch_size, seq_len（序列长度10个片段）, hidden_size]
        self.affine_video = nn.Linear(v_dim, hidden_size) #将视频特征从 v_dim 维度映射到 hidden_size 维度。输入：[batch_size, seq_len, v_dim]；输出：[batch_size, seq_len, hidden_size]
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False) #用于计算视觉内容的线性层
                                                                     #nn.Linear(hidden_size, map_size, bias=False) 也是一个全连接层，但这里设置了 bias=False，即不使用偏置项。该层将经过 self.affine_video 处理后的视频特征从 hidden_size 维度映射到 map_size 维度，用于后续计算视觉内容相关的信息。
                                                                     #偏置项：在神经网络中，偏置项（Bias） 是线性变换中的一个可学习参数，用于调整模型的输出范围。具体来说，偏置项允许模型在输入为零时仍能产生非零输出，从而增强模型的表达能力。y=Wx+b：b 是偏置向量（即偏置项）
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False) #用于计算音频引导信息的线性层
                                                                     #同样是一个不使用偏置项的全连接层，它将经过 self.affine_audio 处理后的音频特征从 hidden_size 维度映射到 map_size 维度，用于计算音频对视觉的引导信息。
        self.affine_h = nn.Linear(map_size, 1, bias=False) #用于计算注意力分数的线性层
                                                           #这是最后一个不使用偏置项的全连接层，它将经过前面处理得到的 map_size 维度的特征映射到 1 维，用于计算注意力分数，最终得到注意力图。

        init.xavier_uniform(self.affine_v.weight) # 这是最后一个不使用偏置项的全连接层，它将经过前面处理得到的 map_size 维度的特征映射到 1 维，用于计算注意力分数，最终得到注意力图。
                                                  # xavier_uniform 是 Xavier 初始化方法的一种实现，它使用均匀分布来初始化权重。Xavier 初始化的核心思想是使得每一层输出的方差尽量等于输入的方差，从而保证信息在网络中的流动更加稳定。
                                                  # self.affine_v 是 AVGA 类中定义的一个线性层（nn.Linear），其作用是将经过 self.affine_video 处理后的视频特征从 hidden_size 维度映射到 map_size 维度，用于后续计算视觉内容相关的信息。
                                                  # self.affine_v.weight 表示 self.affine_v 这个线性层的权重参数。通过 init.xavier_uniform(self.affine_v.weight)，我们将该线性层的权重初始化为 Xavier 均匀分布。
                                                  # 代码作用：通过对 self.affine_v 线性层的权重进行 Xavier 均匀初始化，可以使得模型在训练初期，该层的输入和输出具有相似的分布，从而有助于缓解梯度消失或梯度爆炸问题，提高模型的训练效率和稳定性。
        init.xavier_uniform(self.affine_g.weight) #对 self.affine_g 线性层的权重进行 Xavier 均匀初始化，能够使模型在训练初期，该层的输入和输出具有相似的分布，有助于缓解梯度消失或梯度爆炸问题，提高模型的训练效率和稳定性。
        init.xavier_uniform(self.affine_h.weight) #对 self.affine_h 线性层的权重进行 Xavier 均匀初始化，能够使模型在训练初期，该层的输入和输出具有相似的分布，有助于缓解梯度消失或梯度爆炸问题，提高模型的训练效率和稳定性。
        init.xavier_uniform(self.affine_audio.weight) #self.affine_audio.weight 表示 self.affine_audio 这个线性层的权重参数。通过 init.xavier_uniform(self.affine_audio.weight)，我们将该线性层的权重初始化为 Xavier 均匀分布。
                                                      #对 self.affine_audio 线性层的权重进行 Xavier 均匀初始化，可以使得模型在训练初期，该层的输入和输出具有相似的分布，从而有助于缓解梯度消失或梯度爆炸问题，提高模型的训练效率和稳定性。
        init.xavier_uniform(self.affine_video.weight) #self.affine_video.weight 表示 self.affine_video 这个线性层的权重参数。通过 init.xavier_uniform(self.affine_video.weight) 这行代码，将该线性层的权重初始化为 Xavier 均匀分布。

    def forward(self, audio, video):#forward 方法是 PyTorch 中自定义模块（nn.Module 的子类）必须实现的方法，用于定义前向传播过程。
        # audio: [bs, 10, 128] audio 是输入的音频特征，其形状为 [bs, 10, 128]，其中 bs 表示批量大小（batch size），10 表示每个样本的音频片段数量，128 表示每个音频片段的特征维度。
        # video: [bs, 10, 7, 7, 512] video 是输入的视频特征，其形状为 [bs, 10, 7, 7, 512]，其中 bs 是批量大小，10 是每个样本的视频片段数量，7x7 是每个视频片段的空间维度，512 是每个空间位置的特征维度。
        V_DIM = video.size(-1) #video.size(-1) 用于获取 video 张量的最后一个维度的大小，即 512。这里将其赋值给变量 V_DIM，方便后续使用。
        v_t = video.view(video.size(0) * video.size(1), -1, V_DIM) # [bs*10, 49, 512] view 方法用于改变张量的形状。
                                                                   # video.size(0) * video.size(1) 计算出变换后张量的第一个维度大小，即 bs * 10，将批量大小和视频片段数量合并。-1 表示该维度的大小由其他维度自动推断，这里会将 7x7 的空间维度合并为 49。V_DIM 是最后一个维度的大小，即 512。最终，v_t 的形状变为 [bs*10, 49, 512]。
        V = v_t #将变换后的视频特征 v_t 赋值给变量 V，这里的 V 可能用于后续的计算，作为原始视频特征的一个副本。

        # Audio-guided visual attention  表明接下来的代码用于实现音频引导的视觉注意力机制。
        v_t = self.relu(self.affine_video(v_t)) # [bs*10, 49, 512] self.affine_video 是 AVGA 类初始化时定义的线性层，其定义为 self.affine_video = nn.Linear(512, hidden_size)，这里 hidden_size 默认为 512。所以该线性层会把输入的视频特征 v_t 从 512 维映射到 512 维。
                                                # self.relu 是 ReLU 激活函数，用于引入非线性因素，增强模型的表达能力。它会对线性层的输出进行非线性变换，将所有负数置为 0，只保留正数部分。
                                                # v_t 经过线性变换和激活操作后，形状依然是 [bs*10, 49, 512]，其中 bs 是批量大小，10 是每个样本的视频片段数量，49 是空间维度（由 7x7 合并而来），512 是特征维度。
        a_t = audio.view(-1, audio.size(-1)) # [bs*10, 128] audio 的原始形状是 [bs, 10, 128]，通过 view 方法将其形状变换为 [bs*10, 128]，即将批量大小和视频片段数量合并为一个维度。
                                             # -1 表示该维度的大小由其他维度自动推断，这里会根据 audio 的总元素数量和指定的最后一个维度大小 audio.size(-1)（即 128）来确定第一个维度的大小为 bs*10。
        a_t = self.relu(self.affine_audio(a_t)) # [bs*10, 512] self.affine_audio 是 AVGA 类初始化时定义的线性层，定义为 self.affine_audio = nn.Linear(128, hidden_size)，这里 hidden_size 同样为 512。所以该线性层会把输入的音频特征 a_t 从 128 维映射到 512 维。
                                                # self.relu 对线性层的输出进行 ReLU 激活操作，引入非线性。最终 a_t 的形状变为 [bs*10, 512]。
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2) # [bs*10, 49, 49] + [bs*10, 49, 1]
                                                                         # self.affine_v 是一个线性层，其输入维度为 hidden_size（在 __init__ 方法中通常设置为 512），输出维度为 map_size（在 __init__ 方法中设置为 49）。
                                                                         # self.affine_v(v_t)：self.affine_v 是一个线性层，对处理后的视频特征 v_t 进行线性变换，将其从 512 维映射到 49 维，输出形状为 [bs*10, 49, 49]。
                                                                         # self.affine_g(a_t)：self.affine_g 是一个线性层，对处理后的音频特征 a_t 进行线性变换，将其从 512 维映射到 49 维，输出形状为 [bs*10, 49]。
                                                                         # unsqueeze(2)：在 self.affine_g(a_t) 的结果上，在第 2 个维度（索引从 0 开始）插入一个维度，将其形状变为 [bs*10, 49, 1]。
                                                                         # 最后将两者相加，得到融合后的特征 content_v，形状为 [bs*10, 49, 49]。
                                                                         # 总结：这段代码通过对音频和视频特征分别进行线性变换和激活操作，然后将两者融合起来，为后续计算视频特征的注意力权重提供了基础。融合后的特征 content_v 包含了音频和视频的信息，有助于模型更好地捕捉音频和视频之间的关联。

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2) # 计算注意力分数z_t
                                                            # [bs*10, 49] F.tanh(content_v)：对融合后的特征 content_v 应用 tanh 激活函数，将其值映射到 [-1, 1] 区间，引入非线性。content_v 的形状为 [bs*10, 49, 49]。
                                                            # self.affine_h：是一个线性层，将 tanh 激活后的特征从 49 维映射到 1 维。该线性层定义为 self.affine_h = nn.Linear(map_size, 1, bias=False)，这里 map_size 为 49。
                                                            # squeeze(2)：移除张量中维度大小为 1 的第 2 个维度（索引从 0 开始）。经过线性层和 squeeze 操作后，z_t 的形状变为 [bs*10, 49]，其中每个元素表示对应位置的注意力分数。
        # 此代码行的作用是对 z_t 应用 F.softmax 函数进行归一化处理，然后调整其形状，最终得到注意力图 alpha_t。注意力图用于表示在音频特征引导下，视觉特征中各个部分的重要程度。
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map, [bs*10, 1, 49]。alpha_t表示注意力图
                                                                            # F.softmax 是 PyTorch 中的一个函数，用于将输入张量 z_t 转换为概率分布。
                                                                            # dim=-1 表示在最后一个维度上进行 softmax 操作。z_t 的形状为 [bs*10, 49]，这里会对每个 [bs*10] 批次中的 49 个元素进行 softmax 计算，使得每个批次内的 49 个元素之和为 1，这样得到的结果可以看作是注意力权重。
                                                                            # view 方法用于改变张量的形状。z_t.size(0) 是批次大小 bs*10，z_t.size(1) 是 49，-1 表示该维度的大小会根据张量元素总数和其他指定维度自动推断。
                                                                            # 最终将经过 softmax 处理后的 z_t 形状从 [bs*10, 49] 变为 [bs*10, 1, 49]，得到的 alpha_t 就是注意力图（attention map），其中 1 这个维度是为了后续进行矩阵乘法而添加的。
        c_t = torch.bmm(alpha_t, V).view(-1, V_DIM) # [bs*10, 1, 512] 计算加权后的视觉特征  V 是原始的视觉特征，形状为 [bs*10, 49, 512]。
                                                    # torch.bmm(alpha_t, V) 对注意力映射 alpha_t 和原始视觉特征 V 进行批量矩阵乘法，得到加权后的视觉特征。
                                                    # .view(-1, V_DIM) 对加权后的视觉特征进行形状调整，得到形状为 [bs*10, 512] 的特征 c_t。
        video_t = c_t.view(video.size(0), -1, V_DIM) # attended visual features, [bs, 10, 512] 调整形状得到最终的视觉特征 video_t
                                                     # .view(video.size(0), -1, V_DIM) 对 c_t 进行形状调整，将其恢复到批量大小 bs 和时间步长 10 的维度，最终得到形状为 [bs, 10, 512] 的经过注意力处理后的视觉特征 video_t。
        return video_t # 将经过注意力处理后的视觉特征 video_t 返回。


class LSTM_A_V(nn.Module): # LSTM_A_V 类继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类。通过继承 nn.Module，LSTM_A_V 类可以使用 PyTorch 提供的各种功能，如参数管理、前向传播等。
    def __init__(self, a_dim, v_dim, hidden_dim=128, seg_num=10): # 类的初始化方法，接收音频特征维度(a_dim)、视频特征维度(v_dim)
                                                                  # 隐藏层维度(hidden_dim，默认128)它表示 LSTM（长短期记忆网络）层中隐藏状态的维度。隐藏状态是 LSTM 在处理序列数据时内部维护的一种状态，用于存储和传递序列中的信息，hidden_dim决定了隐藏状态的大小。
                                                                  # seg_num=10：同样是可选参数，默认值为 10。它表示每个样本被分割成的段数，在处理时间序列数据（如音频和视频）时，通常会将数据按时间维度分割成多个段进行处理。
        super(LSTM_A_V, self).__init__() # super(LSTM_A_V, self)：super()函数用于调用父类（基类）的方法。这里LSTM_A_V类继承自nn.Module（从前面完整代码可以看出），super(LSTM_A_V, self).__init__()的作用是调用nn.Module类的构造函数，确保父类的初始化逻辑被正确执行。

# 这两行代码创建了两个 nn.LSTM 实例，分别命名为 self.lstm_audio 和 self.lstm_video，用于对音频和视频特征进行时间序列建模。双向 LSTM 可以同时考虑序列的过去和未来信息，有助于捕捉序列中的长期依赖关系。
        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0) # self.lstm_audio：这是类的一个属性，代表用于处理音频特征的双向 LSTM 层。
                                                                                                           # nn.LSTM：PyTorch 提供的 LSTM 层类，用于创建一个 LSTM 神经网络层。
                                                                                                           # a_dim：这是输入到音频 LSTM 层的特征维度。在处理音频数据时，每个时间步的音频特征会被表示为一个长度为 a_dim 的向量。
                                                                                                           # hidden_dim：这是 LSTM 层隐藏状态的维度。隐藏状态是 LSTM 单元在处理序列数据时内部维护的状态，hidden_dim 决定了隐藏状态的大小。
                                                                                                           # 1：表示 LSTM 层的层数，这里设置为 1 意味着只使用一层 LSTM。
                                                                                                           # batch_first=True：这个参数指定输入和输出张量的维度顺序。当 batch_first 为 True 时，输入和输出张量的形状将为 (batch_size, seq_length, input_size)，其中 batch_size 是批量大小，seq_length 是序列长度，input_size 是输入特征的维度。
                                                                                                           # bidirectional=True：表示使用双向 LSTM。双向 LSTM 会分别从序列的正向和反向进行处理，然后将两个方向的输出合并，这样可以捕捉到序列的过去和未来信息。
                                                                                                           # dropout=0.0：设置 LSTM 层的 dropout 概率。dropout 是一种正则化技术，用于防止过拟合。这里将 dropout 概率设置为 0.0，表示不使用 dropout。
        self.lstm_video = nn.LSTM(v_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0) # self.lstm_video：这是类的另一个属性，代表用于处理视频特征的双向 LSTM 层。
                                                                                                           # v_dim：这是输入到视频 LSTM 层的特征维度。在处理视频数据时，每个时间步的视频特征会被表示为一个长度为 v_dim 的向量。
                                                                                                           # 其余参数的含义与 self.lstm_audio 中的参数相同。

    def init_hidden(self, a_fea, v_fea): # init_hidden：是 LSTM_A_V 类中的一个方法，专门用于初始化双向 LSTM 所需的隐藏状态。
                                         # self：在 Python 类方法里是约定俗成的参数，代表类的实例本身，借助它能够访问和修改对象的属性与方法。
                                         # a_fea：音频特征输入，是一个三维张量，形状为 [bs, seg_num, a_dim]，其中 bs 是批量大小，seg_num 是每个样本的分段数量，a_dim 是音频特征的维度。
                                         # v_fea：视频特征输入，在这个方法中虽然传入了该参数，但实际未使用。
        bs, seg_num, a_dim = a_fea.shape # a_fea.shape 返回 a_fea 张量的形状，是一个元组。通过解包操作，将元组中的值分别赋给 bs、seg_num 和 a_dim 三个变量。
        hidden_a = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda()) # torch.zeros(2, bs, a_dim)：创建一个全零的三维张量，形状为 [2, bs, a_dim]。这里的 2 是因为使用了双向 LSTM，分别对应前向和后向两个方向；bs 是批量大小；a_dim 是音频特征的维度。
                                                                                        # .cuda()：将创建的全零张量移动到 GPU 上进行计算，以加快计算速度。
                                                                                        # hidden_a 是一个元组，包含两个元素，分别是隐藏状态和细胞状态，二者都是形状为 [2, bs, a_dim] 的全零张量。
        hidden_v = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda()) # 与初始化音频 LSTM 的隐藏状态和细胞状态类似，这里为视频 LSTM 初始化隐藏状态和细胞状态，同样是形状为 [2, bs, a_dim] 的全零张量，并移动到 GPU 上。
        return hidden_a, hidden_v # 将初始化好的音频 LSTM 的隐藏状态和细胞状态 hidden_a 以及视频 LSTM 的隐藏状态和细胞状态 hidden_v 作为元组返回。

    def forward(self, a_fea, v_fea):
        # a_fea, v_fea: [bs, 10, 128]
        # 初始化隐藏状态
        hidden_a, hidden_v = self.init_hidden(a_fea, v_fea) # 调用 self.init_hidden 方法来初始化 Bi - LSTM 的隐藏状态。init_hidden 方法在类中定义，用于创建初始的隐藏状态和细胞状态，这里会分别为音频和视频的 Bi - LSTM 初始化对应的隐藏状态。
                                                            # hidden_a 和 hidden_v 是两个元组，每个元组包含两个张量，分别对应隐藏状态和细胞状态。
        # Bi-LSTM for temporal modeling
        # 扁平化参数
        self.lstm_video.flatten_parameters() # .contiguous() flatten_parameters 方法用于将 LSTM 层的参数进行扁平化处理，以提高内存使用效率和计算速度。特别是在使用 GPU 进行计算时，扁平化参数可以避免在计算过程中出现内存碎片化的问题。
        self.lstm_audio.flatten_parameters() # 这里分别对视频和音频的 Bi - LSTM 层调用该方法。
        # 前向传播通过 Bi - LSTM
        lstm_audio, hidden1 = self.lstm_audio(a_fea, hidden_a) # 将输入的音频特征 a_fea 和初始化的隐藏状态 hidden_a 输入到 self.lstm_audio 中进行前向传播，得到输出 lstm_audio 和更新后的隐藏状态 hidden1。
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v) # 同理，将视频特征 v_fea 和初始化的隐藏状态 hidden_v 输入到 self.lstm_video 中进行前向传播，得到输出 lstm_video 和更新后的隐藏状态 hidden2。
                                                               # lstm_audio 和 lstm_video 的形状通常为 [bs, 10, 2 * hidden_dim]，因为是双向 LSTM，输出的特征维度会翻倍。
# 这段代码的主要作用是对输入的音频和视频特征进行双向 LSTM 处理，以捕捉序列数据中的时间依赖关系。通过双向 LSTM，模型可以同时考虑到序列的过去和未来信息，从而更好地理解音频和视频在时间维度上的动态变化。最终，返回经过 Bi - LSTM 处理后的音频和视频特征，用于后续的任务，如分类或特征融合。
        return lstm_audio, lstm_video


class PSP(nn.Module):
    """Postive Sample Propagation module"""

    def __init__(self, a_dim=256, v_dim=256, hidden_dim=256, out_dim=256):
        super(PSP, self).__init__()
        self.v_L1 = nn.Linear(v_dim, hidden_dim, bias=False) # self.v_L1 和 self.v_L2：这两个线性层将视频特征从输入维度 v_dim 映射到隐藏维度 hidden_dim。它们在 forward 方法中分别用于生成视频特征的两个分支（v_branch1 和 v_branch2），用于后续的注意力计算。
        self.v_L2 = nn.Linear(v_dim, hidden_dim, bias=False) # bias=False：不使用偏置项，简化模型结构。
        self.v_fc = nn.Linear(v_dim, out_dim, bias=False) # self.v_fc：最终的视频特征线性层，将视频特征从 v_dim 映射到输出维度 out_dim。它在 forward 方法的最后阶段使用，用于生成最终的视频特征表示。
        self.a_L1 = nn.Linear(a_dim, hidden_dim, bias=False) # self.a_L1 和 self.a_L2：这两个线性层将音频特征从输入维度 a_dim 映射到隐藏维度 hidden_dim。它们在 forward 方法中分别用于生成音频特征的两个分支（a_branch1 和 a_branch2），用于后续的注意力计算。
        self.a_L2 = nn.Linear(a_dim, hidden_dim, bias=False) # bias=False：同样不使用偏置项。
        self.a_fc = nn.Linear(a_dim, out_dim, bias=False) # self.a_fc：最终的音频特征线性层，将音频特征从 a_dim 映射到输出维度 out_dim。它在 forward 方法的最后阶段使用，用于生成最终的音频特征表示。
        self.activation = nn.ReLU() # nn.ReLU()：初始化一个修正线性单元（ReLU）激活函数，定义为 f(x) = max(0, x)。ReLU 可以引入非线性，帮助模型学习复杂的模式。
        # self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU() # 再次初始化一个 ReLU 激活函数。虽然与 self.activation 功能相同，但可能在网络的不同位置使用，以保持代码的清晰性。
        self.dropout = nn.Dropout(p=0.1) # default=0.1，注释 default=0.1：表明 Dropout 的默认丢弃率为 10%。
                                         # nn.Dropout(p=0.1)：初始化一个 Dropout 层，用于防止过拟合。在训练时，该层会以概率 p=0.1 随机将输入的某些元素置为 0，迫使模型学习更鲁棒的特征。
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6) # 层归一化（Layer Normalization）
                                                          # nn.LayerNorm(out_dim)：初始化一个层归一化层，对输入的最后一个维度 out_dim 进行归一化。层归一化通过计算每个样本的均值和方差来标准化输入，有助于加速训练并提高模型稳定性。
                                                          # eps=1e-6：一个很小的数值，用于避免分母为零的情况，确保计算的稳定性。

        layers = [self.v_L1, self.v_L2, self.a_L1, self.a_L2, self.a_fc, self.v_fc]
        self.init_weights(layers)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)
            # nn.init.orthogonal(layer.weight)
            # nn.init.kaiming_normal_(layer.weight, mode='fan_in')

    def forward(self, a_fea, v_fea, thr_val):
        # a_fea: [bs, 10, 256]
        # v_fea: [bs, 10, 256]
        # thr_val: the hyper-parameter for pruing process
        v_branch1 = self.dropout(self.activation(self.v_L1(v_fea))) #[bs, 10, hidden_dim]
                                                                    # self.v_L1 是一个线性层，用于对输入的视频特征 v_fea 进行线性变换，将其从维度 256 映射到 hidden_dim。
                                                                    # self.activation 是激活函数，这里使用的是 ReLU 函数，用于引入非线性因素，增强模型的表达能力。
                                                                    # self.dropout 是 Dropout 层，以概率 0.1 随机丢弃一些神经元的输出，防止模型过拟合。
                                                                    # 最终得到的 v_branch1 是经过线性变换、激活函数和 Dropout 操作后的视频特征分支，形状为 [bs, 10, hidden_dim]。
        v_branch2 = self.dropout(self.activation(self.v_L2(v_fea))) # 与 v_branch1 的处理过程类似，使用另一个线性层 self.v_L2 对输入的视频特征 v_fea 进行线性变换，再经过激活函数和 Dropout 操作，得到另一个视频特征分支 v_branch2。
        a_branch1 = self.dropout(self.activation(self.a_L1(a_fea))) # self.a_L1 是一个线性层，用于对输入的音频特征 a_fea 进行线性变换，将其从维度 256 映射到 hidden_dim。
                                                                    # 经过激活函数和 Dropout 操作后，得到音频特征分支 a_branch1，形状为 [bs, 10, hidden_dim]。
        a_branch2 = self.dropout(self.activation(self.a_L2(a_fea))) # 与 a_branch1 的处理过程类似，使用另一个线性层 self.a_L2 对输入的音频特征 a_fea 进行线性变换，再经过激活函数和 Dropout 操作，得到另一个音频特征分支 a_branch2。

        beta_va = torch.bmm(v_branch2, a_branch1.permute(0, 2, 1)) # row(v) - col(a), [bs, 10, 10]
                                                                   # torch.bmm 是 PyTorch 中的批量矩阵乘法函数，用于在批量数据上执行矩阵乘法。
                                                                   # v_branch2 是视频特征经过处理后的张量，形状为 [bs, 10, hidden_dim]，其中 bs 表示批量大小，10 表示时间步长，hidden_dim 是特征维度。
                                                                   # a_branch1 是音频特征经过处理后的张量，形状同样为 [bs, 10, hidden_dim]。
                                                                   # a_branch1.permute(0, 2, 1) 对 a_branch1 张量的第 1 维和第 2 维进行交换，使其形状变为 [bs, hidden_dim, 10]。
                                                                   # 通过 torch.bmm(v_branch2, a_branch1.permute(0, 2, 1)) 进行批量矩阵乘法，得到 beta_va 张量，其形状为 [bs, 10, 10]。这个张量表示每个视频时间步与每个音频时间步之间的相关性得分。
        beta_va /= torch.sqrt(torch.FloatTensor([v_branch2.shape[2]]).cuda()) # v_branch2.shape[2] 获取 v_branch2 张量的最后一个维度的大小，即 hidden_dim。
                                                                              # torch.FloatTensor([v_branch2.shape[2]]) 创建一个包含 hidden_dim 的单元素浮点型张量。
                                                                              # torch.sqrt(...) 对该张量取平方根。
                                                                              # beta_va /= ... 对 beta_va 张量进行缩放操作，将其每个元素除以 hidden_dim 的平方根。这样做的目的是防止点积结果过大，避免在反向传播过程中出现梯度不稳定的问题。
# 这段代码的主要功能是对视频到音频的相关性矩阵 beta_va 应用 ReLU 激活函数，引入非线性，然后通过转置操作得到音频到视频的相关性矩阵 beta_av。这两个矩阵后续会用于正样本传播和特征融合，以增强音频和视频特征之间的关联。
        beta_va = F.relu(beta_va) # ReLU 它的定义为 f(x) = max(0, x)
                                  # 此操作会将 beta_va 中所有小于 0 的元素置为 0，仅保留大于等于 0 的元素。这样做的目的是引入非线性，增强模型的表达能力，使得模型能够学习到更复杂的特征表示。
                                  # 经过 ReLU 激活后，beta_va 的形状保持不变，依旧是 [bs, 10, 10]
        beta_av = beta_va.permute(0, 2, 1) # transpose 转置
# 这段代码通过归一化和剪枝操作，对视频到音频的相关性矩阵进行了处理，使得相关性较强的连接得到增强，相关性较弱的连接被抑制。这样处理后的矩阵 gamma_va 可以用于后续的正样本传播，帮助模型更好地捕捉音频和视频特征之间的关联。
        #  计算 beta_va 每一行的总和
        sum_v_to_a = torch.sum(beta_va, dim=-1, keepdim=True) # dim=-1 表示沿着张量的最后一个维度进行求和。在 beta_va 的形状 [bs, 10, 10] 中，最后一个维度的大小为 10。
                                                              # keepdim=True 表示保持求和后的维度数量不变，即求和结果的形状为 [bs, 10, 1]。这里的 sum_v_to_a 存储了 beta_va 中每个 [bs, 10] 矩阵每行元素的总和。
        # 对 beta_va 进行归一化处理
        beta_va = beta_va / (sum_v_to_a + 1e-8) # [bs, 10, 10] # 这行代码对 beta_va 进行归一化操作，将 beta_va 中的每个元素除以其所在行的元素总和。
        # 根据阈值 thr_val 进行剪枝操作
        gamma_va = (beta_va > thr_val).float() * beta_va # (beta_va > thr_val) 是一个布尔型张量，其元素为 True 或 False，表示 beta_va 中对应元素是否大于阈值 thr_val。
                                                         # .float() 将布尔型张量转换为浮点型张量，其中 True 转换为 1.0，False 转换为 0.0。
                                                         # 最后将转换后的张量与 beta_va 逐元素相乘，这一步的作用是将 beta_va 中小于阈值 thr_val 的元素置为 0，只保留大于阈值的元素，从而筛选出相关性较强的部分。得到的 gamma_va 形状仍然是 [bs, 10, 10]。
        # 计算剪枝后 gamma_va 每一行的总和
        sum_v_to_a = torch.sum(gamma_va, dim=-1, keepdim=True)  # l1-normalization
                                                                # 再次对筛选后的 gamma_va 沿着最后一个维度进行求和，同样保持维度数量不变，得到形状为 [bs, 10, 1] 的 sum_v_to_a，用于后续的归一化操作。
                                                                # 由于在第三个维度上进行求和操作，该维度上的 10 个元素被加总为一个值，因此该维度的大小变为 1。而 keepdim=True 保证了这个维度不会被去掉，最终 sum_v_to_a 的形状就是 [bs, 10, 1]。
        # 对剪枝后的 gamma_va 进行归一化处理
        gamma_va = gamma_va / (sum_v_to_a + 1e-8) # 将剪枝后的 gamma_va 中的每个元素除以对应行的总和，实现对每一行的归一化。
                                                  # 加上 1e-8 同样是为了避免除零错误。
                                                  # 最终得到的 gamma_va 每一行元素的和为 1，形状为 [bs, 10, 10]。

        sum_a_to_v = torch.sum(beta_av, dim=-1, keepdim=True) # torch.sum 是 PyTorch 中用于计算张量元素总和的函数。
                                                              # dim=-1 表示沿着张量的最后一个维度进行求和。对于形状为 [bs, 10, 10] 的 beta_av 张量，就是对每个 [bs, 10] 矩阵的每一行元素求和。
                                                              # keepdim=True 表示保持求和后的维度数量不变，这样 sum_a_to_v 的形状为 [bs, 10, 1]，存储了 beta_av 中每行元素的总和。
        beta_av = beta_av / (sum_a_to_v + 1e-8) # 这行代码对 beta_av 进行归一化操作，将 beta_av 中的每个元素除以其所在行的元素总和。
                                                # 1e-8 是一个极小的常数，其作用是避免出现除以零的情况。经过这一步处理后，beta_av 中每行元素的总和变为 1，实现了行归一化。
        gamma_av = (beta_av > thr_val).float() * beta_av # (beta_av > thr_val) 会生成一个布尔型张量，其元素为 True 或 False，分别表示 beta_av 中对应元素是否大于阈值 thr_val。
                                                         # 最后将转换后的张量与 beta_av 逐元素相乘，这一操作会把 beta_av 中小于阈值 thr_val 的元素置为 0，只保留大于阈值的元素，从而筛选出相关性较强的部分。得到的 gamma_av 形状仍然是 [bs, 10, 10]。
        sum_a_to_v = torch.sum(gamma_av, dim=-1, keepdim=True) # 再次对筛选后的 gamma_av 沿着最后一个维度进行求和，并且保持维度数量不变，得到形状为 [bs, 10, 1] 的 sum_a_to_v，用于后续的归一化操作。
        gamma_av = gamma_av / (sum_a_to_v + 1e-8) # 对 gamma_av 进行再次归一化，将其每行元素的总和变为 1，完成 L1 归一化操作。这样做的目的是确保筛选后的相关性矩阵仍然满足归一化条件，便于后续的特征融合计算。

        a_pos = torch.bmm(gamma_va, a_branch2) # torch.bmm 是 PyTorch 中的批量矩阵乘法函数，用于对批量的矩阵进行乘法运算。
                                               # gamma_va 的形状为 [bs, 10, 10]，表示视频特征和音频特征之间筛选并归一化后的相关性矩阵。
                                               # a_branch2 的形状为 [bs, 10, 256]，是经过处理后的音频特征分支。
                                               # 这行代码将 gamma_va 和 a_branch2 进行批量矩阵乘法，得到 a_pos，其形状为 [bs, 10, 256]。a_pos 表示将音频特征信息按照 gamma_va 所表示的相关性传播到视频特征上得到的结果。
        v_psp = v_fea + a_pos # v_fea 的形状为 [bs, 10, 256]，是原始的视频特征。
                              # 这行代码将原始的视频特征 v_fea 与传播得到的 a_pos 相加，得到更新后的视频特征 v_psp，其形状仍然为 [bs, 10, 256]。这样做的目的是将音频特征的信息融入到视频特征中，增强视频特征的表达能力。
        v_pos = torch.bmm(gamma_av, v_branch1) # gamma_av 的形状为 [bs, 10, 10]，表示音频特征和视频特征之间筛选并归一化后的相关性矩阵。
                                               # v_branch1 的形状为 [bs, 10, 256]，是经过处理后的视频特征分支。
                                               # 这行代码将 gamma_av 和 v_branch1 进行批量矩阵乘法，得到 v_pos，其形状为 [bs, 10, 256]。v_pos 表示将视频特征信息按照 gamma_av 所表示的相关性传播到音频特征上得到的结果。
        a_psp = a_fea + v_pos # a_fea 的形状为 [bs, 10, 256]，是原始的音频特征。
                              # 这行代码将原始的音频特征 a_fea 与传播得到的 v_pos 相加，得到更新后的音频特征 a_psp，其形状仍然为 [bs, 10, 256]。这样做的目的是将视频特征的信息融入到音频特征中，增强音频特征的表达能力。

        v_psp = self.dropout(self.relu(self.v_fc(v_psp))) # self.v_fc(v_psp)：self.v_fc 是一个全连接层（nn.Linear），将输入的视频特征 v_psp 进行线性变换。v_psp 的形状为 [bs, 10, 256]，经过全连接层后，输出的特征维度由全连接层的参数决定。
                                                          # self.relu(...)：self.relu 是一个 ReLU（Rectified Linear Unit）激活函数，对全连接层的输出进行非线性变换。ReLU 函数的作用是将输入小于 0 的部分置为 0，大于 0 的部分保持不变，引入非线性因素，增强模型的表达能力。
                                                          # self.dropout(...)：self.dropout 是一个随机失活层（nn.Dropout），以一定的概率（这里设置为 0.1）随机将输入的部分元素置为 0。随机失活的作用是防止模型过拟合，使模型更加鲁棒。经过这一步处理后，得到更新后的视频特征 v_psp。
        a_psp = self.dropout(self.relu(self.a_fc(a_psp)))
        v_psp = self.layer_norm(v_psp) # self.layer_norm 是一个层归一化层（nn.LayerNorm），对视频特征 v_psp 进行层归一化操作。层归一化的作用是对每个样本的特征进行归一化，使得特征的均值为 0，方差为 1。这样可以加速模型的训练过程，提高模型的稳定性。经过层归一化后，视频特征 v_psp 的分布更加稳定。

        a_psp = self.layer_norm(a_psp)

        a_v_fuse = torch.mul(v_psp + a_psp, 0.5) # v_psp + a_psp 是将这两个特征逐元素相加，这样可以将视频和音频特征的信息进行初步合并，使得每个对应位置的元素都包含了视频和音频在该位置的特征信息。
                                                 # torch.mul 是 PyTorch 中的逐元素乘法函数，用于将输入的张量与一个标量或另一个张量进行逐元素相乘。
                                                 # 这里将相加后的结果与标量 0.5 相乘，相当于对相加后的特征进行了平均操作。通过乘以 0.5，可以避免特征值在相加过程中变得过大
        return a_v_fuse, v_psp, a_psp



class Classify(nn.Module):
    def __init__(self, hidden_dim=256, category_num=28): # Classify 类继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类。
                                                         # hidden_dim：输入特征的维度，默认值为 256。
                                                         # category_num：分类的类别数量，默认值为 28。
        super(Classify, self).__init__()
        self.L1 = nn.Linear(hidden_dim, 64, bias=False) # self.L1：第一个线性层，将输入特征从 hidden_dim 维度映射到 64 维度。bias=False 表示该线性层不使用偏置项。

        self.L2 = nn.Linear(64, category_num, bias=False) # self.L2：第二个线性层，将 64 维度的特征映射到 category_num 维度，以输出每个类别的得分。同样，bias=False 表示不使用偏置项。
        nn.init.xavier_uniform(self.L1.weight) # # 使用 Xavier 均匀分布对 self.L1 和 self.L2 的权重进行初始化。Xavier 初始化有助于在训练过程中保持信号在神经网络中的方差稳定，避免梯度消失或爆炸问题，使模型训练更加稳定。
        nn.init.xavier_uniform(self.L2.weight)
    def forward(self, feature):
        out = F.relu(self.L1(feature)) # out = F.relu(self.L1(feature))：先将输入特征通过第一个线性层 self.L1，然后使用 ReLU（Rectified Linear Unit）激活函数对输出进行非线性变换。ReLU 函数会将所有负值置为 0，大于 0 的值保持不变，这样可以引入非线性因素，增强模型的表达能力。
        out = self.L2(out) # out = self.L2(out)：将经过第一个线性层和 ReLU 激活函数处理后的特征输入到第二个线性层 self.L2，得到每个类别的得分。
        # out = F.softmax(out, dim=-1) 这行代码被注释掉了。如果取消注释，会对输出应用 Softmax 函数，将得分转换为概率分布，使得所有类别的概率之和为 1。
        return out

# AVSimilarity 模块通过对音频和视频特征进行 L2 归一化，然后计算它们之间的余弦相似度，用于衡量音频和视频在特征层面的相似程度。这个相似度信息可以在音频 - 视频事件定位等任务中用于判断音频和视频是否对应同一个事件。
class AVSimilarity(nn.Module):
    """ function to compute audio-visual similarity"""
    def __init__(self,):
        super(AVSimilarity, self).__init__()

    def forward(self, v_fea, a_fea):
        # fea: [bs, 10, 256]
        v_fea = F.normalize(v_fea, dim=-1) # 使用 F.normalize 函数对视频特征 v_fea 进行 L2 归一化，dim=-1 表示在最后一个维度上进行归一化操作，使得每个特征向量的 L2 范数为 1。
        a_fea = F.normalize(a_fea, dim=-1) # 同样对音频特征 a_fea 进行 L2 归一化。
        cos_simm = torch.sum(torch.mul(v_fea, a_fea), dim=-1) # [bs, 10] torch.mul(v_fea, a_fea)：对归一化后的视频特征和音频特征进行逐元素相乘。
                                                              # torch.sum(..., dim=-1)：在最后一个维度上对相乘的结果进行求和，得到音频和视频特征之间的余弦相似度。由于输入特征的形状为 [bs, 10, 256]，经过求和操作后，输出的余弦相似度张量 cos_simm 的形状为 [bs, 10]。
        return cos_simm



class psp_net(nn.Module):
    '''
    System flow for fully supervised audio-visual event localization. 用于全监督的音频 - 视频事件定位的系统流程。
    '''
    def __init__(self, a_dim=128, v_dim=512, hidden_dim=128, category_num=29): # a_dim：音频特征的维度，默认值为 128。
                                                                               # v_dim：视频特征的维度，默认值为 512。
                                                                               # hidden_dim：隐藏层的维度，默认值为 128。
                                                                               # category_num：分类的类别数量，默认值为 29。
        super(psp_net, self).__init__() # 调用父类 nn.Module 的构造函数，确保父类的初始化操作被正确执行。
        self.fa = nn.Sequential( #  音频特征处理模块 self.fa
                                 # nn.Sequential：用于按顺序组合多个神经网络层。
            nn.Linear(a_dim, 256, bias=False), # 第一个线性层，将输入的音频特征从维度 a_dim（默认 128）映射到维度 256，并且不使用偏置项。
            nn.Linear(256, 128, bias=False), # 第二个线性层，将上一层的输出从维度 256 映射回维度 128，同样不使用偏置项。
        )                                    # self.fa 的作用是对输入的音频特征进行特征变换和提取。
        self.fv = nn.Sequential( # 视频特征处理模块 self.fv
                                 # 与 self.fa 类似，self.fv 也是一个由两个线性层组成的序列。
            nn.Linear(v_dim, 256, bias=False), # 第一个线性层，将输入的视频特征从维度 v_dim（默认 512）映射到维度 256，不使用偏置项。
            nn.Linear(256, 128, bias=False), # 第二个线性层，将上一层的输出从维度 256 映射到维度 128，不使用偏置项。
        )                                    # self.fv 的作用是对输入的视频特征进行特征变换和提取，使其与音频特征的维度一致，便于后续的融合和处理。
        self.linear_v = nn.Linear(v_dim, a_dim) # 一个线性层，将视频特征从 v_dim 维度映射到 a_dim 维度。
        self.relu = nn.ReLU() # ReLU 激活函数，用于引入非线性。
        self.attention = AVGA(v_dim=v_dim) # AVGA 模块，用于音频引导的视觉注意力机制，根据音频特征对视频特征进行加权。
        self.lstm_a_v = LSTM_A_V(a_dim=a_dim, v_dim=hidden_dim, hidden_dim=hidden_dim) # 使用双向 LSTM 对音频和视频特征进行时序建模。
        self.psp = PSP(a_dim=a_dim*2, v_dim=hidden_dim*2) # PSP 模块，用于正样本传播，融合音频和视频特征。
        self.av_simm = AVSimilarity() # AVSimilarity 模块，用于计算音频和视频特征之间的相似度。

# self.v_classify 和 self.a_classify：Classify 模块，分别用于对视频和音频特征进行分类。
        self.v_classify = Classify(hidden_dim=256)
        self.a_classify = Classify(hidden_dim=256)

# 定义最终分类层
        self.L1 = nn.Linear(2*hidden_dim, 64, bias=False) # 一个两层的线性分类器，将融合后的特征从 2*hidden_dim 维度映射到 64 维度，再从 64 维度映射到 category_num 维度，用于最终的分类任务。
        self.L2 = nn.Linear(64, category_num, bias=False)

        self.L3 = nn.Linear(256, 64) # 另一个两层的线性分类器，将 256 维度的特征映射到 64 维度，再从 64 维度映射到 2 维度，可能用于另一个分类任务。
        self.L4 = nn.Linear(64, 2)
        # layers = [self.L1, self.L2]
        # 初始化权重
        layers = [self.L1, self.L2, self.L3, self.L4] # 选择要初始化权重的层，这里选择了 self.L1、self.L2、self.L3 和 self.L4。
        self.init_layers(layers)

# init_layers 方法接收一个层列表作为参数，对列表中的每个层的权重使用 Xavier 均匀分布进行初始化，有助于在训练过程中保持信号在神经网络中的方差稳定，避免梯度消失或爆炸问题。
    def init_layers(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, audio, video, thr_val): # thr_val：PSP（Positive Sample Propagation）模块中的剪枝过程的超参数。
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        bs, seg_num, H, W, v_dim = video.shape # 获取视频特征的形状信息
                                               # 通过 video.shape 获取视频特征的形状信息，并将其分别赋值给 bs（批量大小）、seg_num（片段数量）、H（特征图高度）、W（特征图宽度）和 v_dim（视频特征维度）。
        fa_fea = self.fa(audio) # 音频特征的预处理
                                # self.fa 是一个由两个线性层组成的序列，用于对音频特征进行预处理。经过 self.fa 处理后，音频特征 fa_fea 的维度可能会发生变化。
        video_t = self.attention(fa_fea, video) # [bs, 10, 512] 音频引导的视觉注意力机制
                                                # self.attention 是一个 AVGA 类的实例，用于实现音频引导的视觉注意力机制。它将预处理后的音频特征 fa_fea 和原始视频特征 video 作为输入，输出经过注意力加权的视频特征 video_t，形状为 [bs, 10, 512]。
        video_t = self.fv(video_t) # [bs, 10, 128] 视频特征的进一步处理
                                   # self.fv 是一个由两个线性层组成的序列，用于对经过注意力加权的视频特征 video_t 进行进一步处理。处理后，视频特征的维度变为 128。
        lstm_audio, lstm_video = self.lstm_a_v(fa_fea, video_t) # 双向 LSTM 时间建模
                                                                # self.lstm_a_v 是一个 LSTM_A_V 类的实例，包含两个双向 LSTM 层，分别对音频特征 fa_fea 和视频特征 video_t 进行时间建模。输出 lstm_audio 和 lstm_video 是经过 LSTM 处理后的音频和视频特征。
        fusion, final_v_fea, final_a_fea = self.psp(lstm_audio, lstm_video, thr_val=thr_val) # [bs, 10, 256] 正样本传播模块
                                                                                             # self.psp 是一个 PSP 类的实例，用于实现正样本传播模块。它将经过 LSTM 处理后的音频和视频特征 lstm_audio、lstm_video 以及剪枝超参数 thr_val 作为输入，输出融合特征 fusion、最终视频特征 final_v_fea 和最终音频特征 final_a_fea，形状均为 [bs, 10, 256]。
        cross_att = self.av_simm(final_v_fea, final_a_fea) # 计算视听相似度
                                                           # self.av_simm 是一个 AVSimilarity 类的实例，用于计算最终视频特征 final_v_fea 和最终音频特征 final_a_fea 之间的相似度。输出 cross_att 是一个形状为 [bs, 10] 的张量，表示每个片段的视听相似度。

        out = self.relu(self.L1(fusion)) # 分类输出 首先，对融合特征 fusion 应用线性层 self.L1 和 ReLU 激活函数。
        out = self.L2(out) # [bs, 10, 29] 然后，将结果输入到线性层 self.L2 中，得到最终的分类输出 out，形状为 [bs, 10, 29]，表示每个片段属于 29 个类别的得分。
        return fusion, out, cross_att # 返回融合特征 fusion、分类输出 out 和视听相似度 cross_att。


