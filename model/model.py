from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
import math



def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


# 处理时间序列数据 包含一个双向 GRU 层和一个线性层，用于提取特征并进行降维
class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        # 输入数据维度为 [b, k, d]，b 是批次大小，k 是区域数量，d 是每个区域的时间序列长度
        b, k, d = raw.shape

        # 将输入重塑为 [b*k, -1, kernel_size]
        x = raw.view((b*k, -1, self.kernel_size))

        # GRU 层
        x, h = self.gru(x)

        # 取最后一个时间步的输出作为特征
        x = x[:, -1, :]

        x = x.view((b, k, -1))   # [b, k, -1]

        # 通过线性层进行降维和特征变换
        x = self.linear(x)
        return x    # [b, k, out_size]


# 用于处理时间序列数据。该类主要包含多个卷积层和一个线性层，用于提取特征并进行降维
class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)  # 交换维度

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=360, time_series=512):
        super().__init__()

        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series, pool_size=4, )
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'], dropout=model_config['dropout'])

        self.linear = nn.Sequential(
            Linear(model_config['embedding_size']*roi_num, 256),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(256, 32),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(32, 2)
        )

    def forward(self, x):
        x = self.extract(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()
        # self.sigmoid = torch.nn.Sigmoid()  #

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)
        # m = self.sigmoid(m)  #

        return m



# 用于将输入特征转换为图结构数据
class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        # 初始化两个线性层
        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        # 生成两个矩阵，用于表示节点之间的连接关系
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m


class GNN(nn.Module):
    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)


# 这个模型主要用于处理图数据，通过多层GCN和全连接层逐步提取和降维特征，最终输出分类结果。
class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz,-1)

        return self.fcn(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360, pool_ratio=0.7, gcn_layer=3):
        super().__init__()
        self.num_layers_gcn = gcn_layer
        inner_dim = roi_num
        self.roi_num = roi_num
        self.pool_ratio = pool_ratio
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc_shot_cut = nn.Sequential(
                    nn.Linear(inner_dim, node_input_dim), # 将输入维度扩展为原始维度,原来：16
                    nn.LeakyReLU(negative_slope=0.2), )
        # fc_shot_cut: 一个全连接层，用于快捷连接，包含一个线性层和一个LeakyReLU激活函数。

        # GCN层和归一化层
        for i in range(self.num_layers_gcn):
            if i == self.num_layers_gcn - 1:
                gcn = nn.Sequential(
                    nn.Linear(inner_dim, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 16),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                norm = torch.nn.BatchNorm1d(16)
            else:
                gcn = nn.Sequential(
                    nn.Linear(inner_dim, inner_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                norm = torch.nn.BatchNorm1d(inner_dim)

            self.gcns.append(gcn)
            self.norms.append(norm)

        self.cls = torch.nn.Parameter(torch.zeros(1, 16))
        # cls: 一个可学习的分类参数，初始值为0。

        self.bn = torch.nn.BatchNorm1d(16)
        # bn: 一个批量归一化层，用于归一化特征。

        # self.pool = Encoder(input_dim=16, num_head=4, embed_dim=8, is_cls=True)
        # pool: 一个编码器，用于特征池化，输入维度为16，使用4个头的多头注意力机制，嵌入维度为8
        # Encoder(input_dim=16, num_head=4, embed_dim=8, is_cls=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 16)) #
        # cls_token: 一个可学习的分类标记，初始值为0
        # self.pos_embedding = nn.Parameter(torch.zeros(1, 1+node_num, 16)) # 1+node_num
        self.dropout = nn.Dropout(0.1) #
        # dropout: 一个丢弃层，用于防止过拟合，丢弃概率为0.1

        self.fcn = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

        # fcn: 一个全连接网络，包含三个线性层和两个LeakyReLU激活函数，最终输出维度为2，用于分类任务

    # 定义了一个前向传播函数 forward，用于处理图卷积网络（GCN）的特征提取和池化操作
    def forward(self, m, node_feature):
        bz, node_num = m.shape[0], m.shape[1]

        x_clone = node_feature.clone()
        x_clone[:, :, :] = 0
        x = node_feature
        # 多层GCN处理
        for layer in range(self.num_layers_gcn):
            x = torch.einsum('ijk,ijp->ijp', m, x)
            if layer == self.num_layers_gcn - 1:
                x = self.gcns[layer](x) + self.fc_shot_cut(x_clone)
            else:
                x = self.gcns[layer](x) + x_clone
            x = x.reshape((bz * self.roi_num, -1))
            x = self.norms[layer](x)
            x = x.reshape((bz, self.roi_num, -1))
            x_clone = x.clone()

        self.cls = self.cls.to(device)

        self.bn = self.bn.to(device) #
        self.cls_token = self.cls_token.to(device) #
        # self.pool = Encoder(input_dim=16, num_head=4, embed_dim=8, is_cls=True).to(device)
        self.dropout = self.dropout.to(device) #

        x_in = torch.empty(bz, node_num + 1, 16).to(device)
        for i in range(bz):
            x_in[i] = torch.cat((self.cls, x[i]), 0)  #按维数0拼接（竖着拼）

        out, cor_matrix = self.pool(x_in)

        cor = cor_matrix.clone()
        cor = cor[:, :, 0, :].to(device)

        score = cor[:, 0, :]
        for i in range(3):
            score += cor[:, i + 1, :]

        score = score[:, 1:]
        sc = score.clone()
        score, rank = score.sort(dim=-1, descending=True)

        l = int(node_num * self.pool_ratio)
        x_p = torch.empty(bz, l, 16).to(device)

        x = out[:, 1:, :]
        score = score[:, :l]
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        for i in range(x.shape[0]):
            x_p[i] = x[i, rank[i, :l], :]

        x_p = torch.matmul(score, x_p).squeeze(1)

        x = x_p.view(bz, -1)

        x = self.bn(x) #
        x = self.dropout(x) #
        x = self.cls_token + x #

        x = self.fcn(x)
        return x, torch.sigmoid(sc), cor_matrix


class FBNETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        self.graph_generation = model_config['graph_generation']

        # 根据配置选择特征提取器
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])

        # 根据配置选择图生成方法
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)

        # 预测器
        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)

    def forward(self, t, nodes):
        x = self.extract(t)
        x = F.softmax(x, dim=-1)
        m = self.emb2graph(x) # 图生成

        m = m[:, :, :, 0] # 取图结构 m 的第一个通道。

        bz, _, _ = m.shape

        # 计算边方差。计算图结构 m 中每个样本的边方差，并取平均值
        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))

        # 对图结构 m 和节点特征 nodes 进行预测
        return self.predictor(m, nodes), m, edge_variance


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, roi_num):
        super().__init__()
        self.in_planes = 1
        self.d = roi_num

        self.e2econv1 = E2EBlock(1, 32, roi_num, bias=True)
        self.e2econv2 = E2EBlock(32, 64, roi_num, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out


class FCNet(nn.Module):

    def __init__(self, node_size, seq_len, kernel_size=3):
        super().__init__()

        self.ind1, self.ind2 = torch.triu_indices(node_size, node_size, offset=1)

        seq_len -= kernel_size//2*2
        channel1 = 32
        self.block1 = nn.Sequential(
            Conv1d(in_channels=1, out_channels=channel1,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2

        seq_len -= kernel_size//2*2
        channel2 = 64
        self.block2 = nn.Sequential(
            Conv1d(in_channels=channel1, out_channels=channel2,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2

        seq_len -= kernel_size//2*2
        channel3 = 96
        self.block3 = nn.Sequential(
            Conv1d(in_channels=channel2, out_channels=channel3,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU()
        )

        channel4 = 64
        self.block4 = nn.Sequential(
            Conv1d(in_channels=channel3, out_channels=channel4,
                            kernel_size=kernel_size),
            Conv1d(in_channels=channel4, out_channels=channel4,
                            kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len -= kernel_size//2*2
        seq_len -= kernel_size//2*2
        seq_len //= 2

        self.fc = nn.Linear(in_features=seq_len*channel4, out_features=32)

        self.diff_mode = nn.Sequential(
            nn.Linear(in_features=32*2, out_features=32),
            nn.Linear(in_features=32, out_features=32),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        bz, _, time_series = x.shape

        x = x.reshape((bz*2, 1, time_series))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.reshape((bz, 2, -1))

        x = self.fc(x)

        x = x.reshape((bz, -1))

        diff = self.diff_mode(x)

        return diff


class FCNet2(nn.Module):

    def __init__(self, node_size, seq_len, kernel_size=3):
        super().__init__()

        self.ind1, self.ind2 = torch.triu_indices(node_size, node_size, offset=1)

        # 计算每层后的序列长度
        self.seq_len = seq_len

        channel1 = 32
        self.block1 = nn.Sequential(
            Conv1d(in_channels=1, out_channels=channel1,
                   kernel_size=kernel_size, padding=kernel_size // 2),  # 添加padding
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.seq_len = self.seq_len // 2  # 只考虑maxpool的影响

        channel2 = 64
        self.block2 = nn.Sequential(
            Conv1d(in_channels=channel1, out_channels=channel2,
                   kernel_size=kernel_size, padding=kernel_size // 2),  # 添加padding
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.seq_len = self.seq_len // 2  # 只考虑maxpool的影响

        channel3 = 96
        self.block3 = nn.Sequential(
            Conv1d(in_channels=channel2, out_channels=channel3,
                   kernel_size=kernel_size, padding=kernel_size // 2),  # 添加padding
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU()
        )

        channel4 = 64
        self.block4 = nn.Sequential(
            Conv1d(in_channels=channel3, out_channels=channel4,
                   kernel_size=kernel_size, padding=kernel_size // 2),
            Conv1d(in_channels=channel4, out_channels=channel4,
                   kernel_size=kernel_size, padding=kernel_size // 2),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.seq_len = self.seq_len // 2  # 只考虑maxpool的影响

        self.fc = nn.Linear(in_features=self.seq_len * channel4, out_features=32)

        self.diff_mode = nn.Sequential(
            nn.Linear(in_features=32 * 2, out_features=32),
            #  nn.LeakyReLU(),  # 添加激活函数
            nn.Linear(in_features=32, out_features=32),
            #  nn.LeakyReLU(),  # 添加激活函数
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        # 检查输入维度
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input [batch_size, node_size, time_series], got shape {x.shape}")

        bz = x.shape[0]  # batch size

        # 确保输入数据在设备上
        device = x.device
        x = x.to(device)

        try:
            x = x.reshape((bz * 2, 1, -1))  # 重塑为[bz*2, 1, time_series]

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)

            x = x.reshape((bz, 2, -1))
            x = self.fc(x)
            x = x.reshape((bz, -1))

            diff = self.diff_mode(x)

            return diff

        except Exception as e:
            print(f"Error in FCNet forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            raise e


class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias=False):
        super(SGC, self).__init__()
        self.adj = self.build_adj(nfeat) #
        self.adj = torch.from_numpy(self.adj).float() #

        self.W = nn.Linear(nfeat, nclass, bias=bias)
        self.W.weight = nn.Parameter(self.W.weight.t())  #

        torch.nn.init.xavier_normal_(self.W.weight)
        #
        self.reset_parameters()
        self.dropout = 0
        self.act = nn.Identity()
        self.reset_parameters()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        #

    def forward(self, x):
        x = self.act(x)
        x = torch.spmm(self.adj, x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.normalize(x, p=2, dim=1)

        out = self.W(x)
        return out


# class SGC(nn.Module):
#     """
#     A Simple PyTorch Implementation of Logistic Regression.
#     Assuming the features have been preprocessed with k-step graph propagation.
#     """
#     def __init__(self, nfeat, nclass):
#         super(SGC, self).__init__()
#
#         self.W = nn.Linear(nfeat, nclass)
#
#     def forward(self, x):
#         return self.W(x)