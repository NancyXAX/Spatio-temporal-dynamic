from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.optimizers.lr_finder import plt
from torch.nn import Conv1d, MaxPool1d, Linear, GRU, TransformerEncoder, TransformerEncoderLayer
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
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
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        # self.fc = nn.Linear(kernel_size, out_size)  # 用于调整特征维度
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size * 2, kernel_size),  # 双向GRU输出的特征数是kernel_size * 2
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):
        b, k, d = raw.shape  # b=batch_size, k=sequence_length, d=feature_size

        # 确保输入在正确的设备上
        raw = raw.to(device)

        if d % self.kernel_size != 0:
            raise ValueError(f"Cannot reshape: feature dim {d} not divisible by kernel_size {self.kernel_size}")

        # x = raw.view((b * k, -1, self.kernel_size))
        x = raw.view(b * k, d // self.kernel_size, self.kernel_size)

        # GRU 层的输入尺寸需要是 (batch_size, sequence_length, input_size)
        x, h = self.gru(x)

        x = x[:, -1, :]  # 选择序列的最后一个时间步的输出

        x = x.view((b, k, -1))  # 重塑形状，以适应后续的全连接层

        x = self.linear(x)
        return x


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

        # Normalization layers
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        # Final linear layer
        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

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

        # Final reshape for linear processing
        # Shape: (B*K,C,L) -> (B,K,-1) where -1 is inferred dimension
        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


# class SeqenceModel(nn.Module):
#
#     def __init__(self, model_config, roi_num=360, time_series=512):
#         super().__init__()
#
#         if model_config['extractor_type'] == 'cnn':
#             self.extract = ConvKRegion(
#                 out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
#                 time_series=time_series, pool_size=4, )
#         elif model_config['extractor_type'] == 'gru':
#             self.extract = GruKRegion(
#                 out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
#                 layers=model_config['num_gru_layers'], dropout=model_config['dropout'])
#
#         self.linear = nn.Sequential(
#             Linear(model_config['embedding_size']*roi_num, 256),
#             nn.Dropout(model_config['dropout']),
#             nn.ReLU(),
#             Linear(256, 32),
#             nn.Dropout(model_config['dropout']),
#             nn.ReLU(),
#             Linear(32, 2)
#         )
#
#     def forward(self, x):
#         # x = self.extract(x)
#         x = self.extract(x.to(device))  # Ensure input is on the same device
#         x = x.flatten(start_dim=1)
#         x = self.linear(x)
#         return x


# CNN+GRU
class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=360, time_series=512):
        super().__init__()

        # 双流特征提取器
        self.cnn_extractor = ConvKRegion(
            out_size=model_config['embedding_size'],
            kernel_size=model_config['window_size'],
            time_series=time_series,
            pool_size=4)

        self.gru_extractor = GruKRegion(
            out_size=model_config['embedding_size'],
            kernel_size=model_config['window_size'],
            layers=model_config['num_gru_layers'],
            dropout=model_config['dropout'])

        # 交叉注意力融合模块
        self.fusion_linear1 = nn.Linear(model_config['embedding_size'] * 2, 256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.fusion_linear2 = nn.Linear(256, model_config['embedding_size'])

    def forward(self, x):
        x_cnn = self.cnn_extractor(x)  # (B, K, E)
        x_gru = self.gru_extractor(x)  # (B, K, E)

        # 动态特征融合
        concatenated = torch.cat([x_cnn, x_gru], dim=-1)  # (B, K, 2E)
        x = self.fusion_linear1(concatenated)
        x = self.leaky_relu(x)
        attn_output, _ = self.attention(x, x, x)  # (B, K, 256)
        x = self.fusion_linear2(attn_output)  # (B, K, E)

        # 残差连接（例如与CNN特征相加）
        x_fused = x + x_cnn

        return x_fused  # 输出形状 (B, K, E)

    def _dynamic_fusion(self, cnn_feat, gru_feat):
        # 拼接特征 [B,K,E] + [B,K,E] -> [B,K,2E]
        concatenated = torch.cat([cnn_feat, gru_feat], dim=-1)

        # 通过多头注意力加权
        attn_output, _ = self.fusion_layer(concatenated, concatenated, concatenated)

        # 残差连接
        return concatenated + attn_output


class SeqenceModel2(nn.Module):
    def __init__(self, model_config, roi_num=360, time_series=512):
        super().__init__()
        # 双流特征提取器
        self.cnn_extractor = ConvKRegion(
            out_size=model_config['embedding_size'],
            kernel_size=model_config['window_size'],
            time_series=time_series,
            pool_size=4)

        self.gru_extractor = GruKRegion(
            out_size=model_config['embedding_size'],
            kernel_size=model_config['window_size'],
            layers=model_config['num_gru_layers'],
            dropout=model_config['dropout'])

        # 1. 增加模态间门控机制
        self.gate = nn.Sequential(
            nn.Linear(model_config['embedding_size'] * 2, model_config['embedding_size']),
            nn.Sigmoid()
        )

        # 2. 改进注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=model_config['embedding_size'],
            num_heads=8,
            dropout=model_config['dropout'],
            batch_first=True
        )

        # 3. 增加层归一化
        self.norm1 = nn.LayerNorm(model_config['embedding_size'])
        self.norm2 = nn.LayerNorm(model_config['embedding_size'])

        # 4. 增加模态专用适配层
        self.cnn_adapter = nn.Sequential(
            nn.Linear(model_config['embedding_size'], model_config['embedding_size']),
            nn.LeakyReLU(0.2)
        )
        self.gru_adapter = nn.Sequential(
            nn.Linear(model_config['embedding_size'], model_config['embedding_size']),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # 特征适配
        x_cnn = self.cnn_adapter(self.cnn_extractor(x))
        x_gru = self.gru_adapter(self.gru_extractor(x))

        # 加入维度检查
        assert x_cnn.shape == x_gru.shape, f"维度不匹配: CNN {x_cnn.shape} vs GRU {x_gru.shape}"

        # 门控融合
        gate_weights = self.gate(torch.cat([x_cnn, x_gru], dim=-1))
        gated_fusion = gate_weights * x_cnn + (1 - gate_weights) * x_gru

        # 改进的注意力机制
        attn_output, _ = self.attention(gated_fusion, gated_fusion, gated_fusion)
        attn_output = self.norm1(gated_fusion + attn_output)

        # 残差连接
        final_output = self.norm2(attn_output + gated_fusion)
        return final_output


class DynamicGraphGenerator(nn.Module):
    def __init__(self, input_dim, roi_num=360):
        super().__init__()
        # 动态图权重学习
        self.dynamic_weight = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 静态图先验（可加载脑网络模板）
        self.register_buffer('static_adj',
                             torch.randn(roi_num, roi_num).abs().clamp(0, 0.5))

    def forward(self, x):
        # 动态边权重计算
        b, k, d = x.shape
        x_expand = x.unsqueeze(2).expand(-1, -1, k, -1)
        x_tile = x.unsqueeze(1).expand(-1, k, -1, -1)
        pair_feat = torch.cat([x_expand, x_tile], dim=-1)

        dynamic_weights = self.dynamic_weight(pair_feat).squeeze(-1)

        # 结合静态先验
        final_adj = dynamic_weights * self.static_adj[None, :, :]
        return final_adj.unsqueeze(-1)  # 保持维度兼容性


class EnhancedDynamicGraph(nn.Module):
    def __init__(self, input_dim, roi_num=360):
        super().__init__()
        # 引入稀疏连接先验
        self.static_adj = nn.Parameter(
            torch.randn(roi_num, roi_num).abs().clamp(0, 0.5),
            requires_grad=False
        )

        # 高效边权重计算
        self.dynamic_proj = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 图结构正则化
        self.sparse_reg = nn.L1Loss()

    def forward(self, x):
        b, k, d = x.shape

        # 高效成对特征计算
        x_row = x.unsqueeze(2).expand(-1, -1, k, -1)  # (B,K,K,D)
        x_col = x.unsqueeze(1).expand(-1, k, -1, -1)  # (B,K,K,D)
        pair_feat = torch.cat([x_row, x_col], dim=-1)

        # 动态权重计算
        dynamic_weights = self.dynamic_proj(pair_feat).squeeze(-1)  # (B,K,K)

        # 结合静态先验与稀疏约束
        final_adj = dynamic_weights * self.static_adj
        return final_adj.unsqueeze(-1)


class EnhancedDynamicGraph3(nn.Module):
    def __init__(self, input_dim, roi_num=360, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.roi_num = roi_num
        self.input_dim = input_dim

        # 1. 可训练的对称静态先验
        static_init = torch.randn(roi_num, roi_num).abs().clamp(0, 0.5)
        static_init = (static_init + static_init.t()) / 2
        self.static_adj = nn.Parameter(static_init, requires_grad=True)

        # 2. 多头注意力投影层
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)

        # 3. 边特征提取器
        self.edge_feature = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. 增强型边权重计算器
        self.edge_weight = nn.Sequential(
            nn.Linear(128 + input_dim, 256),
            nn.LayerNorm(256),  # ← 改为 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),  # ← 同样改为 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 5. 图结构正则化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(roi_num)

    def compute_attention_scores(self, x):
        b, k, d = x.shape

        # 多头注意力计算
        q = self.query_proj(x)  # [b, k, d]
        k = self.key_proj(x)  # [b, k, d]

        # 计算注意力分数
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)  # [b, k, k]
        attn = F.softmax(attn, dim=-1)

        return attn

    def compute_edge_features(self, x):
        b, k, d = x.shape

        # 计算节点对特征
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)  # [b, k, k, d]
        x_j = x.unsqueeze(1).expand(-1, k, -1, -1)  # [b, k, k, d]

        # 拼接特征
        pair_feat = torch.cat([x_i, x_j], dim=-1)  # [b, k, k, 2d]

        # 提取边特征
        edge_feat = self.edge_feature(pair_feat)  # [b, k, k, 128]

        return edge_feat

    def forward(self, x):
        b, k, d = x.shape

        # 1. 计算注意力分数
        attn_scores = self.compute_attention_scores(x)  # [b, k, k]

        # 2. 计算边特征
        edge_feat = self.compute_edge_features(x)  # [b, k, k, 128]

        # 3. 融合注意力分数和边特征
        attn_scores = attn_scores.unsqueeze(-1)  # [b, k, k, 1]
        attn_features = attn_scores.expand(-1, -1, -1, self.input_dim)  # [b, k, k, d]

        # 3. 合并特征 [b, k, k, 256]
        combined_feat = torch.cat([edge_feat, attn_features], dim=-1)

        # → reshape 成 [b * k * k, 256]，适配 BatchNorm1d
        b, k, _, feat_dim = combined_feat.shape
        combined_feat = combined_feat.view(b * k * k, feat_dim)

        # 4. edge_weight 输出：[b * k * k, 1]
        edge_weights = self.edge_weight(combined_feat)

        # 5. reshape 回 [b, k, k]
        dynamic_weights = edge_weights.view(b, k, k)

        # 6. 强制对称性
        dynamic_weights = (dynamic_weights + dynamic_weights.transpose(1, 2)) / 2

        # 7. 应用dropout和层归一化
        dynamic_weights = self.dropout(dynamic_weights)
        dynamic_weights = self.layer_norm(dynamic_weights)

        # 8. 结合静态先验
        final_adj = dynamic_weights * self.static_adj

        # 9. 归一化邻接矩阵
        row_sum = final_adj.sum(dim=-1, keepdim=True)
        final_adj = final_adj / (row_sum + 1e-8)

        return final_adj.unsqueeze(-1)  # [b, k, k, 1]

    def get_regularization_loss(self):
        """计算正则化损失"""
        # 1. 稀疏性损失
        sparsity_loss = torch.mean(torch.abs(self.static_adj))

        # 2. 对称性损失
        symmetry_loss = torch.mean(torch.abs(self.static_adj - self.static_adj.t()))

        # 3. 平滑性损失
        smoothness_loss = torch.mean(torch.abs(self.static_adj[:-1, :] - self.static_adj[1:, :]))

        return sparsity_loss + 0.1 * symmetry_loss + 0.1 * smoothness_loss


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)   # Outer product of features

        m = torch.unsqueeze(m, -1)    # Add new dimension for graph representation

        return m


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

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
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

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


class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=k)
            for k in [3, 5, 7]
        ])

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            feat = conv(x.transpose(1, 2))
            multi_scale_features.append(feat.transpose(1, 2))

        # 特征融合
        x = torch.cat(multi_scale_features, dim=-1)

        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)

        return x


class AdaptiveGraphGenerator(nn.Module):
    def __init__(self, input_dim, roi_num=360):
        super().__init__()
        self.edge_learner = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 稀疏性约束
        self.sparse_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, k, d = x.shape
        # 计算节点对特征
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)
        x_j = x.unsqueeze(1).expand(-1, k, -1, -1)
        pair_feat = torch.cat([x_i, x_j], dim=-1)

        # 动态边权重
        edge_weights = self.edge_learner(pair_feat).squeeze(-1)

        # 稀疏性约束
        sparsity_loss = torch.mean(torch.abs(edge_weights)) * self.sparse_weight

        return edge_weights.unsqueeze(-1), sparsity_loss


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

        m = m.squeeze(-1)  # 删除最后一个维度
        x = torch.einsum('ijk,ijp->ijp', m, node_feature)
        # x = torch.einsum('ijkl,ijp->ijlp', m, node_feature)

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


class GNNPredictor2(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_input_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256))
        ])

        # 图注意力池化
        self.pool = nn.MultiheadAttention(256, 4, batch_first=True)

        # 动态特征融合
        self.final_fc = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.GELU(),
            nn.Linear(128, 2))

    def forward(self, adj, features):
        # 图卷积
        for layer in self.gcn_layers:
            features = torch.bmm(adj.squeeze(-1), features)
            features = layer(features)

        # 注意力池化
        pooled, _ = self.pool(features, features, features)
        pooled = torch.mean(pooled, dim=1)

        # 特征融合
        return self.final_fc(torch.cat([
            pooled,
            torch.max(features, dim=1)[0]
        ], dim=-1))


class TransformerWithAttention(nn.Module):
    """返回各层注意力矩阵的自定义Transformer"""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderWithAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward
            ) for _ in range(num_layers)
        ])

    def forward(self, src):
        attn_matrices = []
        for layer in self.layers:
            src, attn = layer(src)
            attn_matrices.append(attn)
        return src, torch.stack(attn_matrices, dim=1)  # [B,L,N,N]


class TransformerEncoderWithAttention(nn.Module):
    """返回单层注意力矩阵的Encoder"""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # 自注意力计算
        attn_output, attn_weights = self.self_attn(src, src, src)
        src = self.norm1(src + attn_output)

        # 前馈网络
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + ff_output)

        return src, attn_weights


class GCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360, pool_ratio = 0.7, gcn_layer=3):
        super().__init__()
        self.num_layers_gcn = gcn_layer
        inner_dim = roi_num
        self.roi_num = roi_num
        self.pool_ratio = pool_ratio
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc_shot_cut = nn.Sequential(
                    nn.Linear(inner_dim, 16),
                    nn.LeakyReLU(negative_slope=0.2),
                )
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

        self.bn = torch.nn.BatchNorm1d(16)

        # self.pool = Encoder(input_dim=16, num_head=4, embed_dim=8, is_cls=True)

        self.fcn = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
    def forward(self, m, node_feature):
        bz, node_num = m.shape[0], m.shape[1]


        x_clone = node_feature.clone()
        x_clone[:, :, :] = 0
        x = node_feature
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
        x = self.fcn(x)
        return x, torch.sigmoid(sc), cor_matrix


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, k, d = x.size()

        # 通道注意力
        avg_out = self.fc(self.avg_pool(x.transpose(1, 2)).view(b, d))
        max_out = self.fc(self.max_pool(x.transpose(1, 2)).view(b, d))
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(1)

        return x * scale


# 改进2：层次化Transformer编码
class HierarchicalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.adaptive_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x):
        layer_weights = F.softmax(self.adaptive_weights, dim=0)
        all_outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            all_outputs.append(x * layer_weights[i])

        return torch.sum(torch.stack(all_outputs), dim=0)


class TransformerLayer(nn.Module):
    """Transformer layer to compute multi-head attention scores."""

    def __init__(self, input_dim, num_heads, num_layers, model_config, hidden_dim):
        super(TransformerLayer, self).__init__()
        assert input_dim % num_heads == 0, f"input_dim {input_dim} must be divisible by num_heads {num_heads}"
        # self.hidden_dim = hidden_dim

        # 假设 encoder_layer 是你自定义的 Transformer 编码器层
        # encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        #
        # # 这里，num_layers 应该是编码器层的数量，通常是一个整数
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            # batch_first=False  # 设置 batch_first 为 True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
            # batch_first=True  # 设置 batch_first=True
        )
        self.attn = nn.MultiheadAttention(input_dim , num_heads, batch_first=True)
       
    def forward(self, x):
        # transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
        # Manually reorder the dimensions if batch_first=False
        # Input is (batch_size, seq_len, features), we need to convert it to (seq_len, batch_size, features)
        x = x.transpose(0, 1)  # (seq_len, batch_size, features)

        x = self.transformer_encoder(x)

        # Convert back to (batch_size, seq_len, features)
        x = x.transpose(0, 1)  # (batch_size, seq_len, features)

        return x


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, src):
        # 正常的 Transformer 前向传播过程
        src2 = self.self_attn(src, src, src)[0]
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + src2)
        return src


class NETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        assert model_config['extractor_type'] in ['cnn', 'gru','seq','seq2']

        # 根据配置选择特征提取器
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])
        elif model_config['extractor_type'] == 'seq':
            self.extract = SeqenceModel(model_config, roi_num=roi_num, time_series=time_series)
        elif model_config['extractor_type'] == 'seq2':
            self.extract = SeqenceModel2(model_config, roi_num=roi_num, time_series=time_series)
        else:
            raise ValueError(f"Unsupported extractor type: {model_config['extractor_type']}")

        self.graph_generation = model_config['graph_generation']
        # 根据配置选择图生成方法
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "dynamic":
            self.emb2graph = DynamicGraphGenerator(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation=='dynamic2':
            self.emb2graph= EnhancedDynamicGraph (
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation=='dynamic3':
            self.emb2graph= EnhancedDynamicGraph3 (
                model_config['embedding_size'], roi_num=roi_num)

        # 定义TransformerLayer
        self.transformer_layer = TransformerLayer(
            input_dim=node_feature_dim,
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_transformer_layers'],
            hidden_dim=model_config['hidden_dim'],
            model_config=model_config  # 传递 model_config
        )
        # 自适应Transformer
        # self.transformer_layer = HierarchicalTransformer(
        #     node_feature_dim,
        #     num_heads=model_config['num_heads'],
        #     num_layers=model_config['num_transformer_layers'],
        #     hidden_dim=model_config['hidden_dim'])

        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)
        # self.predictor2 = GNNPredictor2(node_feature_dim, roi_num=roi_num)

        # 方差正则化
        self.var_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, time_seires, nodes):  # , pseudo
        # Ensure tensors are on the correct device
        time_seires = time_seires.to(device)
        nodes = nodes.to(device)

        extracted_features = self.extract(time_seires)

        graph_matrix = self.emb2graph(extracted_features)

        bz, _, _ = nodes.shape

        transformed_node_features = self.transformer_layer(nodes)

        # 计算边方差。计算图结构 m 中每个样本的边方差，并取平均值
        edge_variance = torch.mean(torch.var(graph_matrix.reshape((bz, -1)), dim=1))

        # # 方差正则化
        # edge_var = torch.var(graph_matrix.view(graph_matrix.size(0), -1))
        # var_loss = torch.mean(edge_var) * self.var_weight

        contribution_scores = self.compute_contribution(transformed_node_features)
        merged_features = self.merge_features(transformed_node_features, nodes)

        # predictions = self.predictor(graph_matrix, nodes)
        # Use the transformed features and the graph matrix for prediction

        predictions= self.predictor(graph_matrix, transformed_node_features)
        # predictions = self.predictor2(graph_matrix, merged_features)

        return predictions, graph_matrix, edge_variance

        # return predictions, nodes_scores, attn_matrix, edge_variance#, contribution_scores   #, merged_features


    def compute_contribution(self, transformed_node_features):
        # 占位符方法，按实际需求实现
        return transformed_node_features.mean(dim=1)  # Example

    def merge_features(self, transformed_node_features, nodes):
        # 合并节点特征和其他信息（占位符方法）
        return torch.cat([transformed_node_features, nodes], dim=-1)  # Example

    def _dynamic_fusion(self, x_cnn, x_gru):
        # 拼接特征 [B,K,E] + [B,K,E] -> [B,K,2E]
        concatenated = torch.cat([x_cnn, x_gru], dim=-1)

        # 通过多头注意力加权
        attn_output, _ = self.fusion_layer(concatenated, concatenated, concatenated)

        # 残差连接
        return concatenated + attn_output


class DGTransNet(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        assert model_config['extractor_type'] in ['cnn', 'gru', 'seq', 'seq2']

        # 根据配置选择特征提取器
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])
        elif model_config['extractor_type'] == 'seq':
            self.extract = SeqenceModel(model_config, roi_num=roi_num, time_series=time_series)
        elif model_config['extractor_type'] == 'seq2':
            self.extract = SeqenceModel2(model_config, roi_num=roi_num, time_series=time_series)
        else:
            raise ValueError(f"Unsupported extractor type: {model_config['extractor_type']}")

        self.graph_generation = model_config['graph_generation']
        # 根据配置选择图生成方法
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "dynamic":
            self.emb2graph = DynamicGraphGenerator(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation=='dynamic2':
            self.emb2graph= EnhancedDynamicGraph (
                model_config['embedding_size'], roi_num=roi_num)

        # 定义TransformerLayer
        self.transformer = HierarchicalTransformer(
            node_feature_dim,
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_transformer_layers'],
            hidden_dim=model_config['hidden_dim'],
            # model_config=model_config  # 传递 model_config
        )
        # 时空编码
        # self.encoder = BrainNetworkEncoder(model_config['embedding_size'], 256)

        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)

        # 方差正则化
        self.var_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, time_seires, nodes, pseudo):
        # Ensure tensors are on the correct device
        time_seires = time_seires.to(device)
        nodes = nodes.to(device)

        extracted_features = self.extract(time_seires)

        graph_matrix = self.emb2graph(extracted_features)

        bz, _, _ = nodes.shape

        transformed_node_features = self.transformer(nodes)

        # 计算边方差。计算图结构 m 中每个样本的边方差，并取平均值
        edge_variance = torch.mean(torch.var(graph_matrix.reshape((bz, -1)), dim=1))

        # 方差正则化
        edge_var = torch.var(graph_matrix.view(graph_matrix.size(0), -1))
        var_loss = torch.mean(edge_var) * self.var_weight

        # contribution_scores = self.compute_contribution(transformed_node_features)
        # merged_features = self.merge_features(transformed_node_features, nodes)

        # predictions = self.predictor(graph_matrix, nodes)
        # Use the transformed features and the graph matrix for prediction
        predictions = self.predictor(graph_matrix, transformed_node_features)  #

        return predictions, graph_matrix, var_loss


class NETGEN2(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        assert model_config['extractor_type'] in ['cnn', 'gru','seq','seq2']

        # 根据配置选择特征提取器
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])
        elif model_config['extractor_type'] == 'seq':
            self.extract = SeqenceModel(model_config, roi_num=roi_num, time_series=time_series)
        elif model_config['extractor_type'] == 'seq2':
            self.extract = SeqenceModel2(model_config, roi_num=roi_num, time_series=time_series)
        else:
            raise ValueError(f"Unsupported extractor type: {model_config['extractor_type']}")

        self.graph_generation = model_config['graph_generation']
        # 根据配置选择图生成方法
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "dynamic":
            self.emb2graph = DynamicGraphGenerator(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation=='dynamic2':
            self.emb2graph= EnhancedDynamicGraph (
                model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)

    def forward(self, time_seires, nodes):  # , pseudo
        # Ensure tensors are on the correct device
        time_seires = time_seires.to(device)
        nodes = nodes.to(device)

        extracted_features = self.extract(time_seires)

        graph_matrix = self.emb2graph(extracted_features)

        bz, _, _ = nodes.shape

        # 计算边方差。计算图结构 m 中每个样本的边方差，并取平均值
        edge_variance = torch.mean(torch.var(graph_matrix.reshape((bz, -1)), dim=1))

        predictions= self.predictor(graph_matrix, nodes)
        # predictions = self.predictor2(graph_matrix, merged_features)

        return predictions, graph_matrix, edge_variance
