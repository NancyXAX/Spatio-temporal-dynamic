import torch


def topk_loss(s,ratio):
    # if ratio > 0.5:
    #     ratio = 1-ratio
    s = s.sort(dim=1).values

    # graph  transformer 在此sigmoid
    # s = torch.sigmoid(s)

    res = -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:-int(s.size(1)*ratio)]+EPS).mean()
    return res

# 计算给定标签 label 和矩阵 matrixs 的内部损失
def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))
        # 检查标签 label 中值为 0 的元素数量是否大于 1，如果是，则计算这些元素对应的矩阵行的方差，并将其均值加到 loss 中

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss

# 计算两类数据之间的内部损失
def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)

    # 计算损失：如果 a 和 b 都不为 None，则计算它们之间的欧氏距离平方的平均值，并返回 1 减去该值；否则返回 0。
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0

# 计算两类数据之间的内部损失
def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):
    # 计算混合标签
    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()
    y_0 = 1 - y_1

    # loss = inner_loss(y_1, matrixs) + intra_loss(y_1, matrixs)
    # 展平矩阵
    # bz, roi_num, _ = matrixs.shape   # 三维
    # bz, roi_num = matrixs.shape  # 二维
    bz, roi_num, _, _ = matrixs.shape  # 如果是四维
    matrixs = matrixs.reshape((bz, -1))
    # 计算总和
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)

    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


