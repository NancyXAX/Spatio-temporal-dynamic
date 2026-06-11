import torch


def inner_loss(label, matrixs):  # 类内方差计算
    loss = 0
    for c in [0, 1]:
        if torch.sum(label == c) > 1:
            loss += torch.mean(torch.var(matrixs[label == c], dim=0))
    return loss

def intra_loss(label, matrixs):   # 类间差异计算
    mu0 = torch.mean(matrixs[label == 0], dim=0) if any(label==0) else None
    mu1 = torch.mean(matrixs[label == 1], dim=0) if any(label==1) else None
    return -torch.norm(mu0 - mu1) if (mu0 and mu1) else 0


def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):
    # bz, roi_num, _ = matrixs.shape
    bz, roi_num, _, _ = matrixs.shape  # 如果是四维
    matrixs = matrixs.reshape((bz, -1))  # 展平为[N, ROI^2]

    # 混合标签计算
    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()
    y_0 = 1 - y_1

    # 类内紧凑性约束
    loss = 0.0
    for y, label in zip([y_0, y_1], [0, 1]):
        if (sum_label := torch.sum(y)) > 0:
            center = torch.matmul(y, matrixs) / sum_label
            diff = torch.norm(matrixs - center, p=1, dim=1)
            loss += torch.matmul(y, diff) / (sum_label * roi_num ** 2)

    # 类间可分离性约束
    if sum(y_0) > 0 and sum(y_1) > 0:
        center0 = torch.matmul(y_0, matrixs) / torch.sum(y_0)
        center1 = torch.matmul(y_1, matrixs) / torch.sum(y_1)
        loss += intra_weight * (1 - torch.norm(center0 - center1, p=1) /(roi_num*roi_num))

    return loss