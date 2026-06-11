import os
from typing import overload
import torch
from numpy.lib import save
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss, topk_loss
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        #
        self.best_acc = 0
        self.best_model = None
        self.best_acc_val = 0
        self.best_auc_val = 0
        #
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.pool_ratio = train_config['pool_ratio']

        self.group_loss = train_config['group_loss']
        self.group_loss_weight = train_config.get('group_loss_weight', 0.1)

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']
        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

        # 启用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 梯度累积步数
        self.accumulation_steps = 4

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):

        self.model.train()
        optimizer.zero_grad()

        for i, (data_in, pearson, label) in enumerate(self.train_dataloader):
            label = label.long()

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            with torch.cuda.amp.autocast():
                inputs, nodes, targets_a, targets_b, lam = mixup_data(
                    data_in, pearson, label, 1, device)

                output, learnable_matrix, edge_variance = self.model(inputs, nodes)

            # loss = 2 * mixup_criterion(
            #     self.loss_fn, output, targets_a, targets_b, lam)
            main_loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)
            l2_reg = torch.tensor(0., device=device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param, p=2)

            loss = main_loss + 0.0001 * l2_reg
            #

            if self.group_loss:
                group_loss = mixup_cluster_loss(learnable_matrix,
                                                targets_a, targets_b, lam)
                # loss += mixup_cluster_loss(learnable_matrix,
                #                            targets_a, targets_b, lam)
                loss += self.group_loss_weight * group_loss

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                    torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            # 定期清理缓存
            if i % 100 == 0:
                torch.cuda.empty_cache()

            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []  # 新增：存储所有预测标签

        self.model.eval()

        for data_in, pearson, label in dataloader:
            label = label.long()
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            output, _, _ = self.model(data_in, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()

            # 新增：收集预测标签和真实标签
            _, preds = torch.max(output, 1)
            all_preds += preds.cpu().tolist()
            labels += label.cpu().tolist()

            # labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0

        # ===== 新增：计算SEN和SPE =====
        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:  # 确保是二分类矩阵
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0  # 异常处理
        # =============================

        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]

        # # 计算每个类别的指标
        # precision, recall, f1, _ = precision_recall_fscore_support(
        #     labels, result, average=None)
        #
        # # 计算SEN和SPE
        # sen = recall[1]  # 正类的召回率就是敏感度
        # spe = recall[0]  # 负类的召回率就是特异度
        #
        # # 计算F1分数
        # f1_score = f1[1]  # 正类的F1分数
        # return [auc, sen, spe, f1_score]

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label in self.test_dataloader:
            label = label.long()
            data_in, nodes, label = data_in.to(
                device), nodes.to(device), label.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results, txt, train_loss, test_loss):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        #
        np.save(self.save_path / "train_loss.npy",
                train_loss, allow_pickle=True)
        np.save(self.save_path / "test_loss.npy",
                test_loss, allow_pickle=True)

        with open(self.save_path / "training_info.txt", 'a', encoding='utf-8') as f:
            f.write(txt)
        #
        torch.save(self.model.state_dict(), self.save_path/f"model_{self.best_acc}%.pt")

    def train(self):
        training_process = []

        txt = ''
        train_loss = []
        test_loss = []

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            # 提取SEN和SPE (新增)
            val_sen, val_spe = val_result[5], val_result[6]
            test_sen, test_spe = test_result[5], test_result[6]

            # #
            if self.best_acc <= self.test_accuracy.avg:
                self.best_acc = self.test_accuracy.avg
                self.best_model = self.model
            #

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                # f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.2f}',
                f'Val SEN:{val_sen:.4f}',    # 新增
                f'Val SPE:{val_spe:.4f}',    # 新增
                f'Test SEN:{test_sen:.4f}',
                f'Test SPE:{test_spe:.4f}',
                f'Test F1:{test_result[7]:.4f}'
            ]))

            txt += (f'Epoch[{epoch}/{self.epochs}] '
                    + f'Train Loss:{self.train_loss.avg: .3f} '
                    + f'Test Loss:{self.test_loss.avg: .3f} '
                    + f'Train Accuracy:{self.train_accuracy.avg: .3f}% '
                    + f'Val Accuracy:{self.val_accuracy.avg: .3f}% '
                    + f'Test Accuracy:{self.test_accuracy.avg: .3f}% '
                    + f'Val AUC:{val_result[0]:.3f} '
                    + f'Test AUC:{test_result[0]:.4f}'
                    + f'Test SEN:{test_sen:.4f}'
                    + f'Test SPE:{test_spe:.4f}'
                    + f'Test F1:{test_result[7]:.4f}'
                    + '\n'
                    )

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_accuracy.avg, self.test_loss.avg,
                                     val_result[0], val_sen, val_spe,  # 验证集指标
                                     test_result[0], test_sen, test_spe, test_result[7]  # 测试集指标
                                     ]
                                    + val_result + test_result)
            train_loss.append(self.train_loss.avg)
            test_loss.append(self.test_loss.avg)
        #
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.save_path = self.save_path / Path(f"{self.best_acc: .3f}%_{date_time}")
        #
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process, txt, train_loss, test_loss)


class BiLevelTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)

    def train(self):
        training_process = []
        txt = ''
        train_loss_log = []
        test_loss_log = []
        matrix_epoch = 5

        for epoch in range(self.epochs):
            self.reset_meters()
            if epoch % 10 < matrix_epoch:
                self.train_per_epoch(self.optimizers[0])
            else:
                self.train_per_epoch(self.optimizers[1])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            val_sen, val_spe = val_result[5], val_result[6]
            test_sen, test_spe, test_f1 = test_result[5], test_result[6], test_result[7]

            self.logger.info(" | ".join([
                f'Epoch[{epoch+1}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Val SEN:{val_sen:.4f}',
                f'Val SPE:{val_spe:.4f}',
                f'Test SEN:{test_sen:.4f}',
                f'Test SPE:{test_spe:.4f}',
                f'Test F1:{test_f1:.4f}'
            ]))

            txt += (f'Epoch[{epoch+1}/{self.epochs}] '
                    + f'Train Loss:{self.train_loss.avg: .3f} '
                    + f'Test Loss:{self.test_loss.avg: .3f} '
                    + f'Train Accuracy:{self.train_accuracy.avg: .3f}% '
                    + f'Val Accuracy:{self.val_accuracy.avg: .3f}% '
                    + f'Test Accuracy:{self.test_accuracy.avg: .3f}% '
                    + f'Val AUC:{val_result[0]:.4f} '
                    + f'Test AUC:{test_result[0]:.4f} '
                    + f'Test SEN:{test_sen:.4f} '
                    + f'Test SPE:{test_spe:.4f} '
                    + f'Test F1:{test_f1:.4f}'
                    + '\n'
                    )

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_accuracy.avg, self.test_loss.avg,
                                     val_result[0], val_sen, val_spe,
                                     test_result[0], test_sen, test_spe, test_f1]
                                    + val_result + test_result)
            
            train_loss_log.append(self.train_loss.avg)
            test_loss_log.append(self.test_loss.avg)

        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.save_path = self.save_path / Path(f"{self.best_acc: .3f}%_{date_time}")

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        
        self.save_result(training_process, txt, train_loss_log, test_loss_log)


class GNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.pure_gnn_graph = train_config['pure_gnn_graph']
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for _, pearson, label in self.train_dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            bz, module_num, _ = pearson.shape

            if self.pure_gnn_graph == "uniform":
                graph = torch.ones(
                    (bz, module_num, module_num)).float().to(device)
            elif self.pure_gnn_graph == "pearson":
                graph = torch.abs(pearson)

            graph, nodes, targets_a, targets_b, lam = mixup_data(
                graph, pearson, label, 1, device)

            output = self.model(graph, nodes)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []

        self.model.eval()

        for _, pearson, label in dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            bz, module_num, _ = pearson.shape

            if self.pure_gnn_graph == "uniform":
                graph = torch.ones(
                    (bz, module_num, module_num)).float().to(device)
            elif self.pure_gnn_graph == "pearson":
                graph = torch.abs(pearson)

            output = self.model(graph, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            
            _, preds = torch.max(output, 1)
            all_preds += preds.cpu().tolist()
            labels += label.cpu().tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        
        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0
        
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]
    


class SeqTrain(BasicTrain):
    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for seq_group, _, label in self.train_dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            seq_group, _, targets_a, targets_b, lam = mixup_data(
                seq_group, seq_group, label, 1, device)

            output = self.model(seq_group)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []

        self.model.eval()

        for seq_group, _, label in dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()

            _, preds = torch.max(output, 1)
            all_preds += preds.cpu().tolist()
            labels += label.cpu().tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0

        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0
        
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]
    
   

class BrainCNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for _, pearson, label in self.train_dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            _, nodes, targets_a, targets_b, lam = mixup_data(
                pearson, pearson, label, 1, device)

            output = self.model(nodes)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []

        self.model.eval()

        for _, pearson, label in dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            output = self.model(pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            
            _, preds = torch.max(output, 1)
            all_preds += preds.cpu().tolist()
            labels += label.cpu().tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        
        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0
        
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]
    

class FCNetTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder):
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.generated_graph = []
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):
        self.model.train()

        for seq_group, pearson, label in self.train_dataloader:  # 修改为匹配dataloader的格式
            label = label.long()
            
            # 确保数据在正确的设备上
            seq_group, label = seq_group.to(device), label.to(device)

            try:
                output = self.model(seq_group)
                loss = self.loss_fn(output, label)

                self.train_loss.update_with_weight(loss.item(), label.shape[0])
     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                top1 = accuracy(output, label)[0]
                self.train_accuracy.update_with_weight(top1, label.shape[0])
                
            except Exception as e:
                print(f"Error in training: {str(e)}")
                print(f"Input shape: {seq_group.shape}")
                print(f"Label shape: {label.shape}")
                raise e

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []

        self.model.eval()

        with torch.no_grad():
            for seq_group, pearson, label in dataloader:  # 修改为匹配dataloader的格式
                label = label.long()
                seq_group, label = seq_group.to(device), label.to(device)

                try:
                    output = self.model(seq_group)
                    loss = self.loss_fn(output, label)
             
                    loss_meter.update_with_weight(loss.item(), label.shape[0])
                    
                    top1 = accuracy(output, label)[0]
                    acc_meter.update_with_weight(top1, label.shape[0])
                    result += F.softmax(output, dim=1)[:, 1].tolist()
                    
                    _, preds = torch.max(output, 1)
                    all_preds += preds.cpu().tolist()
                    labels += label.cpu().tolist()
                    
                except Exception as e:
                    print(f"Error in testing: {str(e)}")
                    print(f"Input shape: {seq_group.shape}")
                    print(f"Label shape: {label.shape}")
                    raise e
                
        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        
        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0
        
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]
    

class GINTrain(BasicTrain):
    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):
        self.model.train()

        for seq_group, pearson, label in self.train_dataloader:
            label = label.long()

            seq_group, pearson, label = seq_group.to(device), pearson.to(device), label.to(device)
            
            output = self.model(seq_group)
            loss = self.loss_fn(output, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []

        self.model.eval()

        for seq_group, pearson, label in dataloader:
            label = label.long()
            seq_group, pearson, label = seq_group.to(device), pearson.to(device), label.to(device)

            output = self.model(seq_group)
            loss = self.loss_fn(output, label)
            
            loss_meter.update_with_weight(loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            
            _, preds = torch.max(output, 1)
            all_preds += preds.cpu().tolist()
            labels += label.cpu().tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        
        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0
        
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]


class GATTrain(BasicTrain):
    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):
        self.model.train()

        for seq_group, pearson, label in self.train_dataloader:
            label = label.long()

            seq_group, pearson, label = seq_group.to(device), pearson.to(device), label.to(device)
            
            output = self.model(seq_group)
            loss = self.loss_fn(output, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        all_preds = []

        self.model.eval()

        for seq_group, pearson, label in dataloader:
            label = label.long()
            seq_group, pearson, label = seq_group.to(device), pearson.to(device), label.to(device)

            output = self.model(seq_group)
            loss = self.loss_fn(output, label)
            
            loss_meter.update_with_weight(loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            
            _, preds = torch.max(output, 1)
            all_preds += preds.cpu().tolist()
            labels += label.cpu().tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        
        cm = confusion_matrix(labels, all_preds)
        if cm.size >= 4:
            TN, FP, FN, TP = cm.ravel()
            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
        else:
            SEN, SPE = 0, 0
        
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        f1_score = metric[2]

        return [auc] + list(metric) + [SEN, SPE, f1_score]