import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from tqdm import tqdm
from loss import Align, Reconstruct, ContrastiveLoss
from torch.optim.lr_scheduler import LambdaLR
from args import args
from model.Encoder_TCN import TemporalConvNet, Encoder_TCN, SiameseClassifier
from utils.figure_help import (
    plot_metric_curve, plot_all_metrics_subplots, save_metrics_to_file
)
from sklearn.metrics import roc_curve

class Trainer():
    def __init__(self, args, model, train_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(self.device)
        # self.model = model.cuda()
        print('model cuda')
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.alpha = args.alpha
        self.beta = args.beta

        self.m = args.m
        self.n = args.n

        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'
        self.contrastiveLoss = ContrastiveLoss(1.5)
    def pretrain(self):
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        eval_loss = float('inf')
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, (data1, data2, labels, seq1_mask, seq2_mask) in enumerate(tqdm_dataloader):
                data1, data2, labels = data1.to(self.device), data2.to(self.device), labels.to(self.device)
                seq1_mask, seq2_mask = seq1_mask.to(self.device), seq2_mask.to(self.device)
                # 合并两个批次数据
                data = torch.cat((data1, data2), dim=0)  # (2 * batch_size, num_features, seq_len)
                data_mask = torch.cat((seq1_mask, seq2_mask), dim=0)  # (2 * batch_size, seq_len)
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(data, data_mask)
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))

            if reconstruct_loss < eval_loss:
                eval_loss = reconstruct_loss
                torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')  # 保存模型参数

    def finetune(self):
        print('finetune')
        if self.args.load_pretrained_model:
            print('load pretrained model')
            state_dict = torch.load(self.save_path + '/pretrain_model.pkl', map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except:
                model_state_dict = self.model.state_dict()
                for pretrain, random_intial in zip(state_dict, model_state_dict):
                    assert pretrain == random_intial
                    if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
                                    'predict_head.bias', 'position.pe.weight']:
                        state_dict[pretrain] = model_state_dict[pretrain]
                self.model.load_state_dict(state_dict)


        # 替换成 SiameseClassifier
        tcn_model = TemporalConvNet(num_inputs=self.args.d_model, num_channels=[64, 128], kernel_size=5)
        encoder_tcn = Encoder_TCN(self.model.input_projection, self.model.encoder, tcn_model)
        self.model = SiameseClassifier(encoder_tcn).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total params: {total_params:,}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)

        loss_history, acc_history, f1_history, auc_history = [], [], [], []
        far_history, frr_history, eer_history = [], [], []
        for epoch in range(self.num_epoch):
            self.model.train()
            loss_sum = 0
            for idx, (data1, data2, labels, mask1, mask2) in enumerate(self.train_loader):
                data1, data2, labels = data1.to(self.device), data2.to(self.device), labels.to(self.device)
                labels = labels.long()
                mask1, mask2 = mask1.to(self.device), mask2.to(self.device)

                self.optimizer.zero_grad()
                out1, out2, logits = self.model(data1, data2, mask1, mask2)
                contrastiveLoss = self.contrastiveLoss(out1, out2, labels)
                criterionLoss = self.criterion(logits, labels)
                # print('contrastiveLoss', contrastiveLoss, 'criterionLoss', criterionLoss)
                loss = self.m * contrastiveLoss + self.n * criterionLoss
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

            print(f'[Epoch {epoch + 1}] train Loss: {loss_sum / len(self.train_loader):.4f}')
            loss_epoch = loss_sum / len(self.train_loader)
            loss_history.append(loss_epoch)

            acc, f1, auc, far, frr, eer, avg_infer_time = self.evaluate_model(self.model, self.test_loader)

            acc_history.append(acc)
            f1_history.append(f1)
            auc_history.append(auc)
            far_history.append(far)
            frr_history.append(frr)
            eer_history.append(eer)
            print(f'→ Eval Acc: {acc:.4f}', f'F1: {f1:.4f}', f'AUC: {auc:.4f}', f'FAR: {far:.4f}', f'FRR: {frr:.4f}', f'EER: {eer:.4f}')

        output_str = (
            f"Total params: {total_params:,}\n"
            f"Average inference time per sample: {avg_infer_time * 1000:.2f} ms\n"
        )

        with open("/work/TouchSeqNet/exp/model_params.txt", "w") as f:
            f.write(output_str)

        picture_dir = os.path.join(self.args.save_path_figure)
        save_metrics_to_file(acc_history, f1_history, auc_history, far_history, frr_history, eer_history, picture_dir)

        plot_metric_curve(loss_history, 'Loss', picture_dir)
        plot_metric_curve(acc_history, 'Accuracy', picture_dir, highlight_best=True)
        plot_metric_curve(f1_history, 'F1 Score', picture_dir, highlight_best=True)
        plot_metric_curve(far_history, 'FAR', picture_dir, highlight_best=True)
        plot_metric_curve(frr_history, 'FRR', picture_dir, highlight_best=True)
        plot_metric_curve(eer_history, 'EER', picture_dir, highlight_best=True)
        plot_all_metrics_subplots(loss_history, acc_history, f1_history, auc_history, picture_dir)

    def evaluate_model(self, model, dataloader):
        model.eval()
        total, correct = 0, 0
        all_labels, all_preds = [], []

        total_infer_time = 0
        infer_count = 0
        with torch.no_grad():
            for data1, data2, labels, mask1, mask2 in dataloader:
                data1, data2, labels = data1.to(self.device), data2.to(self.device), labels.to(self.device)
                mask1, mask2 = mask1.to(self.device), mask2.to(self.device)

                # 记录推理时间（不计数据加载等）
                start = time.time()
                out1, out2, logits = model(data1, data2, mask1, mask2)
                end = time.time()
                total_infer_time += (end - start)

                labels = labels.long()
                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        f1 = f1_score(all_labels, all_preds, average='binary')
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0  # in case of only one class
        accuracy = correct / total if total > 0 else 0
        # 打印推理时间
        avg_infer_time = total_infer_time / infer_count if infer_count > 0 else 0
        print(f'Average inference time per sample: {avg_infer_time * 1000:.2f} ms')

        # Calculate FAR, FRR, and EER
        far, frr, eer = self.calculate_far_frr_eer(all_labels, all_preds)
        return accuracy, f1, auc, far, frr, eer, avg_infer_time

    # New method to calculate FAR, FRR, and EER
    def calculate_far_frr_eer(self, labels, probs):
        # Compute the ROC curve to get false positive rate and true positive rate
        fpr, tpr, thresholds = roc_curve(labels, probs)

        # Calculate FAR = FPR and FRR = 1 - TPR
        far = fpr
        frr = 1 - tpr

        # Find EER (where FAR == FRR)
        eer_idx = np.nanargmin(np.abs(far - frr))  # Index where the difference is smallest
        eer = (far[eer_idx] + frr[eer_idx]) / 2  # EER is the average of FAR and FRR at that point

        return far[eer_idx], frr[eer_idx], eer

    def print_process(self, *x):
        if self.verbose:
            print(*x)