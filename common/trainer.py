# coding: utf-8
import numpy as np
import time
import random  # 新增导入
# 在文件顶部添加
from data_augmentor import DataAugmentor


class Trainer:
    """通用训练器 - 带智能增强触发"""

    def __init__(self, network, x_train, t_train, x_val, t_val,
                 learning_rate=0.01,
                 batch_size=32, epochs=20, verbose=True, use_augmentation=True):
        """
        初始化训练器

        Args:
            network: 神经网络实例 (SimpleConvNet 或 SimpleConvNetButterfly)
            x_train: 训练数据 (N, C, H, W)
            t_train: 训练标签
            x_val: 验证数据 (N, C, H, W)
            t_val: 验证标签
            learning_rate: 学习率
            batch_size: 批大小
            epochs: 训练轮数
            verbose: 是否显示训练进度
        """
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.use_augmentation = use_augmentation  # 新增这行

        # 初始化数据增强器
        if self.use_augmentation:
            self.augmentor = DataAugmentor()
            if self.verbose:
                print("✅ 数据增强已启用")

        # 🆕 新增：智能增强触发配置
        self.augmentation_enabled = use_augmentation
        self.augmentation_triggered = False  # 初始不触发
        self.trigger_acc_threshold = 0.8  # 触发准确率阈值
        self.trigger_stable_epochs = 3  # 需要稳定的轮数
        self.high_acc_epochs = 0  # 记录高准确率轮数

        # 🆕 新增：渐进式增强控制
        self.augmentation_strength = 0.0  # 初始强度为0
        self.augmentation_progress = 0.0  # 增强进度

        # 初始化优化器 - 固定使用Momentum，动量固定为0.9
        from optimizers import Momentum
        self.optimizer = Momentum(lr=learning_rate, momentum=0.9)  # 动量固定

        # 训练记录
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []

        # 训练数据信息
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

    def train_step(self):
        """单次训练步骤"""
        # 随机选择mini-batch
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 🆕 修改：渐进式增强逻辑
        if self.augmentation_triggered and self.use_augmentation:
            # 根据强度决定是否增强当前batch
            if random.random() < self.augmentation_strength:
                x_batch = self.augmentor.augment_batch(x_batch)

        # 计算梯度
        grads = self.network.gradient(x_batch, t_batch)

         #🆕注释掉权值衰减
        weight_decay_lambda = 0.001  # 衰减强度，可以调整
        for key in ['W1', 'W2', 'W3', 'W4']:  # 只对权重矩阵衰减，不对偏置
            if key in grads:  # 确保键存在
                grads[key] += weight_decay_lambda * self.network.params[key]

        # 更新参数
        self.optimizer.update(self.network.params, grads)

        # 计算loss
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        self.current_iter += 1

    def evaluate(self):
        """评估模型准确率 - 使用采样评估避免内存溢出（蝴蝶数据集专用）"""
        if self.x_train is None or self.t_train is None:
            return 0, 0

        # 蝴蝶数据集专用：使用采样评估避免内存溢出
        train_sample_size = min(1000, len(self.x_train))  # 最多评估1000个训练样本
        val_sample_size = min(200, len(self.x_val))  # 最多评估200个验证样本

        # 随机采样
        train_indices = np.random.choice(len(self.x_train), train_sample_size, replace=False)
        val_indices = np.random.choice(len(self.x_val), val_sample_size, replace=False)

        x_train_sample = self.x_train[train_indices]
        t_train_sample = self.t_train[train_indices]
        x_val_sample = self.x_val[val_indices]
        t_val_sample = self.t_val[val_indices]

        # 分批处理采样数据（避免单次处理太多）
        batch_size = 100
        train_acc = 0
        num_batches = (train_sample_size + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_sample_size)
            x_batch = x_train_sample[start_idx:end_idx]
            t_batch = t_train_sample[start_idx:end_idx]
            train_acc += self.network.accuracy(x_batch, t_batch) * len(x_batch)

        train_acc /= train_sample_size

        # 验证集准确率
        val_acc = self.network.accuracy(x_val_sample, t_val_sample)

        return train_acc, val_acc

    def train(self):
        """执行训练"""
        if self.verbose:
            print("开始训练...")
            print(f"训练数据: {self.train_size} 张图片")
            print(f"验证数据: {self.x_val.shape[0]} 张图片")
            print(f"批大小: {self.batch_size}, 总迭代次数: {self.max_iter}")
            print(f"优化器: Momentum (lr={self.optimizer.lr}, momentum=0.9)")
            # 🆕 新增：显示智能增强信息
            if self.augmentation_enabled and not self.augmentation_triggered:
                print(f"🎯 智能增强触发: 训练准确率 > {self.trigger_acc_threshold} 时自动开启")
            print("-" * 60)

        start_time = time.time()
        max_lr = self.optimizer.lr  # 初始学习率作为最大值
        min_lr = max_lr * 0.01  # 最小学习率是最大值的1%
        total_epochs = self.epochs

        for i in range(self.max_iter):
            self.train_step()

            # 每个epoch结束后评估
            if (i + 1) % self.iter_per_epoch == 0:
                self.current_epoch += 1

                # 🆕 保留余弦退火
                #import math
                #progress = self.current_epoch / self.epochs
                #new_lr = max_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
                #self.optimizer.lr = new_lr
                # 🆕 改用阶梯学习率衰减
                if self.current_epoch == 60:
                    self.optimizer.lr = 0.001
                elif self.current_epoch == 100:
                    self.optimizer.lr = 0.0005
                elif self.current_epoch == 140:
                    self.optimizer.lr = 0.0001
                # 其他epoch保持当前学习率不变

                train_acc, val_acc = self.evaluate()
                self.train_acc_list.append(train_acc)
                self.val_acc_list.append(val_acc)

                # 🆕 新增：智能增强触发逻辑

                if (self.augmentation_enabled and
                        not self.augmentation_triggered and
                        train_acc > self.trigger_acc_threshold):
                    self.high_acc_epochs += 1
                    if self.high_acc_epochs >= self.trigger_stable_epochs:
                        self.augmentation_triggered = True
                        # 🆕 新增：触发时使用保守配置
                        self.augmentor.set_config('conservative')
                        if self.verbose:
                            print(
                                f"🚀 数据增强已触发! (训练准确率连续{self.trigger_stable_epochs}轮 > {self.trigger_acc_threshold})")
                            print("📊 使用保守增强配置，逐步增加强度...")
                else:
                    self.high_acc_epochs = 0  # 重置计数

                # 🆕 新增：渐进式增强强度更新
                if self.augmentation_triggered:
                    # 线性增加增强强度，从0.2到1.0
                    self.augmentation_progress = min((self.current_epoch - self.trigger_stable_epochs) / 10, 1.0)
                    self.augmentation_strength = 0.2 + 0.8 * self.augmentation_progress

                if self.verbose:
                    avg_loss = np.mean(self.train_loss_list[-self.iter_per_epoch:])
                    # 🆕 修改：显示增强状态和强度
                    if self.augmentation_triggered:
                        aug_status = f"✅增强({self.augmentation_strength:.1f})"
                    else:
                        aug_status = "⏳等待触发"
                    print(f"Epoch {self.current_epoch:2d}/{self.epochs} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_acc:.4f} | {aug_status}")

        end_time = time.time()
        training_time = end_time - start_time

        if self.verbose:
            print("-" * 60)
            print(f"训练完成! 总耗时: {training_time:.2f}秒")
            if self.val_acc_list:
                best_val_acc = max(self.val_acc_list)
                best_epoch = np.argmax(self.val_acc_list) + 1
                print(f"最佳验证准确率: {best_val_acc:.4f} (第{best_epoch}轮)")

        import matplotlib.pyplot as plt

        # 2. 准备数据
        epochs = range(1, len(self.train_acc_list) + 1)

        # 3. 画第一张图：准确率
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_acc_list, 'b-', label='Train Accuracy')
        plt.plot(epochs, self.val_acc_list, 'r-', label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True)

        # 4. 画第二张图：损失（需要计算平均损失）
        # 计算每个epoch的平均损失
        epoch_losses = []
        for i in range(len(self.train_acc_list)):
            start_idx = i * self.iter_per_epoch
            end_idx = (i + 1) * self.iter_per_epoch
            avg_loss = np.mean(self.train_loss_list[start_idx:end_idx])
            epoch_losses.append(avg_loss)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, epoch_losses, 'g-', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_results.png')  # 保存图片
        plt.show()  # 显示图片

        print("✅ 训练曲线已保存为 training_results.png")

    def get_training_history(self):
        """获取训练历史"""
        return {
            'train_loss': self.train_loss_list,
            'train_acc': self.train_acc_list,
            'val_acc': self.val_acc_list
        }