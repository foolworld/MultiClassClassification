#!/usr/bin/env python3
"""
树叶分类 - 主程序
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from common.dataset import load_dataset
from SC import SimpleConvNet
from common.trainers import Trainer


def main():
    print("=== 树叶分类项目启动 ===")

    # 数据路径
    data_path = "./data/dataset1"

    try:
        # 加载数据
        print("正在加载树叶数据集...")
        (x_train, t_train), (x_val, t_val), classes = load_dataset(data_path, img_size=(64, 64))
        print(f"✅ 数据加载成功")
        print(f"   训练集: {x_train.shape}")
        print(f"   验证集: {x_val.shape}")
        print(f"   类别: {classes}")

        # 训练模型
        print("\n=== 开始训练模型 ===")
        network = SimpleConvNet(input_dim=(3, 64, 64), output_size=len(classes), use_dropout=True,
                                dropout_ratio=0.5)  # 使用实际类别数
        trainer = Trainer(
            network=network,
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            learning_rate=0.001,
            batch_size=32,
            epochs=150,  # 训练轮数
            verbose=True,
            use_augmentation=True
        )
        trainer.train()

        #  保存所有参数，包括BN运行时统计量
        save_dict = {}

        # 1. 保存普通参数
        for key, val in network.params.items():
            save_dict[key] = val

        # 直接提取并保存BN层的运行时统计量
        bn_layers = ['BatchNorm1', 'BatchNorm2', 'BatchNorm3_conv', 'BatchNorm3']
        for i, layer_name in enumerate(bn_layers):
            if layer_name in network.layers:
                bn = network.layers[layer_name]
                if bn.running_mean is not None:
                    save_dict[f'bn{i + 1}_running_mean'] = bn.running_mean
                if bn.running_var is not None:
                    save_dict[f'bn{i + 1}_running_var'] = bn.running_var

        # 3. 保存到文件
        np.savez('leaf_model_weights.npz', **save_dict)
        np.save('leaf_classes.npy', classes)

        # 打印保存的参数信息
        print(f"\n 保存了 {len(save_dict)} 个参数到文件，包括：")
        for key in save_dict:
            if 'running' in key:
                print(f"   - {key}: shape={save_dict[key].shape}")

        print("\n🎉 树叶分类程序执行完成！")

    except Exception as e:
        print(f" 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()