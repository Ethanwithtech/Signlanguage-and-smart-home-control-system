#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import time
import traceback
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import json

# 检查是否有GPU可用
print("TensorFlow版本:", tf.__version__)
print("GPU是否可用:", tf.config.list_physical_devices('GPU'))

def load_sign_mnist_data():
    """
    加载Sign Language MNIST数据集
    """
    print("正在加载Sign Language MNIST数据集...")
    
    # 数据集路径
    train_data_path = "C:/Users/dyc06/Downloads/archive/sign_mnist_train/sign_mnist_train.csv"
    test_data_path = "C:/Users/dyc06/Downloads/archive/sign_mnist_test/sign_mnist_test.csv"
    
    # 检查文件是否存在
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print(f"错误: 找不到数据集文件")
        print(f"训练集: {train_data_path}")
        print(f"测试集: {test_data_path}")
        return None, None, None, None
    
    # 加载CSV文件
    print("读取CSV文件...")
    try:
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        print(f"训练集形状: {train_df.shape}")
        print(f"测试集形状: {test_df.shape}")
        
        # 分离标签和像素值
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        # 删除标签列
        train_df = train_df.drop('label', axis=1)
        test_df = test_df.drop('label', axis=1)
        
        # 转换为numpy数组并重塑为图像
        x_train = train_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        print(f"x_train形状: {x_train.shape}")
        print(f"y_train形状: {y_train.shape}")
        print(f"x_test形状: {x_test.shape}")
        print(f"y_test形状: {y_test.shape}")
        
        # 数据集信息
        print(f"训练样本数量: {len(x_train)}")
        print(f"测试样本数量: {len(x_test)}")
        print(f"类别数量: {len(np.unique(y_train))}")
        print(f"类别标签: {np.unique(y_train)}")
        
        # 保存几个样本供可视化
        os.makedirs("debug_data", exist_ok=True)
        for i in range(5):
            if i < len(x_train):
                sample_img = x_train[i] * 255
                cv2.imwrite(f"debug_data/sample_{i}_label_{y_train[i]}.jpg", sample_img.reshape(28, 28))
                
        return x_train, y_train, x_test, y_test
    
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

def try_load_existing_model():
    """尝试加载已有模型权重"""
    print("尝试加载已有模型权重...")
    
    weights_path = "C:/Users/dyc06/Desktop/signlanguage/sign_language_model_weights.weights (2).h5"
    
    if not os.path.exists(weights_path):
        print(f"权重文件不存在: {weights_path}")
        return None
    
    try:
        # 创建模型结构
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 尝试直接加载完整模型
        try:
            loaded_model = tf.keras.models.load_model(weights_path)
            print("✓ 成功加载完整模型")
            return loaded_model
        except Exception as e:
            print(f"加载完整模型失败: {str(e)}")
        
        # 尝试加载权重
        try:
            model.load_weights(weights_path)
            print("✓ 成功加载权重")
            return model
        except Exception as e:
            print(f"加载权重失败: {str(e)}")
            
            # 尝试加载权重（跳过不匹配）
            try:
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print("✓ 成功部分加载权重（跳过不匹配）")
                return model
            except Exception as e:
                print(f"部分加载权重失败: {str(e)}")
        
        print("无法加载现有模型，将创建新模型")
        return None
    
    except Exception as e:
        print(f"加载模型过程中出错: {str(e)}")
        traceback.print_exc()
        return None

def create_model():
    """创建ASL手语识别模型"""
    print("正在创建模型...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 打印模型摘要
    model.summary()
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """训练ASL手语识别模型"""
    print("正在训练模型...")
    
    # 设置检查点回调，保存最佳模型
    checkpoint_path = "best_sign_language_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # 提前停止回调，避免过拟合
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )
    
    # 保存最终模型
    final_model_path = "sign_language_model_final.h5"
    model.save(final_model_path)
    
    print(f"训练完成。最佳模型保存在 {checkpoint_path}，最终模型保存在 {final_model_path}")
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n测试准确率: {test_acc:.4f}")
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.savefig("training_history.png")
    print("训练历史已保存为 training_history.png")
    
    return history, checkpoint_path, final_model_path

def create_weights_file(model_path):
    """从模型创建兼容的权重文件"""
    print(f"从 {model_path} 创建兼容的权重文件...")
    
    try:
        # 加载训练好的模型
        model = tf.keras.models.load_model(model_path)
        
        # 保存为权重文件
        weights_path = "sign_language_model_weights.h5"
        model.save_weights(weights_path)
        print(f"权重文件保存在 {weights_path}")
        
        # 为兼容性创建一个h5py格式的权重文件
        compat_weights_path = "sign_language_model_weights.weights.h5"
        model.save_weights(compat_weights_path)
        print(f"兼容性权重文件保存在 {compat_weights_path}")
        
        # 另外创建一个完整模型文件，命名为原始文件名以便兼容
        target_path = "C:/Users/dyc06/Desktop/signlanguage/sign_language_model_weights.weights (2).h5.new"
        model.save(target_path)
        print(f"完整模型保存在 {target_path}")
        
        # 创建模型配置文件
        config = {
            "model_path": model_path,
            "weights_path": weights_path,
            "compat_weights_path": compat_weights_path,
            "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_shape": [28, 28, 1],
            "output_shape": 26
        }
        
        with open("model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("模型配置已保存到 model_config.json")
        
        return True
    except Exception as e:
        print(f"创建权重文件时出错: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("=" * 50)
    print("  ASL手语识别模型训练器 - Sign Language MNIST")
    print("=" * 50)
    
    # 加载真实的MNIST手语数据集
    x_train, y_train, x_test, y_test = load_sign_mnist_data()
    if x_train is None:
        print("无法加载数据集，训练终止")
        return
    
    # 尝试加载现有模型
    model = try_load_existing_model()
    
    # 如果无法加载模型，创建新模型
    if model is None:
        model = create_model()
    
    # 训练模型
    history, best_model_path, final_model_path = train_model(
        model, x_train, y_train, x_test, y_test
    )
    
    # 创建兼容的权重文件
    if create_weights_file(best_model_path):
        print("成功创建兼容的权重文件")
    
    print("\n训练完成。您现在可以使用以下命令启动应用程序:")
    print("python run_app.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        traceback.print_exc() 