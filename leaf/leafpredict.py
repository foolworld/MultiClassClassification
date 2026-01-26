import gradio as gr
import gradio as gr
import numpy as np
from PIL import Image
from SC import SimpleConvNet


def predict(image):
    # 1. 加载模型和类别
    model_weights = np.load('leaf_model_weights.npz')
    classes = np.load('leaf_classes.npy', allow_pickle=True)

    # 2. 创建网络
    network = SimpleConvNet((3, 64, 64), len(classes), False)

    # 3. 创建参数的深拷贝，避免修改原始数据
    loaded_weights = {k: v.copy() for k, v in model_weights.items()}

    # 4. 更新Conv层（用拷贝的值）
    network.layers['Conv1'].W = loaded_weights['W1']
    network.layers['Conv1'].b = loaded_weights['b1']

    network.layers['Conv2'].W = loaded_weights['W2']
    network.layers['Conv2'].b = loaded_weights['b2']

    network.layers['Conv3'].W = loaded_weights['W3_conv']
    network.layers['Conv3'].b = loaded_weights['b3_conv']

    # 5. 更新Affine层
    network.layers['Affine1'].W = loaded_weights['W3']
    network.layers['Affine1'].b = loaded_weights['b3']

    network.layers['Affine2'].W = loaded_weights['W4']
    network.layers['Affine2'].b = loaded_weights['b4']

    # 6. 创建BN层运行时参数的临时拷贝（关键！）
    bn1_running_mean = loaded_weights['bn1_running_mean'].copy()
    bn1_running_var = loaded_weights['bn1_running_var'].copy()
    bn2_running_mean = loaded_weights['bn2_running_mean'].copy()
    bn2_running_var = loaded_weights['bn2_running_var'].copy()
    bn3_running_mean = loaded_weights['bn3_running_mean'].copy()
    bn3_running_var = loaded_weights['bn3_running_var'].copy()
    bn4_running_mean = loaded_weights['bn4_running_mean'].copy()
    bn4_running_var = loaded_weights['bn4_running_var'].copy()

    # 7. 更新BatchNorm层参数（用临时拷贝）
    network.layers['BatchNorm1'].gamma = loaded_weights['gamma_bn1']
    network.layers['BatchNorm1'].beta = loaded_weights['beta_bn1']
    network.layers['BatchNorm1'].running_mean = bn1_running_mean
    network.layers['BatchNorm1'].running_var = bn1_running_var

    network.layers['BatchNorm2'].gamma = loaded_weights['gamma_bn2']
    network.layers['BatchNorm2'].beta = loaded_weights['beta_bn2']
    network.layers['BatchNorm2'].running_mean = bn2_running_mean
    network.layers['BatchNorm2'].running_var = bn2_running_var

    network.layers['BatchNorm3_conv'].gamma = loaded_weights['gamma_bn3_conv']
    network.layers['BatchNorm3_conv'].beta = loaded_weights['beta_bn3_conv']
    network.layers['BatchNorm3_conv'].running_mean = bn3_running_mean
    network.layers['BatchNorm3_conv'].running_var = bn3_running_var

    network.layers['BatchNorm3'].gamma = loaded_weights['gamma_bn3']
    network.layers['BatchNorm3'].beta = loaded_weights['beta_bn3']
    network.layers['BatchNorm3'].running_mean = bn4_running_mean
    network.layers['BatchNorm3'].running_var = bn4_running_var

    # 8. 同时更新network.params（用拷贝的值）
    for key in loaded_weights.keys():
        if key in network.params:
            network.params[key] = loaded_weights[key]

    # 9. 图像预处理
    img = image.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_batch = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :]

    # 10. 预测（使用测试模式！这样BN层不会更新运行时参数）
    predictions = network.predict(img_batch, train_flg=False)

    print(f"预测值范围: [{predictions.min():.2e}, {predictions.max():.2e}]")

    # 11. 如果预测值过大，进行缩放
    max_abs = np.max(np.abs(predictions))
    if max_abs > 100:
        print(f"预测值过大 ({max_abs:.2e})，进行缩放")
        predictions = predictions / (max_abs / 10.0)

    # 12. Softmax
    max_val = np.max(predictions, axis=1, keepdims=True)
    exp_vals = np.exp(predictions - max_val)
    probabilities = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    probabilities = probabilities.astype(np.float32)[0]
    # 13. 返回结果
    results = {classes[i]: float(probabilities[i]) for i in range(len(classes))}

    # 显示top-3
    top_indices = np.argsort(probabilities)[-3:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"Top-{i + 1}: {classes[idx]} - {probabilities[idx]:.2%}")

    return results


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传树叶图片"),
    outputs=gr.Label(num_top_classes=5, label="分类结果"),
    title="树叶分类器",
    description="树叶分类"
)

if __name__ == "__main__":
    demo.launch()