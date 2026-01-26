import numpy as np
from PIL import Image
from SC import SimpleConvNet
from common.dataset import load_test_dataset

def evaluate_leaf():

    print("=== 树叶模型测试集评估 ===")

    #加载并创建模型
    classes = np.load('leaf_classes.npy',allow_pickle=True)
    weights = np.load('leaf_model_weights.npz',allow_pickle=True)
    network = SimpleConvNet((3,64,64),len(classes),False)

    #加载权重
    loaded_weights = {k:v.copy() for k,v in weights.items()}

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

    print("模型加载完成")
    print(f"类别数{len(classes)}")

    test_path ='competition/dataset/test'
    print(f"测试集路径：{test_path}")

    x_test,y_test,_ = load_test_dataset(test_path,img_size=(64,64))

    test_acc=network.accuracy(x_test,y_test)

    print(f"\n 测试结果：")
    print(f" 准确率：{test_acc:.4f}")

    return test_acc

if __name__=="__main__":
    evaluate_leaf()