import numpy as np
from collections import OrderedDict
from layers import Convolution, Affine, Pooling, Relu, SoftmaxWithLoss, Dropout


class SimpleConvNet:

    def __init__(self, input_dim=(3, 64, 64), output_size=8, use_dropout=True, dropout_ratio=0.5,
                 use_global_pooling=False):
        self.params = {}
        self.layers = OrderedDict()
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_global_pooling = use_global_pooling  # 新增：是否使用全局平均池化

        C, H, W = input_dim

        # 改为He初始化（适合ReLU）
        # 卷积层1: 输入通道C, 卷积核3x3
        scale1 = np.sqrt(2.0 / (C * 3 * 3))
        self.params['W1'] = scale1 * np.random.randn(12, C, 3, 3)
        self.params['b1'] = np.zeros(12)

        # 卷积层2: 输入通道12, 卷积核3x3
        scale2 = np.sqrt(2.0 / (12 * 3 * 3))
        self.params['W2'] = scale2 * np.random.randn(24, 12, 3, 3)
        self.params['b2'] = np.zeros(24)

        # 新增卷积层3: 输入通道24, 卷积核3x3
        scale3_conv = np.sqrt(2.0 / (24 * 3 * 3))
        self.params['W3_conv'] = scale3_conv * np.random.randn(48, 24, 3, 3)
        self.params['b3_conv'] = np.zeros(48)

        # 计算特征图尺寸
        H_conv1 = (H + 2 * 1 - 3) // 1 + 1  # 64 -> 64
        H_pool1 = H_conv1 // 2  # 64 -> 32
        H_conv2 = (H_pool1 + 2 * 1 - 3) // 1 + 1  # 32 -> 32
        H_pool2 = H_conv2 // 2  # 32 -> 16
        H_conv3 = (H_pool2 + 2 * 1 - 3) // 1 + 1  # 16 -> 16 (新增)
        H_pool3 = H_conv3 // 2  # 16 -> 8 (新增)

        W_conv1 = (W + 2 * 1 - 3) // 1 + 1  # 64 -> 64
        W_pool1 = W_conv1 // 2  # 64 -> 32
        W_conv2 = (W_pool1 + 2 * 1 - 3) // 1 + 1  # 32 -> 32
        W_pool2 = W_conv2 // 2  # 32 -> 16
        W_conv3 = (W_pool2 + 2 * 1 - 3) // 1 + 1  # 16 -> 16 (新增)
        W_pool3 = W_conv3 // 2  # 16 -> 8 (新增)

        # 根据是否使用全局平均池化调整全连接层输入大小
        if self.use_global_pooling:
            fc_input_size = 48  # 全局平均池化后，每个通道一个值
        else:
            fc_input_size = 48 * H_pool3 * W_pool3  # 48 × 8 × 8 = 3072

        # 全连接层也使用He初始化
        scale3 = np.sqrt(2.0 / fc_input_size)
        self.params['W3'] = scale3 * np.random.randn(fc_input_size, 256)
        self.params['b3'] = np.zeros(256)

        scale4 = np.sqrt(2.0 / 256)
        self.params['W4'] = scale4 * np.random.randn(256, output_size)
        self.params['b4'] = np.zeros(output_size)

        # 添加BatchNorm参数
        self.params['gamma_bn1'] = np.ones(12)  # Conv1后的BN
        self.params['beta_bn1'] = np.zeros(12)
        self.params['gamma_bn2'] = np.ones(24)  # Conv2后的BN
        self.params['beta_bn2'] = np.zeros(24)
        self.params['gamma_bn3_conv'] = np.ones(48)  # 新增Conv3后的BN
        self.params['beta_bn3_conv'] = np.zeros(48)
        self.params['gamma_bn3'] = np.ones(256)  # Affine1后的BN
        self.params['beta_bn3'] = np.zeros(256)

        # 网络结构
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=1)
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma_bn1'], self.params['beta_bn1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride=1, pad=1)
        self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma_bn2'], self.params['beta_bn2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 新增第三层卷积
        self.layers['Conv3'] = Convolution(self.params['W3_conv'], self.params['b3_conv'], stride=1, pad=1)
        self.layers['BatchNorm3_conv'] = BatchNormalization(self.params['gamma_bn3_conv'], self.params['beta_bn3_conv'])
        self.layers['Relu3'] = Relu()

        # 根据是否使用全局平均池化选择池化方式
        if self.use_global_pooling:
            self.layers['GlobalPool'] = GlobalAveragePooling()  # 新增全局平均池化层
        else:
            self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma_bn3'], self.params['beta_bn3'])
        self.layers['Relu4'] = Relu()  # 重命名为Relu4避免重复

        if self.use_dropout:
            self.layers['Dropout1'] = Dropout(self.dropout_ratio)

        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])



        self.last_layer = SoftmaxWithLoss()

    # 其他方法保持不变...
    def predict(self, x, train_flg=False):
        """预测方法，添加train_flg参数"""
        for key, layer in self.layers.items():
            # 对于Dropout和BatchNorm层，需要传递train_flg
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 收集梯度 - 更新梯度收集
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3_conv'] = self.layers['Conv3'].dW  # 新增
        grads['b3_conv'] = self.layers['Conv3'].db  # 新增
        grads['W3'] = self.layers['Affine1'].dW
        grads['b3'] = self.layers['Affine1'].db
        grads['W4'] = self.layers['Affine2'].dW
        grads['b4'] = self.layers['Affine2'].db

        # 添加所有BatchNorm层的梯度
        grads['gamma_bn1'] = self.layers['BatchNorm1'].dgamma
        grads['beta_bn1'] = self.layers['BatchNorm1'].dbeta
        grads['gamma_bn2'] = self.layers['BatchNorm2'].dgamma
        grads['beta_bn2'] = self.layers['BatchNorm2'].dbeta
        grads['gamma_bn3_conv'] = self.layers['BatchNorm3_conv'].dgamma  # 新增
        grads['beta_bn3_conv'] = self.layers['BatchNorm3_conv'].dbeta  # 新增
        grads['gamma_bn3'] = self.layers['BatchNorm3'].dgamma
        grads['beta_bn3'] = self.layers['BatchNorm3'].dbeta

        return grads


class BatchNormalization:
    """Batch Normalization层 - 支持4D卷积数据"""

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma  # 缩放参数
        self.beta = beta  # 平移参数
        self.momentum = momentum
        self.input_shape = None

        # 运行时均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # 反向传播中间变量
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

        # 新增：用于处理4D数据
        self.reshape_from_4d = False
        self.original_4d_shape = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape

        # 处理4D卷积数据
        if x.ndim == 4:
            self.reshape_from_4d = True
            self.original_4d_shape = x.shape
            N, C, H, W = x.shape
            # 重塑为 (N, H, W, C) 然后 reshape 为 (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            self.reshape_from_4d = False

        out = self.__forward(x, train_flg)

        # 恢复原始形状
        if self.reshape_from_4d:
            N, C, H, W = self.original_4d_shape
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            out = out.reshape(*self.input_shape)

        return out

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            D = x.shape[1]  # 特征维度
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)  # 按特征维度计算均值
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)  # 按特征维度计算方差
            std = np.sqrt(var + 1e-7)  # 标准差
            xn = xc / std  # 标准化

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 1e-7))

        # 现在 gamma 和 beta 的形状与 xn 的特征维度匹配
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        # 处理4D卷积数据的梯度
        if self.reshape_from_4d:
            N, C, H, W = self.original_4d_shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            dout = dout.reshape(self.xn.shape)

        dx = self.__backward(dout)

        # 恢复原始形状
        if self.reshape_from_4d:
            N, C, H, W = self.original_4d_shape
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            dx = dx.reshape(*self.input_shape)

        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class GlobalAveragePooling:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        out = x.mean(axis=(2, 3))
        return out

    def backward(self, dout):
        N, C = dout.shape
        H, W = self.input_shape[2], self.input_shape[3]
        dx = dout.reshape(N, C, 1, 1) / (H * W)
        dx = np.broadcast_to(dx, self.input_shape)
        return dx.copy()  # 确保返回可写副本