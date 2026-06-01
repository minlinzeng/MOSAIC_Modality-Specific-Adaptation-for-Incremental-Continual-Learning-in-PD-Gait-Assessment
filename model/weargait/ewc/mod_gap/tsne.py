import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# ==========================================
# Step 1: 定义无偏置的随机投影器
# ==========================================
class RandomProjector:
    def __init__(self, target_dim=1024, seed=42):
        self.target_dim = target_dim
        self.seed = seed
        self.W_dict = {} # 存储不同输入维度的冻结投影矩阵

    def project(self, X):
        """
        X: shape (N_samples, Original_Dim)
        """
        N, orig_dim = X.shape
        # 如果是第一次遇到这个维度，生成并冻结一个随机矩阵
        if orig_dim not in self.W_dict:
            np.random.seed(self.seed)
            # 采用标准高斯分布初始化，保证正交性和尺度不变性
            W = np.random.randn(orig_dim, self.target_dim) / np.sqrt(self.target_dim)
            self.W_dict[orig_dim] = W
            
        # 纯线性投影，不包含任何激活函数或学习权重
        return np.dot(X, self.W_dict[orig_dim])