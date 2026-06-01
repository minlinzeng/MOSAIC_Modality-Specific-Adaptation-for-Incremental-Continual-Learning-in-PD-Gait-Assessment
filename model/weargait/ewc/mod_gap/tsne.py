import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# ==========================================
# Step 1: unbiased random projector
# ==========================================
class RandomProjector:
    def __init__(self, target_dim=1024, seed=42):
        self.target_dim = target_dim
        self.seed = seed
        self.W_dict = {} # Frozen projection per input dim

    def project(self, X):
        """
        X: shape (N_samples, Original_Dim)
        """
        N, orig_dim = X.shape
        # Create frozen random matrix on first use
        if orig_dim not in self.W_dict:
            np.random.seed(self.seed)
            # Gaussian init for approximate orthogonality
            W = np.random.randn(orig_dim, self.target_dim) / np.sqrt(self.target_dim)
            self.W_dict[orig_dim] = W
            
        # Linear projection only (no trainable weights)
        return np.dot(X, self.W_dict[orig_dim])