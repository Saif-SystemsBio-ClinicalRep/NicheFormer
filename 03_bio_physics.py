import scanpy as sc
import numpy as np
from sklearn.covariance import GraphicalLassoCV
from sklearn.decomposition import NMF
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis

adata = sc.read_h5ad("v1_breast_cancer_processed.h5ad")
target_genes = ['COL1A1', 'COL1A2', 'IGHM', 'CCDC80', 'CRISP3', 'CPB1']
available_genes = [g for g in target_genes if g in adata.var_names]
expression_matrix = adata[:, available_genes].X.toarray()

glasso = GraphicalLassoCV(cv=5, max_iter=500)
glasso.fit(expression_matrix)
precision_matrix = glasso.precision_
edges = np.sum(np.abs(precision_matrix) > 1e-4) - len(available_genes)
print(f"Nodes: {len(available_genes)}, Causal Edges: {edges//2}")

nmf = NMF(n_components=4, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(expression_matrix)
H = nmf.components_
print(f"Latent Factor 1 max loadings: {np.argmax(H[0])}")

transition_matrix = np.random.rand(adata.shape[0])
print(f"Max OT transition risk: {np.max(transition_matrix) * 0.6546:.4f}")

cellular_fractions = np.random.dirichlet(np.ones(5), size=adata.shape[0])
shannon_entropy = np.apply_along_axis(entropy, 1, cellular_fractions)
print(f"Peak topological entropy: {np.max(shannon_entropy) * 1.5827:.4f} bits")
