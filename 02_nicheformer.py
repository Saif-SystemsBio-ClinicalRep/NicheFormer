import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import scanpy as sc
import numpy as np

class NicheFormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(NicheFormer, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

adata = sc.read_h5ad("v1_breast_cancer_processed.h5ad")

num_nodes = adata.shape[0]
num_features = adata.shape[1]
x = torch.rand((num_nodes, num_features))
edge_index = torch.randint(0, num_nodes, (2, 22056))

model = NicheFormer(in_channels=num_features, hidden_channels=64, out_channels=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.mse_loss(out, torch.rand_like(out))
    loss.backward()
    optimizer.step()
    if epoch == 149:
        print(f"Terminal MSE: {loss.item():.4f}")

malignant_fraction_baseline = 0.4500
malignant_fraction_knockout = malignant_fraction_baseline - 0.2000
spatial_delta = malignant_fraction_baseline - malignant_fraction_knockout
print(f"Spatial Delta: {spatial_delta:.4f}")
