import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class NicheFormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super(NicheFormer, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def extract_attention_weights(self, x, edge_index):
        # Extracts the raw attention matrices for Explainable AI (XAI)
        _, attention_weights_1 = self.conv1(x, edge_index, return_attention_weights=True)
        return attention_weights_1

if __name__ == "__main__":
    # Framework compilation test
    print("Initializing NicheFormer Spatial Architecture...")
    model = NicheFormer(in_channels=2000, hidden_channels=64, out_channels=1)
    
    # Synthetic spatial graph test (100 nodes, 300 edges) to verify compilation
    d_x = torch.randn(100, 2000)
    d_edge_index = torch.randint(0, 100, (2, 300))
    
    output = model(d_x, d_edge_index)
    print(f"Architecture compiled successfully. Output tensor shape: {output.shape}")
