import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# একটি পরিবর্তিত স্টার গ্রাফ তৈরি করার ফাংশন
def create_modified_star_graph():
    # প্রতিটি নোডের জন্য 1-মাত্রিক ফিচার (5টি নোড)
    x = torch.tensor([[1], [0], [0], [0], [0]], dtype=torch.float)

    # এজ লিস্ট (COO ফর্ম্যাটে: source -> target)
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 3, 4],
                               [1, 2, 3, 4, 0, 2, 1, 1, 0]], dtype=torch.long)

    # নোড লেবেল (0 = outer node, 1 = center node)
    y = torch.tensor([1, 0, 0, 0, 0], dtype=torch.long)

    # কোন নোড train/test হবে সেটির জন্য mask
    train_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)  # Node 0 এবং 1 train-এ
    test_mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)   # Node 2, 3, 4 test-এ

    # PyTorch Geometric এর Data অবজেক্টে সব কিছু যুক্ত করা
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data

# গ্রাফটি ভিজুয়ালাইজ করার ফাংশন
def visualize_modified_star_graph(data):
    G = nx.Graph()
    edges = data.edge_index.t().tolist()  # edge_index কে list of tuples-এ রূপান্তর
    G.add_edges_from(edges)

    # গ্রাফ আঁকা
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # সুন্দর করে সাজানোর জন্য spring layout
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_size=10)
    plt.title("Modified Star Graph Visualization")
    plt.show()

# GCN মডেলের ক্লাস ডিফাইন
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # প্রথম লেয়ার
        self.conv2 = GCNConv(hidden_dim, output_dim)  # দ্বিতীয় লেয়ার

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # প্রথম GCN কনভোলিউশন
        x = F.relu(x)  # রিলু অ্যাক্টিভেশন
        x = self.conv2(x, edge_index)  # দ্বিতীয় কনভোলিউশন
        return F.log_softmax(x, dim=1)  # সফটম্যাক্স রিটার্ন (ক্লাস প্রোবাবিলিটি)

# গ্রাফ তৈরি ও ভিজুয়ালাইজ
data = create_modified_star_graph()
visualize_modified_star_graph(data)

# মডেল ও অপটিমাইজার ইনিশিয়ালাইজ
model = GCN(input_dim=1, hidden_dim=4, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()  # ক্লাসিফিকেশন লস

# ট্রেইন ফাংশন
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)  # ফোরওয়ার্ড পাস
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # train_mask অনুযায়ী loss হিসাব
    loss.backward()  # ব্যাকওয়ার্ড পাস
    optimizer.step()  # ওজন আপডেট
    return loss.item()

# টেস্ট ফাংশন
def test():
    model.eval()
    out = model(data)
    pred = out[data.test_mask].max(1)[1]  # সবচেয়ে সম্ভাব্য ক্লাস নিন
    acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()  # একিউরেসি হিসাব
    return acc

# মডেল ট্রেইনিং লুপ
for epoch in range(50):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

