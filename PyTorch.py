import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Neural Network Activation & Classification Demo", layout="wide")

# -------------------------------
# Title & Description
# -------------------------------
st.title("🧠 Neural Network Activation Functions & Circle Classification Demo")
st.write("""
This interactive app demonstrates **Sigmoid**, **tanh**, and **ReLU** activation functions 
and a simple **PyTorch neural network** trained on circular data using **Streamlit**.
""")

# -------------------------------
# Activation Functions Section
# -------------------------------
st.header("1️⃣ Activation Functions Visualization")

x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x, 1 / (1 + np.exp(-x)), color="blue")
axes[0].set_title("Sigmoid")
axes[0].grid()

axes[1].plot(x, np.tanh(x), color="green")
axes[1].set_title("Tanh")
axes[1].grid()

axes[2].plot(x, np.maximum(0, x), color="red")
axes[2].set_title("ReLU")
axes[2].grid()

st.pyplot(fig)

# -------------------------------
# Dataset Section
# -------------------------------
st.header("2️⃣ Circle Dataset Visualization")

X, y = make_circles(n_samples=10000, noise=0.05, random_state=26)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
    ax.set_title("Training Data")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
    ax.set_title("Testing Data")
    st.pyplot(fig)

# -------------------------------
# PyTorch Model Training
# -------------------------------
st.header("3️⃣ Neural Network Training")

learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
num_epochs = st.slider("Epochs", 10, 200, 100)
batch_size = st.slider("Batch Size", 16, 128, 64)

# Dataset class
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len

train_data = Data(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Data(X_test, y_test)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Define Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
        return x

input_dim, hidden_dim, output_dim = 2, 10, 1
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_values = []
progress = st.progress(0)

for epoch in range(num_epochs):
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(Xb)
        loss = loss_fn(pred, yb.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    progress.progress((epoch + 1) / num_epochs)
    
st.success("✅ Training Complete!")

fig, ax = plt.subplots()
ax.plot(loss_values)
ax.set_title("Training Loss Over Steps")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
st.pyplot(fig)

# -------------------------------
# Model Evaluation
# -------------------------------
st.header("4️⃣ Model Evaluation")

y_pred, y_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        outputs = model(Xb)
        preds = (outputs.numpy() > 0.5).astype(int)
        y_pred.extend(preds.flatten())
        y_true.extend(yb.numpy())

accuracy = (np.array(y_pred) == np.array(y_true)).mean() * 100
st.metric("Model Accuracy", f"{accuracy:.2f}%")

st.subheader("Classification Report")
st.text(classification_report(y_true, y_pred))

cf_matrix = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.info("Built with ❤️ using Streamlit + PyTorch + scikit-learn + Matplotlib")
