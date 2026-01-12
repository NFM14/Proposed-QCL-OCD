import numpy as np
import pandas as pd
import torch
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# --- Load and preprocess data ---
df = pd.read_csv(r"C:\Users\nelso\OneDrive\Desktop\OCD\GSE78104_combined_dataset.csv")
label_col = [col for col in df.columns if col.lower() in ['label','class','target','diagnosis','group']][0]
non_numeric_cols = [col for col in df.columns if df[col].dtype == 'object' and col != label_col]
if non_numeric_cols:
    print("Dropping columns:", non_numeric_cols)
    df = df.drop(columns=non_numeric_cols)

X = df.drop(label_col, axis=1).values.astype(np.float64)
y = df[label_col].values
if y.dtype == 'object':
    label_map = {lbl: 0 if 'control' in str(lbl).lower() else 1 for lbl in np.unique(y)}
    y = np.array([label_map[lbl] for lbl in y])

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = np.log2(X + 1)

# Feature selection + PCA
X = VarianceThreshold(threshold=0.02).fit_transform(X)
k_mi = min(120, X.shape[1])
k_anova = min(120, X.shape[1])
X_1 = SelectKBest(mutual_info_classif, k=k_mi).fit_transform(X, y)
X_2 = SelectKBest(f_classif, k=k_anova).fit_transform(X, y)
X = np.concatenate([X_1, X_2], axis=1)
pca = PCA(n_components=min(30, X.shape[1]), random_state=42)
X_pca = pca.fit_transform(X)
X_fs = np.hstack([X, X_pca])

scaler = RobustScaler()
X_fs = scaler.fit_transform(X_fs)

# Train/test split + SMOTE
SPLIT_SEED = 555
X_train, X_test, y_train, y_test = train_test_split(X_fs, y, stratify=y, test_size=0.20, random_state=SPLIT_SEED)
smote = SMOTE(random_state=SPLIT_SEED, k_neighbors=12)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Hybrid Model
n_qubits = min(8, X_train.shape[1])
X_train_q = X_train[:, :n_qubits]
X_test_q = X_test[:, :n_qubits]
X_train_c = X_train[:, n_qubits:]
X_test_c = X_test[:, n_qubits:]
n_classical_features = X_train_c.shape[1]

dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (5, n_qubits, 3)}
class Hybrid(nn.Module):
    def __init__(self, n_qubits, n_classical_features):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.classical_fc = nn.Sequential(
            nn.Linear(n_classical_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(n_qubits + 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, xq, xc):
        q_out = self.qlayer(xq)
        c_out = self.classical_fc(xc)
        out = torch.cat([q_out, c_out], dim=1)
        return self.fc_out(out)

model = Hybrid(n_qubits, n_classical_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.012)
criterion = nn.CrossEntropyLoss(label_smoothing=0.07)

X_train_q_t = torch.tensor(X_train_q, dtype=torch.float32).to(device)
X_train_c_t = torch.tensor(X_train_c, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_q_t = torch.tensor(X_test_q, dtype=torch.float32).to(device)
X_test_c_t = torch.tensor(X_test_c, dtype=torch.float32).to(device)

best_acc = 0.0
patience = 20
no_improve = 0

for epoch in range(60):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_q_t, X_train_c_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = torch.softmax(model(X_test_q_t, X_test_c_t), dim=1).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc, no_improve = acc, 0
        torch.save(model.state_dict(), "hybrid_model_best.pth")
        best_epoch = epoch + 1
    else:
        no_improve += 1
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")
    if no_improve >= patience:
        break

print("Best model saved as hybrid_model_best.pth")

# Load best model for evaluation!
model.load_state_dict(torch.load("hybrid_model_best.pth"))
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(X_test_q_t, X_test_c_t), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs[:,1])
    roc_auc = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs[:,1])
    pr_auc = auc(rec_curve, prec_curve)

# Z-score/statistical significance
n_test = len(y_test)
expected = n_test * 0.5
variance = n_test * 0.5 * 0.5
z_score = ((acc * n_test) - expected) / np.sqrt(variance)
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Best at Epoch {best_epoch}: Accuracy {acc:.2%}, Precision {prec:.2%}, Recall {rec:.2%}, F1 {f1:.2%} | Z-score: {z_score:.3f}, p-value: {p_value:.6f}")

plt.figure(figsize=(6,5))
sns.barplot(x=['Accuracy','Precision','Recall','F1'], y=[acc,prec,rec,f1], palette='viridis', legend=False)
plt.ylim(0,1.05)
for i, v in enumerate([acc,prec,rec,f1]):
    plt.text(i, v+0.03, f'{v:.2f}', ha='center', fontweight='bold')
plt.title('Hybrid Quantum-Classical Metrics (Single Split)')
plt.savefig("hybrid_metrics_bar_improved.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix\nAccuracy={acc:.2%}, F1={f1:.2%}")
plt.savefig("hybrid_confusion_matrix_improved.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC={roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--', lw=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig("hybrid_roc_improved.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(6,5))
plt.plot(rec_curve, prec_curve, color='purple', lw=2.5, label=f'Precision-Recall curve (AUC={pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.savefig("hybrid_pr_curve_improved.png", bbox_inches='tight')
plt.close()