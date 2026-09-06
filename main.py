import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import product
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, confusion_matrix

# Фиксируем random seed для воспроизводимости
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cpu")
print("Using device:", DEVICE)


# -----------------------------
# 1. Генерация синтетического датасета (3 спирали, 3 класса)
# -----------------------------
def generate_spiral_dataset(N=900, K=3):
    """
    Генерирует двумерный датасет из K спиралей (K классов), всего N объектов.
    """
    D = 2
    X = np.zeros((N, D), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    points_per_class = N // K

    for j in range(K):
        ix = range(points_per_class * j, points_per_class * (j + 1))
        r = np.linspace(0.0, 1, points_per_class)  # радиус
        t = np.linspace(j * np.pi / 2, j * np.pi / 2 + np.pi, points_per_class)  # угол
        t += np.random.randn(points_per_class) * 0.15
        X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
        y[ix] = j

    return torch.from_numpy(X), torch.from_numpy(y)


# -----------------------------
# 2. Определение модели MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, activation="relu", dropout_p=0.0):
        super().__init__()
        layers = []
        in_features = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 3. Обучение и оценка модели
# -----------------------------
def train_one_model(X_train, y_train, X_val, y_val,
                    hidden_layers, activation, epochs=200,
                    lr=1e-3, batch_size=64, dropout_p=0.0):
    """
    Обучает одну конфигурацию MLP и возвращает историю потерь и точность.
    """
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    model = MLP(input_dim, hidden_layers, num_classes,
                activation=activation, dropout_p=dropout_p).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader'ы
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        t0 = timer()
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Валидация
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)

                preds = torch.argmax(logits, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        t1 = timer()
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | "
                  f"Val acc: {val_acc:.4f} | "
                  f"Time: {t1 - t0:.2f}s")

    return model, history


# -----------------------------
# 4. Визуализация границы решений
# -----------------------------
def plot_decision_boundary(model, X, y, title="Decision boundary", fname="decision_boundary.png"):
    model.eval()
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.from_numpy(grid).to(DEVICE)

    with torch.no_grad():
        logits = model(grid_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.tab10)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, s=20, cmap=plt.cm.tab10, edgecolor="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)


def plot_loss_curves(history, title="Loss curves", fname="loss_curves.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)


# -----------------------------
# 5. Основной эксперимент: перебор конфигураций
# -----------------------------
def main():
    # 5.1. Генерируем данные
    X, y = generate_spiral_dataset(N=900, K=3)
    X = X.to(DEVICE)
    y = y.to(DEVICE)

    # Разбиение на train/val/test: 70/15/15
    N = X.shape[0]
    indices = torch.randperm(N)
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Train size:", X_train.shape[0])
    print("Val size:", X_val.shape[0])
    print("Test size:", X_test.shape[0])

    # 5.2. Конфигурации гиперпараметров
    hidden_layer_options = [
        [32],
        [64],
        [64, 32],
        [128, 64],
        [128, 64, 32]
    ]
    activations = ["relu", "tanh"]
    epochs_list = [100, 200]
    dropout_options = [0.0, 0.2]

    configs = list(product(hidden_layer_options, activations, epochs_list, dropout_options))

    results = []
    best_val_acc = 0.0
    best_model = None
    best_history = None
    best_config = None

    # 5.3. Перебор конфигураций
    for i, (hidden_layers, act, epochs, dropout_p) in enumerate(configs, start=1):
        print(f"\n=== Config {i}/{len(configs)} ===")
        print(f"Hidden layers: {hidden_layers}, activation: {act}, "
              f"epochs: {epochs}, dropout: {dropout_p}")

        model, history = train_one_model(
            X_train, y_train, X_val, y_val,
            hidden_layers=hidden_layers,
            activation=act,
            epochs=epochs,
            lr=1e-3,
            batch_size=64,
            dropout_p=dropout_p
        )

        # Оценка на валидации
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val)
            preds_val = torch.argmax(logits_val, dim=1)
            val_acc = accuracy_score(y_val.cpu().numpy(), preds_val.cpu().numpy())

        # Оценка на тесте
        with torch.no_grad():
            logits_test = model(X_test)
            preds_test = torch.argmax(logits_test, dim=1)
            test_acc = accuracy_score(y_test.cpu().numpy(), preds_test.cpu().numpy())

        results.append({
            "hidden_layers": str(hidden_layers),
            "activation": act,
            "epochs": epochs,
            "dropout": dropout_p,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1]
        })

        print(f"Config result -> Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_history = history
            best_config = (hidden_layers, act, epochs, dropout_p)

    # 5.4. Сохранение результатов в CSV
    df = pd.DataFrame(results)
    df.to_csv("mlp_classification_results.csv", index=False)
    print("\nSaved results to mlp_classification_results.csv")
    print(df.sort_values("val_acc", ascending=False).head())

    # 5.5. Графики и граница решений для лучшей модели
    if best_model is not None:
        hl, act, ep, dr = best_config
        title_loss = f"Loss curves (layers={hl}, act={act}, epochs={ep}, dropout={dr})"
        plot_loss_curves(best_history, title=title_loss,
                         fname="best_loss_curves.png")

        title_db = f"Decision boundary (layers={hl}, act={act})"
        plot_decision_boundary(best_model, X.cpu(), y.cpu(),
                               title=title_db,
                               fname="best_decision_boundary.png")

        # Матрица ошибок на тесте
        best_model.eval()
        with torch.no_grad():
            logits_test = best_model(X_test)
            preds_test = torch.argmax(logits_test, dim=1).cpu().numpy()
        cm = confusion_matrix(y_test.cpu().numpy(), preds_test)
        print("\nConfusion matrix (test):\n", cm)


if __name__ == "__main__":
    main()