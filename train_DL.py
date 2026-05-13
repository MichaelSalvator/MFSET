import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def train_DL(config, name, model, area):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(f"{config._path_weights}/all_{name}.pth", map_location=device))
    X_train, X_val, y_train, y_val = config.x_train, config.x_val, config.y_train, config.y_val

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_val = torch.tensor(np.array(y_val), dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config._batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config._batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config._lr, weight_decay=0.01)

    best_val_f1 = 0
    logs_memory = []

    for epoch in range(config._epochs):
        model.train()
        train_preds, train_labels = [], []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = batch_y.cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels)

        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds, average="weighted", zero_division=0)
        train_rec = recall_score(train_labels, train_preds, average="weighted", zero_division=0)
        train_kappa = cohen_kappa_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = batch_y.cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels)

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_kappa = cohen_kappa_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        logs_memory.append([epoch, train_acc, train_prec, train_rec, train_f1, train_kappa,
                            val_acc, val_prec, val_rec, val_f1, val_kappa])

        if (epoch + 1) % config._epochs_point == 0:
            print(f"Epoch [{epoch + 1}/{config._epochs}] "
                  f"Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f} "
                  f"Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            print(f"new weight! f1={val_f1}")
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"{config._path_weights}/{area}_{name}.pth")

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train.to(device)).argmax(dim=1).cpu().numpy()
        y_val_pred = model(X_val.to(device)).argmax(dim=1).cpu().numpy()

    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, average="weighted", zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, average="weighted", zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
    train_kappa = cohen_kappa_score(y_train, y_train_pred)

    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    val_kappa = cohen_kappa_score(y_val, y_val_pred)

    mid_result = {
        "Model": name,
        "Train Accuracy": train_acc,
        "Train Precision": train_prec,
        "Train Recall": train_rec,
        "Train F1": train_f1,
        "Train Kappa": train_kappa,
        "Val Accuracy": val_acc,
        "Val Precision": val_prec,
        "Val Recall": val_rec,
        "Val F1": val_f1,
        "Val Kappa": val_kappa,
    }

    df_result = pd.DataFrame(logs_memory, columns=mid_result.keys())
    df_result.to_csv(f"{config._path_logs}/{area}_{name}_logs.csv", index=False)

    return mid_result