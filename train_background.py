import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from util import calculate_ndvi, calculate_evi, calculate_savi, calculate_mndwi, calculate_ndmi, calculate_gci
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from Setting import Config
from Data_csv import data_csv
from train_ML import train_ML
from train_DL import train_DL
from predict_DL import predict_DL
from fig_radar import fig_radar
from fig_plot import fig_plot
from model_compare import *
from MFSET import ImprovedFixedAdaptiveTransformerV3 as MFSET

config = Config()

models_ML = {
    # "RF": RandomForestClassifier(),
    # "XGB": XGBClassifier(),
    # "SVM": SVC(probability=True),
    # "LR": LogisticRegression(max_iter=1000),
    # "Ridge": RidgeClassifier(),
    # "GBR": GradientBoostingClassifier(n_estimators=100),
    # "AdaBoost": AdaBoostClassifier(n_estimators=100),
    # "ExtraTrees": ExtraTreesClassifier(n_estimators=100),
    # "DT": DecisionTreeClassifier()
}

models_DL = {
    "MFSET": MFSET(input_dim=config._dim, num_classes=config._num_classes),

    "MLP": MLP(input_dim=config._dim, num_classes=config._num_classes),
    "LSTM": LSTM(input_dim=config._dim, num_classes=config._num_classes),
    "Transformer": Transformer(input_dim=config._dim, num_classes=config._num_classes),
    "GRU": GRU(input_dim=config._dim, num_classes=config._num_classes),
    "RNN": RNN(input_dim=config._dim, num_classes=config._num_classes),
    "CNN": CNN(input_dim=config._dim, num_classes=config._num_classes),
    "CNN-LSTM": CNNLSTM(input_dim=config._dim, num_classes=config._num_classes),
    "CNN-GRU": CNNGRU(input_dim=config._dim, num_classes=config._num_classes),
}


def co_data(config):
    df_all = pd.DataFrame()

    for area in config._obj_list:
        mid_df = pd.read_csv(f"{config._path_train_data}/{area}.csv")
        positive_samples = mid_df[mid_df["Target"] == 1]
        negative_samples = mid_df[mid_df["Target"] == 0]
        n_positive = min(len(positive_samples), 20000)
        n_negative = min(len(negative_samples), 20000)
        positive_samples = positive_samples.sample(n=n_positive, random_state=42, replace=False)
        negative_samples = negative_samples.sample(n=n_negative, random_state=42, replace=False)
        area_samples = pd.concat([positive_samples, negative_samples])
        df_all = pd.concat([df_all, area_samples], ignore_index=True)

    return df_all

def Train_test_split(config):
    csv_path = f"{config._path_train_data}/all_data.csv"
    df = pd.read_csv(csv_path)

    df["NDVI"] = calculate_ndvi(df)
    df["EVI"] = calculate_evi(df)
    df["SAVI"] = calculate_savi(df)
    df["MNDWI"] = calculate_mndwi(df)
    df["NDMI"] = calculate_ndmi(df)
    # df["GCI"] = calculate_gci(df)

    features = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "NDVI", "EVI", "SAVI",
                "MNDWI", "NDMI"]

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    train_size = 1 - config._data_split
    df_train, df_val = train_test_split(
        df,
        test_size=config._data_split,
        stratify=df["Target"],
        random_state=config._random_seed
    )

    X_train = df_train[features]
    y_train = df_train["Target"]
    X_val = df_val[features]
    y_val = df_val["Target"]

    df_train.to_csv(f"{config._path_train_data}/all_train.csv", index=False)
    df_val.to_csv(f"{config._path_train_data}/all_val.csv", index=False)

    config.x_train = X_train
    config.x_val = X_val
    config.y_train = y_train
    config.y_val = y_val
    config.scaler = scaler

    return config


def train_DL(config, name, model):
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
    optimizer = optim.AdamW(model.parameters(), lr=config._lr_all, weight_decay=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_f1 = 0
    logs_memory = []

    for epoch in range(config._epochs_all):
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
            print(f"Epoch [{epoch + 1}/{config._epochs_all}] "
                  f"Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f} "
                  f"Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            print(f"new weight! f1={val_f1}")
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"{config._path_weights}/all_{name}.pth")

    mid_result = {
            "Model": 1,
            "Train Accuracy": 2,
            "Train Precision": 3,
            "Train Recall": 4,
            "Train F1": 5,
            "Train Kappa": 6,
            "Val Accuracy": 7,
            "Val Precision": 8,
            "Val Recall": 9,
            "Val F1": 10,
            "Val Kappa": 11,
        }
    df_result = pd.DataFrame(logs_memory, columns=mid_result.keys())
    df_result.to_csv(f"{config._path_logs}/all_{name}_logs.csv", index=False)

    # model.load_state_dict(best_model_state)
    #
    # model.eval()
    # with torch.no_grad():
    #     y_train_pred = model(X_train.to(device)).argmax(dim=1).cpu().numpy()
    #     y_val_pred = model(X_val.to(device)).argmax(dim=1).cpu().numpy()
    #
    # train_acc = accuracy_score(y_train, y_train_pred)
    # train_prec = precision_score(y_train, y_train_pred, average="weighted", zero_division=0)
    # train_rec = recall_score(y_train, y_train_pred, average="weighted", zero_division=0)
    # train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
    # train_kappa = cohen_kappa_score(y_train, y_train_pred)
    #
    # val_acc = accuracy_score(y_val, y_val_pred)
    # val_prec = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    # val_rec = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
    # val_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    # val_kappa = cohen_kappa_score(y_val, y_val_pred)
    #
    # mid_result = {
    #     "Model": name,
    #     "Train Accuracy": train_acc,
    #     "Train Precision": train_prec,
    #     "Train Recall": train_rec,
    #     "Train F1": train_f1,
    #     "Train Kappa": train_kappa,
    #     "Val Accuracy": val_acc,
    #     "Val Precision": val_prec,
    #     "Val Recall": val_rec,
    #     "Val F1": val_f1,
    #     "Val Kappa": val_kappa,
    # }


config = Config()
df_all = co_data(config)
df_all.to_csv(f"{config._path_train_data}/all_data.csv", index=False)

Train_test_split(config)
for name, model in models_DL.items():
    print(f"---------------- {name} ----------------")
    mid_result = train_DL(config, name, model)




