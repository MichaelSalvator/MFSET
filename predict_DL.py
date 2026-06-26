import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import cv2
import os
from util import calculate_ndvi, calculate_evi, calculate_savi, calculate_mndwi, calculate_ndmi, calculate_gci


# def predict_DL(config, name, model, area, X_test_img):
#     bands, original_h, original_w = X_test_img.shape
#
#     df = pd.DataFrame(X_test_img.reshape(bands, original_h * original_w).T,
#                       columns=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"])
#
#     df["NDVI"] = calculate_ndvi(df)
#     df["EVI"] = calculate_evi(df)
#     df["SAVI"] = calculate_savi(df)
#     df["MNDWI"] = calculate_mndwi(df)
#     df["NDMI"] = calculate_ndmi(df)
#     df["GCI"] = calculate_gci(df)
#
#     features = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "NDVI", "EVI", "SAVI",
#                 "MNDWI", "NDMI", "GCI"]
#
#     df[features] = df[features].replace([np.inf, -np.inf], np.nan)
#     df[features] = df[features].fillna(df[features].mean())
#
#     X_test_enhanced = config.scaler.transform(df[features])
#
#     model_path = f"{config._path_weights}/{area}_{name}.pth"
#     model.load_state_dict(torch.load(model_path))
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#
#     X_test_tensor = torch.tensor(X_test_enhanced, dtype=torch.float32).to(device)
#
#     test_dataset = TensorDataset(X_test_tensor)
#     test_loader = DataLoader(test_dataset, batch_size=config._batch_size, shuffle=False)
#
#     predictions = []
#     with torch.no_grad():
#         for batch_X in test_loader:
#             batch_X = batch_X[0].to(device)
#             outputs = model(batch_X)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#             predictions.extend(preds)
#
#     predicted_image = np.array(predictions).reshape(original_h, original_w)
#
#     colored_image = np.zeros((original_h, original_w, 3), dtype=np.uint8)
#     for label, color in config._predict_colors.items():
#         colored_image[predicted_image == label] = color
#     print(f"{config._path_predict}/{area}_predict_{name}.png！")
#     cv2.imwrite(f"{config._path_predict}/{area}_predict_{name}.png", colored_image)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import cv2
import os
from util import calculate_ndvi, calculate_evi, calculate_savi, calculate_mndwi, calculate_ndmi, calculate_gci

def predict_DL(config, name, model, area, X_test_img):
    bands, original_h, original_w = X_test_img.shape

    df = pd.DataFrame(X_test_img.reshape(bands, original_h * original_w).T,
                      columns=['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'])

    df['NDVI'] = calculate_ndvi(df)
    df['EVI'] = calculate_evi(df)
    df['SAVI'] = calculate_savi(df)
    df['MNDWI'] = calculate_mndwi(df)
    df['NDMI'] = calculate_ndmi(df)
    # df['GCI'] = calculate_gci(df)

    features = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12',
                'NDVI', 'EVI', 'SAVI', 'MNDWI', 'NDMI']

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].fillna(df[features].mean())

    if hasattr(config, "scaler") and config.scaler is not None:
        X_test_enhanced = config.scaler.transform(df[features])
    else:
        X_test_enhanced = df[features].values

    model.load_state_dict(torch.load(f"{config._path_weights}/{area}_{name}.pth", weights_only=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test_enhanced, dtype=torch.float32).to(device)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config._batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X[0].to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    predicted_image = np.array(predictions).reshape(original_h, original_w)


    mangrove_label = 1
    mangrove_mask = (predicted_image == mangrove_label).astype(np.uint8)
    num_features, labeled_mangrove = cv2.connectedComponents(mangrove_mask)


    water_mask = cv2.imread(f"{config._path_water_mask}/{area}_water_mask.png", cv2.IMREAD_GRAYSCALE)
    water_mask = (water_mask > 127).astype(np.uint8)
    dilated_water = cv2.dilate(water_mask, np.ones((3, 3), np.uint8), iterations=1)


    filtered_mask = np.zeros_like(mangrove_mask)
    for i in range(1, num_features):
        component_mask = (labeled_mangrove == i).astype(np.uint8)
        if np.any(component_mask & dilated_water):
            filtered_mask[component_mask == 1] = 1


    filtered_path = f"{config._path_predict}/{area}_predict_filtered_{name}.png"
    cv2.imwrite(filtered_path, filtered_mask * 255)
    print(f"{filtered_path}！")

    raw_path = f"{config._path_predict}/{area}_predict_{name}.png"
    cv2.imwrite(raw_path, predicted_image * 255)
    print(f"{raw_path}！")
