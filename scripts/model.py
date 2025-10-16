#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------
# PowerCo SME Churn Analysis Project - DEPLOYMENT VERSION
# ------------------------------------------------

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import joblib

warnings.filterwarnings("ignore")


def main():
    # ------------------------------
    # Load and preprocess data
    # ------------------------------
    clientdf = pd.read_csv('client_data.csv')
    pricedf = pd.read_csv('price_data.csv')

    clientdf['id'] = clientdf['id'].str[:7]
    pricedf['id'] = pricedf['id'].str[:7]
    clientdf["date_activ"] = pd.to_datetime(clientdf["date_activ"])
    clientdf["date_end"] = pd.to_datetime(clientdf["date_end"])
    clientdf['tenure'] = clientdf['date_end'].dt.year - clientdf['date_activ'].dt.year
    clientdf['churn'] = clientdf['churn'].replace({0: 'retention', 1: 'churn'})

    # Channel mapping
    clientdf['channel_sales'] = clientdf['channel_sales'].replace({
        'foosdfpfkusacimwkcsosbicdxkicaua': 'channel_sales_1',
        'MISSING': 'not_specified',
        'lmkebamcaaclubfxadlmueccxoimlema': 'channel_sales_2',
        'usilxuppasemubllopkaafesmlibmsdf': 'channel_sales_3',
        'ewpakwlliwisiwduibdlfmalxowmwpci': 'channel_sales_4',
        'sddiedcslfslkckwlfkdpoeeailfpeds': 'channel_sales_5',
        'epumfxlbckeskwekxbiuasklxalciiuu': 'channel_sales_6',
        'fixdbufsefwooaasfcxdxadsiekoceaa': 'channel_sales_7'
    })
    clientdf['has_gas'] = clientdf['has_gas'].replace({'f': 'false', 't': 'true'})
    clientdf['origin_up'] = clientdf['origin_up'].fillna('not_specified')

    # Aggregate price data
    pricedf = pricedf.groupby('id', as_index=False)[[
        'price_off_peak_var', 'price_peak_var', 'price_mid_peak_var',
        'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix'
    ]].mean()

    # Merge
    df = clientdf.merge(pricedf, on="id", how="inner")

    # Drop unused columns
    df = df.drop(columns=['id', 'date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], errors='ignore')

    # Encode categoricals
    le = LabelEncoder()
    for col in ['channel_sales', 'has_gas', 'origin_up']:
        df[col] = le.fit_transform(df[col].astype(str))

    # Feature selection
    features_to_keep = [
        'channel_sales', 'cons_12m', 'cons_gas_12m', 'forecast_cons_12m',
        'forecast_discount_energy', 'forecast_meter_rent_12m', 'has_gas',
        'imp_cons', 'margin_net_pow_ele', 'net_margin', 'num_years_antig',
        'origin_up', 'pow_max', 'price_mid_peak_fix'
    ]
    available_features = [f for f in features_to_keep if f in df.columns]

    X = df[available_features]
    y = df["churn"].map({'churn': 1, 'retention': 0})

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    rf = RandomForestClassifier(
        n_estimators=1000, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_best = (y_proba >= best_threshold).astype(int)

    # ------------------------------
    # Output Results
    # ------------------------------
    print("\n=== POWERCO CHURN MODEL DEPLOYMENT RESULTS ===")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Average Precision (AUC-PR): {average_precision_score(y_test, y_proba):.3f}\n")
    print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred_best, digits=3))
    print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred_best))
    print(f"\nChurn Detection Rate (Recall): {recall_score(y_test, y_pred_best):.1%}")
    print(f"Model Precision: {precision_score(y_test, y_pred_best):.1%}")
    print(f"F1 Score: {f1_score(y_test, y_pred_best):.3f}")

    # Save model + scaler for deployment
    joblib.dump(rf, "powerco_churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nModel and scaler saved successfully for deployment.")


if __name__ == "__main__":
    main()
