"""
preprocessing.py — Dataset generation, cleaning, scaling, and splitting.

Data Dictionary:
  N           : Nitrogen content in soil (mg/kg)
  P           : Phosphorous content in soil (mg/kg)
  K           : Potassium content in soil (mg/kg)
  temperature : Average temperature (°C)
  humidity    : Relative humidity (%)
  ph          : Soil pH (0–14)
  rainfall    : Annual rainfall (mm)
  label       : Crop type (classification target)
  yield       : Crop yield kg/ha (regression target)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore')

from src.utils import FEATURE_COLS

# Per-crop parameter profiles: (mean, std) for each feature + base yield
CROP_PROFILES = {
    'rice':         {'N':(80,10),  'P':(48,5),   'K':(40,5),  'temp':(23,2), 'humidity':(82,5), 'ph':(6.0,0.5), 'rainfall':(200,30), 'yield':(4000,500)},
    'wheat':        {'N':(85,10),  'P':(50,5),   'K':(48,5),  'temp':(20,3), 'humidity':(50,5), 'ph':(6.5,0.5), 'rainfall':(75,15),  'yield':(3500,400)},
    'maize':        {'N':(78,10),  'P':(48,5),   'K':(20,5),  'temp':(22,2), 'humidity':(65,5), 'ph':(6.0,0.5), 'rainfall':(85,20),  'yield':(5000,600)},
    'chickpea':     {'N':(40,5),   'P':(68,5),   'K':(80,10), 'temp':(18,3), 'humidity':(18,3), 'ph':(7.5,0.3), 'rainfall':(73,15),  'yield':(1200,200)},
    'kidneybeans':  {'N':(20,5),   'P':(68,5),   'K':(20,5),  'temp':(20,2), 'humidity':(22,3), 'ph':(5.7,0.3), 'rainfall':(105,20), 'yield':(1800,250)},
    'pigeonpeas':   {'N':(20,5),   'P':(68,5),   'K':(20,5),  'temp':(28,2), 'humidity':(48,5), 'ph':(5.8,0.3), 'rainfall':(149,20), 'yield':(1500,200)},
    'mothbeans':    {'N':(21,5),   'P':(48,5),   'K':(20,5),  'temp':(28,2), 'humidity':(53,5), 'ph':(6.8,0.3), 'rainfall':(51,10),  'yield':(900,150)},
    'mungbean':     {'N':(21,5),   'P':(48,5),   'K':(20,5),  'temp':(28,2), 'humidity':(85,5), 'ph':(6.7,0.3), 'rainfall':(49,10),  'yield':(1000,150)},
    'blackgram':    {'N':(40,5),   'P':(68,5),   'K':(20,5),  'temp':(30,2), 'humidity':(65,5), 'ph':(7.0,0.3), 'rainfall':(68,15),  'yield':(800,120)},
    'lentil':       {'N':(18,3),   'P':(68,5),   'K':(20,5),  'temp':(24,2), 'humidity':(65,5), 'ph':(6.9,0.3), 'rainfall':(46,10),  'yield':(1100,180)},
    'pomegranate':  {'N':(18,3),   'P':(18,3),   'K':(40,5),  'temp':(21,2), 'humidity':(90,3), 'ph':(6.5,0.3), 'rainfall':(107,20), 'yield':(8000,1000)},
    'banana':       {'N':(100,10), 'P':(82,8),   'K':(50,5),  'temp':(27,2), 'humidity':(80,5), 'ph':(6.0,0.3), 'rainfall':(105,20), 'yield':(30000,3000)},
    'mango':        {'N':(20,5),   'P':(27,3),   'K':(30,5),  'temp':(31,2), 'humidity':(50,5), 'ph':(5.7,0.3), 'rainfall':(95,20),  'yield':(7000,800)},
    'grapes':       {'N':(23,5),   'P':(133,10), 'K':(200,15),'temp':(24,2), 'humidity':(82,5), 'ph':(6.0,0.3), 'rainfall':(69,15),  'yield':(15000,2000)},
    'watermelon':   {'N':(100,10), 'P':(10,2),   'K':(50,5),  'temp':(25,2), 'humidity':(85,5), 'ph':(6.5,0.3), 'rainfall':(50,10),  'yield':(20000,2500)},
    'muskmelon':    {'N':(100,10), 'P':(18,3),   'K':(50,5),  'temp':(28,2), 'humidity':(92,3), 'ph':(6.5,0.3), 'rainfall':(25,5),   'yield':(12000,1500)},
    'apple':        {'N':(21,3),   'P':(134,10), 'K':(200,15),'temp':(22,2), 'humidity':(92,3), 'ph':(5.8,0.3), 'rainfall':(114,20), 'yield':(10000,1200)},
    'orange':       {'N':(20,3),   'P':(16,3),   'K':(10,2),  'temp':(22,2), 'humidity':(92,3), 'ph':(7.0,0.3), 'rainfall':(110,20), 'yield':(9000,1000)},
    'papaya':       {'N':(50,5),   'P':(59,5),   'K':(50,5),  'temp':(33,2), 'humidity':(92,3), 'ph':(6.7,0.3), 'rainfall':(142,20), 'yield':(25000,3000)},
    'coconut':      {'N':(22,3),   'P':(16,3),   'K':(30,3),  'temp':(27,2), 'humidity':(95,2), 'ph':(5.9,0.3), 'rainfall':(175,20), 'yield':(5000,600)},
    'cotton':       {'N':(118,10), 'P':(45,5),   'K':(43,5),  'temp':(25,3), 'humidity':(65,5), 'ph':(6.5,0.3), 'rainfall':(80,15),  'yield':(2000,300)},
    'jute':         {'N':(78,10),  'P':(46,5),   'K':(40,5),  'temp':(25,2), 'humidity':(80,5), 'ph':(6.7,0.3), 'rainfall':(175,25), 'yield':(2500,350)},
}


def _gaussian_match(val, mean, std):
    return np.exp(-((val - mean) ** 2) / (2 * (std * 2) ** 2))


def generate_dataset(n_per_crop=100, random_state=42, save_path=None):
    """Generate synthetic agricultural dataset based on real crop parameter ranges."""
    np.random.seed(random_state)
    rows = []

    for crop, p in CROP_PROFILES.items():
        for _ in range(n_per_crop):
            N         = max(0.0, np.random.normal(p['N'][0],         p['N'][1]))
            P         = max(0.0, np.random.normal(p['P'][0],         p['P'][1]))
            K         = max(0.0, np.random.normal(p['K'][0],         p['K'][1]))
            temp      = np.random.normal(p['temp'][0],      p['temp'][1])
            humidity  = float(np.clip(np.random.normal(p['humidity'][0],  p['humidity'][1]),  1, 100))
            ph        = float(np.clip(np.random.normal(p['ph'][0],        p['ph'][1]),        0, 14))
            rainfall  = max(0.0, np.random.normal(p['rainfall'][0],  p['rainfall'][1]))

            # Yield correlated with how close conditions are to crop ideal
            match = (
                0.25 * _gaussian_match(N,        p['N'][0],        p['N'][1])        +
                0.20 * _gaussian_match(P,        p['P'][0],        p['P'][1])        +
                0.20 * _gaussian_match(K,        p['K'][0],        p['K'][1])        +
                0.15 * _gaussian_match(temp,     p['temp'][0],     p['temp'][1])     +
                0.10 * _gaussian_match(humidity, p['humidity'][0], p['humidity'][1]) +
                0.10 * _gaussian_match(rainfall, p['rainfall'][0], p['rainfall'][1])
            )
            base_yield = np.random.normal(p['yield'][0], p['yield'][1])
            yield_val  = max(100.0, base_yield * (0.5 + match) + np.random.normal(0, p['yield'][1] * 0.1))

            rows.append({
                'N': round(N, 2), 'P': round(P, 2), 'K': round(K, 2),
                'temperature': round(temp, 2), 'humidity': round(humidity, 2),
                'ph': round(ph, 2), 'rainfall': round(rainfall, 2),
                'label': crop, 'yield': round(yield_val, 2),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def load_and_preprocess(data_path=None, random_state=42):
    """Load (or generate) dataset and return processed splits + artifacts."""
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_dataset(n_per_crop=100, random_state=random_state, save_path=data_path)

    # --- Missing value imputation (median) ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # --- Outlier treatment: 1st–99th percentile capping on soil/climate features ---
    for col in FEATURE_COLS:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lo, upper=hi)

    # --- Yield outlier removal via Z-score (|z| < 3) ---
    y_raw  = df['yield'].values
    y_mean = y_raw.mean()
    y_std  = y_raw.std()
    mask   = np.abs((y_raw - y_mean) / y_std) < 3.0
    df     = df[mask].reset_index(drop=True)

    # --- Feature scaling (StandardScaler) ---
    scaler = StandardScaler()
    X_raw    = df[FEATURE_COLS].values
    X_scaled = scaler.fit_transform(X_raw)

    # --- Label encoding ---
    le       = LabelEncoder()
    y_class  = le.fit_transform(df['label'].values)
    y_reg    = df['yield'].values
    n_classes = len(le.classes_)

    # --- Stratified train/test split (80/20) via index-based split ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    tr_idx, te_idx = next(sss.split(X_scaled, y_class))

    X_tr, X_te   = X_scaled[tr_idx], X_scaled[te_idx]
    yc_tr, yc_te = y_class[tr_idx],  y_class[te_idx]
    yr_tr, yr_te = y_reg[tr_idx],    y_reg[te_idx]

    # --- Regression feature matrix: soil/climate + one-hot crop label ---
    # One-hot encoding lets the model learn a distinct intercept per crop,
    # fixing the low R² that occurs when yields span 800–30,000 kg/ha across crops.
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    crop_ohe_tr = ohe.fit_transform(yc_tr.reshape(-1, 1))
    crop_ohe_te = ohe.transform(yc_te.reshape(-1, 1))
    X_train_reg = np.hstack([X_tr, crop_ohe_tr])
    X_test_reg  = np.hstack([X_te, crop_ohe_te])

    return {
        'X_train': X_tr,            'X_test': X_te,
        'X_train_reg': X_train_reg, 'X_test_reg': X_test_reg,
        'y_cls_train': yc_tr,       'y_cls_test': yc_te,
        'y_reg_train': yr_tr,       'y_reg_test': yr_te,
        'X_all_scaled': X_scaled,
        'X_raw': X_raw,
        'scaler': scaler,
        'label_encoder': le,
        'crop_ohe': ohe,
        'n_classes': n_classes,
        'feature_names': FEATURE_COLS,
        'df': df,
    }
