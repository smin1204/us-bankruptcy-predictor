
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_recall_curve, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

print(" 데이터 로딩 중...")
df = pd.read_csv("American_Bankruptcy.csv")
df.columns = df.columns.str.strip()
print(f"   전체 데이터: {len(df):,}행 × {df.shape[1]}열")

df["target"] = df["status_label"].apply(
    lambda x: 1 if str(x).strip().lower() == "failed" else 0
)
print(f"   파산(1): {df['target'].sum():,}건  |  정상(0): {(df['target']==0).sum():,}건")

eps = 1e-9  

def safe_div(a, b):
    return a / (b + eps)

df["WC_TA"]   = safe_div(df["X1"] - df["X14"], df["X10"])
df["RE_TA"]   = safe_div(df["X15"], df["X10"])
df["EBIT_TA"] = safe_div(df["X12"], df["X10"])
df["MV_TL"]   = safe_div(df["X8"],  df["X17"])
df["S_TA"]    = safe_div(df["X9"],  df["X10"])

df["NI_S"]    = safe_div(df["X6"],  df["X9"])
df["GP_S"]    = safe_div(df["X13"], df["X9"])
df["EBITDA_TA"] = safe_div(df["X4"], df["X10"])

df["TL_TA"]   = safe_div(df["X17"], df["X10"])
df["CA_CL"]   = safe_div(df["X1"],  df["X14"])
df["LTD_TA"]  = safe_div(df["X11"], df["X10"])
df["REC_S"]   = safe_div(df["X7"],  df["X9"])

FEATURES = [
    "WC_TA", "RE_TA", "EBIT_TA", "MV_TL", "S_TA",
    "NI_S",  "GP_S",  "EBITDA_TA",
    "TL_TA", "CA_CL", "LTD_TA",  "REC_S"
]

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
for col in FEATURES:
    p1, p99 = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(p1, p99).fillna(df[col].median())

train_df = df[(df["year"] >= 1999) & (df["year"] <= 2011)]
val_df   = df[(df["year"] >= 2012) & (df["year"] <= 2014)]
test_df  = df[(df["year"] >= 2015) & (df["year"] <= 2018)]

X_train, y_train = train_df[FEATURES].values, train_df["target"].values
X_val,   y_val   = val_df[FEATURES].values,   val_df["target"].values
X_test,  y_test  = test_df[FEATURES].values,  test_df["target"].values

print(f"\n 분할 완료")
print(f"   훈련(1999-2011): {len(X_train):,}행  파산률 {y_train.mean()*100:.2f}%")
print(f"   검증(2012-2014): {len(X_val):,}행   파산률 {y_val.mean()*100:.2f}%")
print(f"   테스트(2015-2018): {len(X_test):,}행  파산률 {y_test.mean()*100:.2f}%")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print("\n  SMOTE 오버샘플링 적용 중...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
print(f"   오버샘플링 후: {len(X_train_res):,}행  (파산/정상 = 1:1)")

print("\n 모델 훈련 중...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft",
    weights=[1, 2]  
)
ensemble.fit(X_train_res, y_train_res)

print(" 확률 캘리브레이션 적용 중...")
calibrated = CalibratedClassifierCV(ensemble, method="isotonic", cv="prefit")
calibrated.fit(X_val_sc, y_val)

val_proba = calibrated.predict_proba(X_val_sc)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
best_idx   = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

val_pred_opt = (val_proba >= best_threshold).astype(int)
print(f"\n 검증 세트 결과 (최적 임계값: {best_threshold:.3f})")
print(classification_report(y_val, val_pred_opt, target_names=["정상", "파산"]))
print(f"   ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

test_proba    = calibrated.predict_proba(X_test_sc)[:, 1]
test_pred_opt = (test_proba >= best_threshold).astype(int)

print(f"\n 테스트 세트 최종 결과 (2015-2018)")
print(classification_report(y_test, test_pred_opt, target_names=["정상", "파산"]))
print(f"   ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")

cm = confusion_matrix(y_test, test_pred_opt)
print(f"\n   혼동 행렬:\n{cm}")

rf_fitted = ensemble.estimators_[0]
importances = pd.Series(rf_fitted.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
print("\n 특징 중요도 (상위 12개):")
print(importances.to_string())

joblib.dump(calibrated, "us_bankruptcy_model.joblib")
joblib.dump(scaler, "us_scaler.joblib")
joblib.dump(FEATURES, "us_features.joblib")
joblib.dump(best_threshold, "us_threshold.joblib")
joblib.dump(importances.to_dict(), "us_feature_importance.joblib")

print("\n 낱개 파일 5개 강제 저장 완료!")
print(f"   최적 임계값: {best_threshold:.4f}")