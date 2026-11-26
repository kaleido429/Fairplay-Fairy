"""홀드아웃 평가: 학습·임계값 결정에 한 번도 쓰이지 않은 영상으로 최종 검증.

절차 (홀드아웃 결과를 보기 전에 고정)
1. 기존 255개 영상(preprocessed_data.xlsx)만으로 임계값 결정: 영상 단위 5-fold OOF 예측에서 정상 영상 오탐 0이 되는 값
2. 기존 255개 전체로 모델 학습 (하이퍼파라미터는 기존 노트북 값 고정)
3. 홀드아웃 영상에 적용. 기존 설계(윈도우 최댓값)와 개선안(영상 평균) 모두 보고

사용: python holdout_eval.py preprocessed_data.xlsx holdout_aim.csv holdout_summary.json
"""
import io
import itertools
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import beta, entropy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

WINDOW, STEP, BINS = 150, 30, 10
RAW = ["dx", "dy", "velocity", "acceleration", "jerk", "angle_deg", "angle_change"]
PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", n_jobs=-1, random_state=42,
    colsample_bytree=0.987032220608099, gamma=0.27398594162404366,
    learning_rate=0.09469418846162371, max_depth=5, min_child_weight=1,
    n_estimators=298, subsample=0.7650470707645706,
)


def window_features(frames):
    """01_training_xgboost_model.ipynb의 윈도우 요약과 동일."""
    rows = []
    for name, vd in frames.groupby("filename", sort=True):
        for i in range(0, len(vd) - WINDOW + 1, STEP):
            ch = vd.iloc[i:i + WINDOW]
            r = {"filename": name, "start_frame": i}
            for c in RAW:
                s = ch[c]
                mode = s.mode()
                r[f"{c}_mean"], r[f"{c}_median"] = s.mean(), s.median()
                r[f"{c}_mode"] = mode.iloc[0] if not mode.empty else np.nan
                r[f"{c}_range"], r[f"{c}_std"] = s.max() - s.min(), s.std()
                r[f"{c}_iqr"] = s.quantile(0.75) - s.quantile(0.25)
                r[f"{c}_max"], r[f"{c}_min"] = s.max(), s.min()
                r[f"{c}_skewness"], r[f"{c}_kurtosis"] = s.skew(), s.kurtosis()
                v = s.values
                r[f"{c}_zcr"] = ((v[:-1] * v[1:]) < 0).sum() / WINDOW
                r[f"{c}_entropy"] = entropy(pd.cut(s, bins=BINS, labels=False).value_counts(normalize=True))
            for a, b in itertools.combinations(RAW, 2):
                k = ch[a].corr(ch[b])
                r[f"corr_{a}_{b}"] = 0 if pd.isna(k) else k
            rows.append(r)
    return pd.DataFrame(rows)


def fit(X, y):
    return xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1), **PARAMS).fit(X, y)


def video_scores(names, probs):
    t = pd.DataFrame({"filename": names, "p": probs})
    return t.groupby("filename").agg(pmax=("p", "max"), pmean=("p", "mean"))


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return round(float(lo), 3), round(float(hi), 3)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    orig = pd.read_excel(sys.argv[1])
    feat_cols = [c for c in orig.columns if c not in ("filename", "cheat", "start_frame")]
    X, y, g = orig[feat_cols].to_numpy(float), orig["cheat"].to_numpy(), orig["filename"].to_numpy()

    # 1) 임계값: 기존 데이터의 OOF 예측에서만 결정
    oof = np.zeros(len(orig))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        oof[va] = fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    vs = video_scores(g, oof).join(orig.groupby("filename")["cheat"].first())
    thr = {"max": float(vs.loc[vs.cheat == 0, "pmax"].max()), "mean": float(vs.loc[vs.cheat == 0, "pmean"].max())}
    print(f"임계값 (기존 데이터 OOF 기준): max 규칙 {thr['max']:.5f}, mean 규칙 {thr['mean']:.4f}")

    # 2) 기존 데이터 전체로 학습
    model = fit(X, y)

    # 3) 홀드아웃 적용
    hold = window_features(pd.read_csv(sys.argv[2]))
    hv = video_scores(hold["filename"].to_numpy(), model.predict_proba(hold[feat_cols].to_numpy(float))[:, 1])
    hv["cheat"] = hv.index.str.startswith("cheats/").astype(int)
    hv["flag_max"], hv["flag_mean"] = hv.pmax > thr["max"], hv.pmean > thr["mean"]
    hv["flag_half"] = hv.pmean > 0.5
    pos, neg = hv[hv.cheat == 1], hv[hv.cheat == 0]
    out = {"n_cheat": len(pos), "n_normal": len(neg), "thresholds": thr,
           "auc_max": float(roc_auc_score(hv.cheat, hv.pmax)), "auc_mean": float(roc_auc_score(hv.cheat, hv.pmean))}
    print(f"\n홀드아웃: 치트 {len(pos)}개, 정상 {len(neg)}개")
    print(f"ROC-AUC  max 집계 {out['auc_max']:.3f} | mean 집계 {out['auc_mean']:.3f}")
    for k, name in [("max", "기존 설계 (Baseline)"), ("mean", "개선안 (Improved)"), ("half", "참고: 평균 > 0.5")]:
        tp, fp = int(pos[f"flag_{k}"].sum()), int(neg[f"flag_{k}"].sum())
        out[k] = dict(tp=tp, fp=fp, recall=tp / len(pos), fpr=fp / len(neg),
                      recall_ci=cp(tp, len(pos)), fpr_ci=cp(fp, len(neg)))
        print(f"[{name}] 적발 {tp}/{len(pos)} (재현율 {tp/len(pos):.2f}, 95% CI {out[k]['recall_ci']}) | "
              f"오탐 {fp}/{len(neg)} (오탐률 {fp/len(neg):.3f}, 95% CI {out[k]['fpr_ci']})")
    hv.reset_index().to_csv(sys.argv[3].replace(".json", "_videos.csv"), index=False)
    json.dump(out, open(sys.argv[3], "w"), indent=2)
