"""영상 단위 반복 교차검증 재평가.

원 노트북과의 차이
- 분할 1회(seed 42) -> StratifiedGroupKFold 5-fold x 10회 반복 (모든 영상이 반복마다 1번씩 테스트됨)
- 임계값을 테스트셋에서 고름 -> 학습 폴드의 out-of-fold 예측에서만 고름 (테스트셋은 한 번도 안 봄)
- 윈도우 단위 지표 -> 영상 단위 판정(영상 = 제재 단위)으로 보고
하이퍼파라미터는 원 노트북이 찾은 값을 그대로 고정.
"""
import io
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = sys.argv[1]
N_REPEATS = 10
PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", n_jobs=-1, random_state=42,
    colsample_bytree=0.987032220608099, gamma=0.27398594162404366,
    learning_rate=0.09469418846162371, max_depth=5, min_child_weight=1,
    n_estimators=298, subsample=0.7650470707645706,
)

df = pd.read_pickle(DATA) if DATA.endswith(".pkl") else pd.read_excel(DATA)
y = df["cheat"].to_numpy()
g = df["filename"].to_numpy()
X = df.drop(columns=["filename", "cheat", "start_frame"]).to_numpy(dtype=float)


def fit(Xtr, ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return xgb.XGBClassifier(scale_pos_weight=spw, **PARAMS).fit(Xtr, ytr)


def video_table(groups, labels, probs):
    t = pd.DataFrame({"g": groups, "y": labels, "p": probs})
    return t.groupby("g").agg(y=("y", "first"), pmax=("p", "max"), pmean=("p", "mean"))


def cp(k, n, a=0.05):  # Clopper-Pearson
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


rows = []
all_videos = []
for rep in range(N_REPEATS):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    vids = []
    for tr, te in outer.split(X, y, g):
        # 1) 학습 폴드 안에서만 OOF 예측 -> 임계값 결정
        oof = np.zeros(len(tr))
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + rep)
        for itr, iva in inner.split(X[tr], y[tr], g[tr]):
            oof[iva] = fit(X[tr][itr], y[tr][itr]).predict_proba(X[tr][iva])[:, 1]
        vt_tr = video_table(g[tr], y[tr], oof)
        thr_max = vt_tr.loc[vt_tr.y == 0, "pmax"].max()    # 학습 정상 영상 오탐 0이 되는 임계값
        thr_mean = vt_tr.loc[vt_tr.y == 0, "pmean"].max()
        # 2) 전체 학습 폴드로 학습 -> 테스트 영상 채점
        p = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        vt = video_table(g[te], y[te], p)
        vt["flag_max"] = vt.pmax > thr_max      # 원 설계: 한 윈도우라도 임계값 초과면 영상 전체 치트
        vt["flag_mean"] = vt.pmean > thr_mean   # 대안: 영상 평균 확률
        vt["flag_half"] = vt.pmean > 0.5        # 참고: 임계값 조정 없음
        vt["win_tp"] = pd.Series((p > 0.5) & (y[te] == 1)).groupby(g[te]).sum()
        vids.append(vt)
    v = pd.concat(vids)
    all_videos.append(v.assign(rep=rep))
    pos, neg = v[v.y == 1], v[v.y == 0]
    r = dict(rep=rep, n_pos=len(pos), n_neg=len(neg),
             auc_max=roc_auc_score(v.y, v.pmax), auc_mean=roc_auc_score(v.y, v.pmean))
    for k in ("max", "mean", "half"):
        r[f"tp_{k}"] = int(pos[f"flag_{k}"].sum())
        r[f"fp_{k}"] = int(neg[f"flag_{k}"].sum())
    rows.append(r)
    print(f"rep {rep}: " + ", ".join(f"{k} TP {r[f'tp_{k}']}/{len(pos)} FP {r[f'fp_{k}']}/{len(neg)}"
                                      for k in ("max", "mean", "half")), flush=True)

res = pd.DataFrame(rows)
# 영상별 판정 결과 저장 (plot_reeval.py 입력)
pd.concat(all_videos).reset_index(names="filename").to_csv(
    sys.argv[2].replace(".json", "_videos.csv"), index=False)
n_pos, n_neg = int(res.n_pos[0]), int(res.n_neg[0])
print(f"\n영상 수: 치트 {n_pos}, 정상 {n_neg} / 반복 {N_REPEATS}회 x 5-fold")
print(f"영상 단위 ROC-AUC  max 집계: {res.auc_max.mean():.3f} (범위 {res.auc_max.min():.3f}~{res.auc_max.max():.3f})"
      f" | mean 집계: {res.auc_mean.mean():.3f} (범위 {res.auc_mean.min():.3f}~{res.auc_mean.max():.3f})")
names = {"max": "원 설계(윈도우 1개 이상 초과, 임계값은 학습 OOF에서 FP=0)",
         "mean": "대안(영상 평균 확률, 임계값은 학습 OOF에서 FP=0)",
         "half": "참고(영상 평균 확률 > 0.5)"}
summary = {}
for k in ("max", "mean", "half"):
    tp, fp = res[f"tp_{k}"], res[f"fp_{k}"]
    tpm, fpm = int(round(tp.median())), int(round(fp.median()))
    rl, rh = cp(tpm, n_pos)
    fl, fh = cp(fpm, n_neg)
    print(f"\n[{names[k]}]")
    print(f"  재현율  평균 {tp.mean()/n_pos:.3f} (반복별 {tp.min()}~{tp.max()}/{n_pos}), 중앙값 {tpm}/{n_pos} 기준 95% CI {rl:.2f}~{rh:.2f}")
    print(f"  오탐률  평균 {fp.mean()/n_neg:.4f} (반복별 {fp.min()}~{fp.max()}/{n_neg}), 중앙값 {fpm}/{n_neg} 기준 95% CI {fl:.3f}~{fh:.3f}")
    summary[k] = dict(recall_mean=tp.mean() / n_pos, tp_min=int(tp.min()), tp_max=int(tp.max()),
                      fpr_mean=fp.mean() / n_neg, fp_min=int(fp.min()), fp_max=int(fp.max()),
                      recall_ci=[rl, rh], fpr_ci=[fl, fh])
summary["auc_max_mean"] = res.auc_max.mean()
summary["auc_mean_mean"] = res.auc_mean.mean()
json.dump(summary, open(sys.argv[2], "w"), indent=2)
