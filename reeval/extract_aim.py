"""영상 폴더 -> 프레임별 에임 특징 CSV.

opticalflowExcel.ipynb(학습 데이터 aim_analysis_*.xlsx를 만든 노트북)의 처리 과정을 그대로 옮긴 것.
- 시작 3초 스킵 (fps 메타데이터 기준)
- 원본과 다른 점 2가지
  1) 경로에 한글 등 비ASCII 문자가 있으면 OpenCV(Windows)가 못 열어서 임시 폴더로 복사 후 처리
  2) fps 메타데이터가 비정상(0 이하 또는 120 초과)이면 30으로 간주. 원본은 이 경우 분석 실패 처리.
     해당 영상은 meta의 fps_fixed 열에 표시
- 화면 중앙 60% ROI, Farneback(0.5, 3, 15, 3, 5, 1.2, 0), 평균 흐름의 역벡터 = 에임 이동

사용: python extract_aim.py <영상 루트 폴더> <출력 csv> [프로세스 수]
filename 열은 루트 기준 상대 경로 (예: cheats/xxxx.webm)
"""
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd


def track_aim(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fixed = not (0 < raw_fps <= 120)
    fps = 30 if fixed else int(raw_fps)
    if fps > 0:
        for _ in range(int(fps * 3)):
            if not cap.read()[0]:
                cap.release()
                return None
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return None
    x, y, rw, rh = int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)[y:y + rh, x:x + rw]
    moves = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + rh, x:x + rw]
        flow = cv2.calcOpticalFlowFarneback(prev, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        moves.append((-float(np.mean(flow[..., 0])), -float(np.mean(flow[..., 1]))))
        prev = roi
    cap.release()
    return dict(moves=moves, width=w, height=h, fps=fps, raw_fps=raw_fps, fps_fixed=fixed)


def per_frame(name, moves):
    m = np.array(moves)
    dx, dy = m[:, 0], m[:, 1]
    acc = np.concatenate([[np.nan], np.linalg.norm(np.diff(m, axis=0), axis=1)])
    jerk = np.concatenate([[np.nan, np.nan], np.linalg.norm(np.diff(m, axis=0, n=2), axis=1)])
    ang = np.arctan2(dy, dx) * 180 / np.pi
    dang = np.concatenate([[np.nan], np.abs((np.diff(ang) + 180) % 360 - 180)])
    return pd.DataFrame(dict(filename=name, frame=np.arange(len(m)), dx=dx, dy=dy,
                             velocity=np.linalg.norm(m, axis=1), acceleration=acc, jerk=jerk,
                             angle_deg=ang, angle_change=dang))


def work(args):
    root, rel = args
    path = os.path.join(root, rel)
    if path.isascii():
        r = track_aim(path)
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "v.webm")
            shutil.copyfile(path, tmp)
            r = track_aim(tmp)
    if r is None or len(r["moves"]) < 3:
        return rel, None, None
    return rel, per_frame(rel, r["moves"]), (r["width"], r["height"], r["fps"], r["raw_fps"], r["fps_fixed"], len(r["moves"]))


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else max(os.cpu_count() // 2, 1)
    rels = sorted(os.path.relpath(os.path.join(d, f), root).replace("\\", "/")
                  for d, _, fs in os.walk(root) for f in fs if f.endswith(".webm"))
    frames, meta = [], []
    with Pool(n_proc) as pool:
        for i, (rel, df, info) in enumerate(pool.imap_unordered(work, [(root, r) for r in rels]), 1):
            print(f"[{i}/{len(rels)}] {rel} {'FAILED' if df is None else info}", flush=True)
            if df is not None:
                frames.append(df)
                meta.append((rel, *info))
    pd.concat(frames).sort_values(["filename", "frame"]).to_csv(out, index=False)
    pd.DataFrame(meta, columns=["filename", "width", "height", "fps", "raw_fps", "fps_fixed", "frames"]).sort_values("filename").to_csv(
        out.replace(".csv", "_meta.csv"), index=False)
