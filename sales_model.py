"""
점포당 매출 예측 모델 (학습 + 검증)
===================================
이전 방식: "이 상권 같은 업종 가게들의 지난 분기 평균 매출"을 그대로 쓰고, 보수/낙관 폭은 임의로 정함.
이 모듈:   서울 상권 × 업종 수만 개 사례로 '어떤 조건이면 가게 하나가 얼마를 버는지'를 학습하고,
           실제로 얼마나 맞히는지 검증한 뒤, 검증된 오차로 예측 범위를 만든다.

[무엇을 맞히나]  log(점포당 월매출)   — 상권 × 업종 한 쌍이 사례 하나. 점포 3곳 미만은 학습에서 제외
[무엇을 보고]    상권: 하루 유동인구·밀도, 연령/시간대/주말/여성 비율, 직장인구, 집객시설(역·대학·관공서 등),
                       상권 면적·구분·자치구, 상권 전체 점포 수, '다른 업종' 점포당 매출(상권 활력)
                 업종: 업종 종류, 같은 업종 점포 수(경쟁), 프랜차이즈 비율, 개업률·폐업률
                 (모델 A만) 지난 분기 같은 상권·업종의 점포당 매출
[방법]           Gradient Boosting(sklearn HistGradientBoosting) 분위수 회귀 3개 (하위10% / 중앙 / 상위10%)
                 → 80% 예측 구간. 검증 데이터로 구간 폭을 보정(Conformalized Quantile Regression)
[검증]           ① 시간 검증: 직전 분기로 학습 → 최신 분기를 예측해 실제와 비교
                 ② 공간 검증: 상권을 5묶음으로 나눠, 학습에 없던 상권을 예측 (처음 보는 동네에 대한 성능)
                 ③ 기준선과 비교: '지난 분기 값 그대로', '서울 업종 중앙값' 보다 나은지 확인
[내가 들어가면]  예측할 때 같은 업종 점포 수를 +1 해서, 경쟁이 하나 늘어난 상황의 점포당 매출을 예측
"""

import math
import os
import pickle
import tempfile
import threading
import time

import numpy as np
import pandas as pd

from compare_trade_area import CONFIG
from foot_traffic import load_quarter_table, find_latest_quarter, previous_quarter
from location_recommender import get_trade_area_meta, SALES_FIELDS, STORE_FIELDS, MIN_STORES

QUANTILES = (0.1, 0.5, 0.9)
COVERAGE = 0.8
MODEL_TTL_SEC = 12 * 3600
MODEL_PATH = os.path.join(tempfile.gettempdir(), "seoul_sales_model_v1.pkl")

AGE_F = ["AGRDE_10_FLPOP_CO", "AGRDE_20_FLPOP_CO", "AGRDE_30_FLPOP_CO",
         "AGRDE_40_FLPOP_CO", "AGRDE_50_FLPOP_CO", "AGRDE_60_ABOVE_FLPOP_CO"]
TIME_F = ["TMZON_00_06_FLPOP_CO", "TMZON_06_11_FLPOP_CO", "TMZON_11_14_FLPOP_CO",
          "TMZON_14_17_FLPOP_CO", "TMZON_17_21_FLPOP_CO", "TMZON_21_24_FLPOP_CO"]
FCLTY_F = ["VIATR_FCLTY_CO", "SUBWAY_STATN_CO", "BUS_STTN_CO", "RLROAD_STATN_CO", "UNIV_CO",
           "HGSCHL_CO", "MSKUL_CO", "ELESCH_CO", "PBLOFC_CO", "GEHSPT_CO", "GNRL_HSPTL_CO",
           "BANK_CO", "DRTS_CO", "SUPMK_CO", "THEAT_CO", "STAYNG_FCLTY_CO"]
CAT_F = ["svc", "area_type", "gu"]

_LOCK = threading.Lock()
_MODEL = {"bundle": None, "error": None}


def _df(rows):
    df = pd.DataFrame(rows or [])
    if "TRDAR_CD" in df:
        df["TRDAR_CD"] = df["TRDAR_CD"].astype(str)
    return df


def _num_cols(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df


# ---------------------------------------------------------------------------
# 1) 데이터 불러오기
# ---------------------------------------------------------------------------

def latest_sales_quarter():
    return find_latest_quarter("VwsmTrdarSelngQq") or find_latest_quarter() or "20241"


def load_inputs(Q):
    q1, q2 = previous_quarter(Q), previous_quarter(previous_quarter(Q))
    sales = {q: load_quarter_table("VwsmTrdarSelngQq", q, SALES_FIELDS)[0] for q in (Q, q1, q2)}
    stores = {q: load_quarter_table("VwsmTrdhlStorQq", q, STORE_FIELDS)[0] for q in (Q, q1)}
    flpop = {q: load_quarter_table("VwsmTrdarFlpopQq", q)[0] for q in (Q, q1)}
    # 직장인구·집객시설은 분기마다 거의 안 변하므로 최신 것 하나를 공통으로 쓴다
    wrc = load_quarter_table("VwsmTrdarWrcPopltnQq", Q)[0]
    fclty = load_quarter_table("VwsmTrdarFcltyQq", Q)[0]
    meta = get_trade_area_meta()
    return {"Q": Q, "q1": q1, "q2": q2, "sales": sales, "stores": stores,
            "flpop": flpop, "wrc": wrc, "fclty": fclty, "meta": meta}


# ---------------------------------------------------------------------------
# 2) 특징(입력값) 만들기
# ---------------------------------------------------------------------------

def area_features(flpop_rows, wrc_rows, fclty_rows, meta):
    """상권 단위 특징표 (index = 상권코드)."""
    fl = _num_cols(_df(flpop_rows), ["TOT_FLPOP_CO", "FML_FLPOP_CO", "SAT_FLPOP_CO", "SUN_FLPOP_CO"] + AGE_F + TIME_F)
    out = pd.DataFrame(index=fl["TRDAR_CD"] if len(fl) else [])
    if len(fl):
        fl = fl.drop_duplicates("TRDAR_CD").set_index("TRDAR_CD")
        tot = fl["TOT_FLPOP_CO"].replace(0, np.nan)
        out["flpop_log"] = np.log1p(fl["TOT_FLPOP_CO"] / 91)
        out["female_share"] = fl["FML_FLPOP_CO"] / tot
        out["weekend_share"] = (fl["SAT_FLPOP_CO"] + fl["SUN_FLPOP_CO"]) / tot
        age_tot = fl[AGE_F].sum(axis=1).replace(0, np.nan)
        for c in AGE_F:
            out["age_" + c.split("_")[1]] = fl[c] / age_tot
        t_tot = fl[TIME_F].sum(axis=1).replace(0, np.nan)
        for c in TIME_F:
            out["time_" + c[6:11]] = fl[c] / t_tot

    wr = _num_cols(_df(wrc_rows), ["TOT_WRC_POPLTN_CO", "AGRDE_20_WRC_POPLTN_CO", "AGRDE_30_WRC_POPLTN_CO"])
    if len(wr):
        wr = wr.drop_duplicates("TRDAR_CD").set_index("TRDAR_CD")
        out = out.join(pd.DataFrame({
            "worker_log": np.log1p(wr["TOT_WRC_POPLTN_CO"]),
            "worker_2030_share": (wr["AGRDE_20_WRC_POPLTN_CO"] + wr["AGRDE_30_WRC_POPLTN_CO"])
                                 / wr["TOT_WRC_POPLTN_CO"].replace(0, np.nan),
        }), how="left")

    fc = _num_cols(_df(fclty_rows), FCLTY_F)
    if len(fc):
        fc = fc.drop_duplicates("TRDAR_CD").set_index("TRDAR_CD")[FCLTY_F]
        fc.columns = ["f_" + c[:-3].lower() for c in FCLTY_F]
        out = out.join(fc, how="left")

    m = pd.DataFrame.from_dict(meta, orient="index")
    if len(m):
        m.index = m.index.astype(str)
        area = pd.to_numeric(m.get("면적"), errors="coerce")
        out = out.join(pd.DataFrame({
            "area_log": np.log1p(area),
            "area_type": m.get("상권구분"),
            "gu": m.get("자치구"),
        }), how="left")
        if "flpop_log" in out:
            out["flpop_density_log"] = out["flpop_log"] - out["area_log"]
    return out


def pair_frame(sales_rows, store_rows, prev_sales_rows, area_df):
    """상권 × 업종 사례표. per_store(만원) = 매출 ÷ 점포 수, lag = 지난 분기 점포당 매출."""
    months = CONFIG["SALES_AMT_MONTHS"]
    st = _num_cols(_df(store_rows), ["SIMILR_INDUTY_STOR_CO", "STOR_CO", "FRC_STOR_CO", "OPBIZ_RT", "CLSBIZ_RT"])
    if not len(st):
        return pd.DataFrame()
    st["n"] = st["SIMILR_INDUTY_STOR_CO"].fillna(st["STOR_CO"])
    st = st.rename(columns={"SVC_INDUTY_CD_NM": "svc"})[["TRDAR_CD", "svc", "n", "FRC_STOR_CO", "OPBIZ_RT", "CLSBIZ_RT"]]
    st = st[st["n"] > 0].drop_duplicates(["TRDAR_CD", "svc"])

    sa = _num_cols(_df(sales_rows), ["THSMON_SELNG_AMT", "THSMON_SELNG_CO"])
    if len(sa):
        sa = sa.rename(columns={"SVC_INDUTY_CD_NM": "svc", "THSMON_SELNG_AMT": "amt", "THSMON_SELNG_CO": "cnt"})
        sa = sa.groupby(["TRDAR_CD", "svc"], as_index=False)[["amt", "cnt"]].sum()
        df = st.merge(sa, on=["TRDAR_CD", "svc"], how="left")
    else:
        df = st.assign(amt=np.nan, cnt=np.nan)
    df["amt"] = df["amt"] / months

    ps = _num_cols(_df(prev_sales_rows), ["THSMON_SELNG_AMT"])
    if len(ps):
        ps = ps.rename(columns={"SVC_INDUTY_CD_NM": "svc", "THSMON_SELNG_AMT": "amt_prev"})
        ps = ps.groupby(["TRDAR_CD", "svc"], as_index=False)["amt_prev"].sum()
        df = df.merge(ps, on=["TRDAR_CD", "svc"], how="left")
    else:
        df["amt_prev"] = np.nan

    df["per_store"] = df["amt"] / df["n"] / 10000
    df["ticket"] = df["amt"] / df["cnt"]
    df["lag_log"] = np.log(df["amt_prev"] / months / df["n"] / 10000)

    # 상권 전체 점포 수, 다른 업종의 점포당 매출(상권 활력)
    tot = df.groupby("TRDAR_CD").agg(total_n=("n", "sum"), total_amt=("amt", "sum"), n_types=("svc", "count"))
    df = df.join(tot, on="TRDAR_CD")
    df = df.join(area_df, on="TRDAR_CD")
    return add_pair_features(df)


def add_pair_features(df):
    other_amt = df["total_amt"] - df["amt"].fillna(0)
    other_n = (df["total_n"] - df["n"]).replace(0, np.nan)
    df["other_per_store_log"] = np.log1p((other_amt / other_n / 10000).clip(lower=0))
    df["n_log"] = np.log(df["n"])
    df["total_n_log"] = np.log1p(df["total_n"])
    df["svc_share"] = df["n"] / df["total_n"]
    df["frc_share"] = df["FRC_STOR_CO"] / df["n"]
    return df


FEATS_B = (["svc", "area_type", "gu", "n_log", "total_n_log", "svc_share", "frc_share", "OPBIZ_RT", "CLSBIZ_RT",
            "other_per_store_log", "n_types", "flpop_log", "flpop_density_log", "female_share", "weekend_share",
            "worker_log", "worker_2030_share", "area_log"]
           + ["age_" + a for a in ("10", "20", "30", "40", "50", "60")]
           + ["time_" + t for t in ("00_06", "06_11", "11_14", "14_17", "17_21", "21_24")]
           + ["f_" + c[:-3].lower() for c in FCLTY_F])
FEATS_A = FEATS_B + ["lag_log"]


def _X(df, feats, categories):
    X = pd.DataFrame(index=df.index)
    for c in feats:
        if c in CAT_F:
            X[c] = pd.Categorical(df[c] if c in df else None, categories=categories[c])
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
            if X[c].isna().all():
                X[c] = 0.0  # 값이 전혀 없는 열은 학습 오류를 일으켜서 0으로 채움 (정보 없음과 같음)
    return X


def _trainable(df):
    return df[(df["n"] >= MIN_STORES) & (df["per_store"] > 0)].copy()


# ---------------------------------------------------------------------------
# 3) 학습 · 평가
# ---------------------------------------------------------------------------

def _fit(df, feats, categories):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y = _X(df, feats, categories), np.log(df["per_store"].values)
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, categorical_features="from_dtype",
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, random_state=0,
        )
        m.fit(X, y)
        models[q] = m
    return models


def _predict_log(models, df, feats, categories, qhat=0.0):
    X = _X(df, feats, categories)
    P = np.column_stack([models[q].predict(X) for q in QUANTILES])
    P.sort(axis=1)  # 하위 ≤ 중앙 ≤ 상위 보장
    P[:, 0] -= qhat
    P[:, 2] += qhat
    return P


def _metrics(y_manwon, pred_manwon, lo=None, hi=None):
    y, p = np.asarray(y_manwon, float), np.asarray(pred_manwon, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    y, p = y[ok], p[ok]
    if not len(y):
        return None
    ape = np.abs(p - y) / y
    ly, lp = np.log(y), np.log(p)
    r2 = 1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2)
    out = {
        "사례수": int(len(y)),
        "오차_중앙값(%)": round(float(np.median(ape)) * 100, 1),
        "오차_30%이내_비율(%)": round(float(np.mean(ape <= 0.3)) * 100, 1),
        "R2(로그)": round(float(r2), 3),
    }
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float)[ok], np.asarray(hi, float)[ok]
        out["80%구간_적중률(%)"] = round(float(np.mean((y >= lo) & (y <= hi))) * 100, 1)
        out["구간폭_중앙값(배)"] = round(float(np.median(hi / lo)), 2)
    return out


def _cqr_qhat(P_log, y_log):
    """검증 데이터에서 80% 적중하도록 구간을 넓힐 양(로그 단위)."""
    E = np.maximum(P_log[:, 0] - y_log, y_log - P_log[:, 2])
    n = len(E)
    level = min(1.0, COVERAGE * (1 + 1 / n)) if n else COVERAGE
    return float(max(0.0, np.quantile(E, level))) if n else 0.0


def _eval_block(models, test, feats, categories, qhat=0.0):
    P = np.exp(_predict_log(models, test, feats, categories, qhat))
    return _metrics(test["per_store"], P[:, 1], P[:, 0], P[:, 2]), P


def train_and_validate(inp):
    Q, q1 = inp["Q"], inp["q1"]
    area_Q = area_features(inp["flpop"][Q], inp["wrc"], inp["fclty"], inp["meta"])
    area_1 = area_features(inp["flpop"][q1] or inp["flpop"][Q], inp["wrc"], inp["fclty"], inp["meta"])
    frame_Q = pair_frame(inp["sales"][Q], inp["stores"][Q], inp["sales"][q1], area_Q)
    frame_1 = pair_frame(inp["sales"][q1], inp["stores"][q1] or inp["stores"][Q], inp["sales"][inp["q2"]], area_1)
    train_Q, train_1 = _trainable(frame_Q), _trainable(frame_1)
    if len(train_Q) < 300:
        raise RuntimeError("학습 사례가 너무 적습니다 (" + str(len(train_Q)) + "건)")

    categories = {c: sorted(set(frame_Q[c].dropna().astype(str)) | set(frame_1[c].dropna().astype(str)))
                  for c in CAT_F}
    for df in (frame_Q, frame_1, train_Q, train_1):
        for c in CAT_F:
            df[c] = df[c].astype(str).where(df[c].notna())

    report = {"기준_분기": Q, "학습_분기(시간검증)": q1, "학습_사례수": int(len(train_Q))}
    qhat = {}

    # ① 시간 검증: 직전 분기로 학습 -> 최신 분기 예측
    if len(train_1) >= 300:
        test = train_Q
        for name, feats in (("A", FEATS_A), ("B", FEATS_B)):
            models = _fit(train_1, feats, categories)
            P_log = _predict_log(models, test, feats, categories)
            qhat[name] = _cqr_qhat(P_log, np.log(test["per_store"].values))
            P = np.exp(P_log)
            m = _metrics(test["per_store"], P[:, 1], P[:, 0], P[:, 2])
            if m and "80%구간_적중률(%)" in m:
                # 이 값은 구간 보정 '전' 적중률. 이 검증 결과로 보정값(qhat)을 정해 배포 모델에 적용한다
                m["80%구간_적중률_보정전(%)"] = m.pop("80%구간_적중률(%)")
            report["시간검증_모델" + name] = m
        lagged = test[np.isfinite(test["lag_log"])]
        report["시간검증_기준선_지난분기값"] = _metrics(lagged["per_store"], np.exp(lagged["lag_log"]))
        med = train_1.groupby("svc")["per_store"].median()
        report["시간검증_기준선_업종중앙값"] = _metrics(test["per_store"], test["svc"].map(med))
    else:
        qhat = {"A": 0.0, "B": 0.0}
        report["시간검증"] = "직전 분기 데이터가 부족해 건너뜀"

    # ② 공간 검증: 학습에 없던 상권 예측 (모델 B, 상권 단위 5묶음)
    from sklearn.model_selection import GroupKFold
    preds = np.full((len(train_Q), 3), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(train_Q, groups=train_Q["TRDAR_CD"]):
        models = _fit(train_Q.iloc[tr], FEATS_B, categories)
        preds[te] = np.exp(_predict_log(models, train_Q.iloc[te], FEATS_B, categories, qhat.get("B", 0.0)))
    report["공간검증_모델B(처음보는상권)"] = _metrics(train_Q["per_store"], preds[:, 1], preds[:, 0], preds[:, 2])

    # ③ 배포용: 최신 분기 전체로 다시 학습
    final = {name: _fit(train_Q, feats, categories) for name, feats in (("A", FEATS_A), ("B", FEATS_B))}
    report["구간보정값(로그)"] = {k: round(v, 3) for k, v in qhat.items()}
    report["설명"] = ("모델A=지난 분기 매출까지 사용, 모델B=동네 특성만 사용. 오차_중앙값은 예측이 실제와 보통 몇 % "
                    "차이 나는지, 80%구간_적중률은 '예측 범위 안에 실제 값이 들어간 비율'(목표 80%).")
    return {
        "created": time.time(), "Q": Q, "models": final, "qhat": qhat, "categories": categories,
        "frame_Q": frame_Q, "area_Q": area_Q, "report": report,
    }


# ---------------------------------------------------------------------------
# 4) 모델 가져오기 (메모리 → 디스크 → 새로 학습)
# ---------------------------------------------------------------------------

def get_model(force=False):
    with _LOCK:
        b = _MODEL["bundle"]
        if b and not force and time.time() - b["created"] < MODEL_TTL_SEC:
            return b
        if not force and os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    b = pickle.load(f)
                if time.time() - b["created"] < MODEL_TTL_SEC:
                    _MODEL["bundle"] = b
                    return b
            except Exception:
                pass
        try:
            b = train_and_validate(load_inputs(latest_sales_quarter()))
            _MODEL["bundle"], _MODEL["error"] = b, None
            try:
                with open(MODEL_PATH, "wb") as f:
                    pickle.dump(b, f)
            except Exception:
                pass
            return b
        except Exception as e:
            _MODEL["error"] = str(e)
            return None


def model_report():
    b = get_model()
    if not b:
        return {"error": "모델 학습 실패: " + str(_MODEL.get("error"))}
    return b["report"]


# ---------------------------------------------------------------------------
# 5) 예측: "이 상권에 이 업종 가게를 하나 더 열면, 점포당 월매출은?"
# ---------------------------------------------------------------------------

def predict_entrant(svc, trdar_cds):
    """
    반환: {상권코드: {"하위10%": 만원, "중앙": 만원, "상위10%": 만원, "점포수_진입후": n+1,
                     "사용모델": "A"|"B", "신뢰도": 설명}}
    같은 업종이 없는 상권(n=0)도 예측하지만, 학습 범위(점포 3곳 이상) 밖이라 신뢰도가 낮다고 표시한다.
    """
    b = get_model()
    if not b:
        return None
    fq, area, cats = b["frame_Q"], b["area_Q"], b["categories"]
    rows = []
    for cd in trdar_cds:
        cd = str(cd)
        hit = fq[(fq["TRDAR_CD"] == cd) & (fq["svc"] == svc)]
        if len(hit):
            r = hit.iloc[0].to_dict()
        else:
            base = fq[fq["TRDAR_CD"] == cd]
            r = {"TRDAR_CD": cd, "svc": svc, "n": 0, "FRC_STOR_CO": 0, "OPBIZ_RT": np.nan, "CLSBIZ_RT": np.nan,
                 "amt": 0.0, "lag_log": np.nan,
                 "total_n": float(base["n"].sum()) if len(base) else 0.0,
                 "total_amt": float(base["amt"].sum()) if len(base) else 0.0,
                 "n_types": float(len(base))}
            if cd in area.index:
                r.update(area.loc[cd].to_dict())
        r["n_before"] = r["n"]
        r["n"] = r["n"] + 1           # 내가 들어간 뒤의 점포 수
        r["total_n"] = r["total_n"] + 1
        rows.append(r)
    if not rows:
        return {}
    df = add_pair_features(pd.DataFrame(rows))
    for c in CAT_F:
        df[c] = df[c].astype(str).where(df[c].notna())

    out = {}
    has_lag = np.isfinite(pd.to_numeric(df["lag_log"], errors="coerce"))
    for name, feats, mask in (("A", FEATS_A, has_lag), ("B", FEATS_B, ~has_lag)):
        sub = df[mask.values]
        if not len(sub):
            continue
        P = np.exp(_predict_log(b["models"][name], sub, feats, cats, b["qhat"].get(name, 0.0)))
        for (_, r), p in zip(sub.iterrows(), P):
            nb = int(r["n_before"])
            out[r["TRDAR_CD"]] = {
                "하위10%": round(float(p[0])), "중앙": round(float(p[1])), "상위10%": round(float(p[2])),
                "점포수_진입전": nb, "점포수_진입후": nb + 1, "사용모델": name,
                "신뢰도": ("보통" if nb >= MIN_STORES else "낮음 - 같은 업종 점포가 " + str(MIN_STORES)
                         + "곳 미만이라 학습 범위 밖"),
            }
    return out


def model_summary_for_user():
    """도구 결과에 붙일 짧은 검증 요약."""
    b = get_model()
    if not b:
        return None
    r = b["report"]
    t = r.get("시간검증_모델A") or {}
    s = r.get("공간검증_모델B(처음보는상권)") or {}
    return {
        "기준_분기": r.get("기준_분기"),
        "학습_사례수": r.get("학습_사례수"),
        "시간검증_오차중앙값(%)": t.get("오차_중앙값(%)"),
        "처음보는상권_오차중앙값(%)": s.get("오차_중앙값(%)"),
        "80%예측범위_적중률(%)(처음보는상권, 보정후)": s.get("80%구간_적중률(%)"),
        "기준선(지난분기값)_오차중앙값(%)": (r.get("시간검증_기준선_지난분기값") or {}).get("오차_중앙값(%)"),
    }
