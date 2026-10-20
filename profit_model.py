"""
매물 수익 예측 모듈 (창업자용)
==============================
"이 매물(보증금 O, 월세 O, O평)에서 카페를 하면 한 달에 얼마 남을까?"에 답한다.

계산 순서
  1) 예상 월매출
     - 방법 0 (주 계산): sales_model.py의 학습·검증된 예측 모델. 내가 들어가 점포가 1곳 늘어난 상황의
                          점포당 월매출 하위10%/중앙/상위10% → 보수/보통/낙관. 층수·면적 보정만 가정값.
     - 방법 1 (모델 실패 시): 이 상권 같은 업종의 점포당 월매출 × 층수·면적 보정
                          (점포당 월매출 = 서울시 추정매출 ÷ 점포 수. 추정매출을 전체 매출로 간주)
     - 방법 2 (검산용): 하루 예상 방문객 × 객단가(결제 1건당 평균 금액) × 한 달 영업일
     - 보수(평균의 70%) / 보통(100%) /낙관(120%) 세 가지 시나리오
       (새 가게는 첫해에 동네 평균보다 못 버는 경우가 많아서 보수 시나리오를 함께 본다)
  2) 월 비용 = 변동비(재료비·인건비·공과금·배달수수료·카드수수료·소모품, 매출에 비례)
              + 고정비(월세·관리비·기타 고정비)
  3) 월 순이익 = 매출 − 비용   (세금 전, 사장 본인 인건비는 빼지 않은 금액)
  4) 손익분기 매출, 손익분기 하루 손님 수, 초기 투자금 회수 기간

아래 비율들은 업계에서 흔히 쓰는 평균적인 가정값이다. 실제 가게마다 크게 다를 수 있으므로
COST_PROFILES 표를 고치면 계산 전체에 반영된다.
"""

from compare_trade_area import CONFIG, CATEGORY_DEMAND_PROFILE, summarize_population
from foot_traffic import (
    find_latest_quarter, load_quarter_table, summarize_detailed_flpop, estimate_target_customers,
)
from location_recommender import (
    resolve_business, _load_sales_and_stores, get_trade_area_meta, DEMAND_PROFILE_MAP, _num,
    per_store_table, calibrated_sales, BENCHMARK_MONTHLY_SALES, DEFAULT_BENCHMARK, MIN_STORES,
)

# 업종별 비용 구조 (매출 대비 비율) + 초기 투자 기준값(만원)
#   재료비: 원두·식자재·제품 원가 / 인건비: 아르바이트·직원 (사장 본인 제외)
#   공과금: 전기·가스·수도 / 배달: 배달앱 수수료·배달대행비 (매출 전체 대비 평균)
#   표준평수: 이 업종 가게의 흔한 크기 / 인테리어_평당: 평당 인테리어비 / 설비: 주방·기계 등
COST_PROFILES = {
    "커피-음료":     {"재료비": 0.30, "인건비": 0.20, "공과금": 0.04, "배달": 0.02, "표준평수": 15, "인테리어_평당": 200, "설비": 3000},
    "한식음식점":    {"재료비": 0.36, "인건비": 0.25, "공과금": 0.05, "배달": 0.03, "표준평수": 25, "인테리어_평당": 180, "설비": 2500},
    "중식음식점":    {"재료비": 0.33, "인건비": 0.25, "공과금": 0.05, "배달": 0.06, "표준평수": 25, "인테리어_평당": 180, "설비": 2500},
    "일식음식점":    {"재료비": 0.38, "인건비": 0.25, "공과금": 0.05, "배달": 0.02, "표준평수": 20, "인테리어_평당": 200, "설비": 2500},
    "양식음식점":    {"재료비": 0.35, "인건비": 0.25, "공과금": 0.05, "배달": 0.03, "표준평수": 25, "인테리어_평당": 200, "설비": 2500},
    "분식전문점":    {"재료비": 0.33, "인건비": 0.20, "공과금": 0.05, "배달": 0.08, "표준평수": 12, "인테리어_평당": 150, "설비": 1500},
    "패스트푸드점":  {"재료비": 0.38, "인건비": 0.20, "공과금": 0.05, "배달": 0.08, "표준평수": 20, "인테리어_평당": 180, "설비": 3000},
    "치킨전문점":    {"재료비": 0.42, "인건비": 0.15, "공과금": 0.04, "배달": 0.12, "표준평수": 12, "인테리어_평당": 150, "설비": 2000},
    "제과점":        {"재료비": 0.35, "인건비": 0.22, "공과금": 0.06, "배달": 0.02, "표준평수": 15, "인테리어_평당": 200, "설비": 4000},
    "호프-간이주점": {"재료비": 0.33, "인건비": 0.20, "공과금": 0.05, "배달": 0.02, "표준평수": 20, "인테리어_평당": 180, "설비": 2000},
    "미용실":        {"재료비": 0.10, "인건비": 0.35, "공과금": 0.03, "배달": 0.00, "표준평수": 15, "인테리어_평당": 150, "설비": 1500},
    "네일숍":        {"재료비": 0.10, "인건비": 0.30, "공과금": 0.02, "배달": 0.00, "표준평수": 8,  "인테리어_평당": 150, "설비": 500},
    "피부관리실":    {"재료비": 0.10, "인건비": 0.30, "공과금": 0.03, "배달": 0.00, "표준평수": 15, "인테리어_평당": 150, "설비": 1500},
    "편의점":        {"재료비": 0.72, "인건비": 0.10, "공과금": 0.03, "배달": 0.00, "표준평수": 20, "인테리어_평당": 100, "설비": 0},
    "일반교습학원":  {"재료비": 0.05, "인건비": 0.40, "공과금": 0.03, "배달": 0.00, "표준평수": 30, "인테리어_평당": 120, "설비": 1000},
}
DEFAULT_COST = {"재료비": 0.35, "인건비": 0.22, "공과금": 0.04, "배달": 0.02, "표준평수": 15, "인테리어_평당": 150, "설비": 1500}

CARD_FEE = 0.015          # 카드 결제 수수료
SUPPLIES = 0.03           # 소모품·포장재 등
FIXED_OTHER = 50          # 마케팅·보험·세무기장·통신 등 월 고정비(만원)
MAINT_PER_PYEONG = 1.0    # 관리비를 모를 때 평당 월 관리비(만원)
SCENARIOS = {"보수": 0.7, "보통": 1.0, "낙관": 1.2}
DESTINATION_BUSINESSES = {"미용실", "네일숍", "피부관리실", "일반교습학원"}  # 일부러 찾아오는 업종 (층수 영향 작음)


def floor_factor(floor, svc):
    """1층=1.0 기준. 지나가다 들르는 손님이 줄어드는 만큼 매출을 깎는다."""
    if floor is None or floor == 1:
        f = 1.0
    elif floor == 2:
        f = 0.75
    elif floor >= 3:
        f = 0.6
    else:  # 지하
        f = 0.65
    if svc in DESTINATION_BUSINESSES:
        f = 1 - (1 - f) / 2
    return f


def size_factor(area_pyeong, svc):
    """표준 크기보다 넓으면 매출이 늘지만, 넓이에 정비례하지는 않는다(제곱근)."""
    std = COST_PROFILES.get(svc, DEFAULT_COST)["표준평수"]
    if not area_pyeong or area_pyeong <= 0:
        return 1.0
    return max(0.6, min(1.5, (area_pyeong / std) ** 0.5))


def compute_profit(svc, base_monthly_sales, avg_ticket, area_pyeong, monthly_rent, deposit=0,
                   maintenance=None, premium=0, interior_total=None, open_days_per_month=26,
                   scenario_sales=None):
    """
    순수 계산 함수 (데이터 조회 없음). 금액 단위는 모두 '만원'.
    base_monthly_sales: 보정까지 끝난 '보통' 시나리오 월매출
    avg_ticket: 객단가(원)
    """
    cp = COST_PROFILES.get(svc, DEFAULT_COST)
    if maintenance is None:
        maintenance = round(area_pyeong * MAINT_PER_PYEONG)
    var_ratio = cp["재료비"] + cp["인건비"] + cp["공과금"] + cp["배달"] + CARD_FEE + SUPPLIES
    fixed = monthly_rent + maintenance + FIXED_OTHER

    scenarios = {}
    for name, mult in SCENARIOS.items():
        # 예측 모델이 준 범위가 있으면 그걸 쓰고, 없으면 평균의 70%/100%/120%
        rev = scenario_sales[name] if scenario_sales else base_monthly_sales * mult
        costs = {
            "재료비": rev * cp["재료비"],
            "인건비(사장 제외)": rev * cp["인건비"],
            "월세": monthly_rent,
            "관리비": maintenance,
            "공과금": rev * cp["공과금"],
            "배달수수료": rev * cp["배달"],
            "카드수수료": rev * CARD_FEE,
            "소모품·기타": rev * SUPPLIES + FIXED_OTHER,
        }
        total_cost = sum(costs.values())
        profit = rev - total_cost
        scenarios[name] = {
            "월매출_만원": round(rev),
            "월비용_만원": round(total_cost),
            "월순이익_만원": round(profit),
            "이익률": round(profit / rev, 3) if rev else None,
            "비용_내역_만원": {k: round(v) for k, v in costs.items() if round(v)},
        }

    bep_sales = fixed / (1 - var_ratio) if var_ratio < 1 else None
    bep_daily_customers = None
    if bep_sales and avg_ticket:
        bep_daily_customers = round(bep_sales * 10000 / avg_ticket / open_days_per_month)

    if interior_total is None:
        interior_total = round(area_pyeong * cp["인테리어_평당"] + cp["설비"])
    sunk = premium + interior_total                # 돌려받지 못하는 돈
    working_capital = round(fixed * 2)             # 초반 2개월 버틸 운영자금
    total_invest = deposit + sunk + working_capital

    base = scenarios["보통"]
    payback = round(sunk / base["월순이익_만원"], 1) if base["월순이익_만원"] > 0 else None
    rent_ratio = round((monthly_rent + maintenance) / base["월매출_만원"], 3) if base["월매출_만원"] else None

    if base["월순이익_만원"] <= 0:
        verdict = "적자 위험 - 평균 수준 매출로는 비용을 감당하기 어려움"
    elif scenarios["보수"]["월순이익_만원"] <= 0 or (rent_ratio and rent_ratio > 0.15):
        verdict = "빠듯함 - 평균은 남지만 매출이 조금만 줄어도 적자, 또는 임대료 부담이 큼"
    elif base["월순이익_만원"] < 300:
        verdict = "가능하지만 사장 인건비 수준 - 직접 일하는 월급 정도가 남음"
    else:
        verdict = "해볼 만함 - 평균 매출 기준으로 여유가 있음"

    return {
        "판정": verdict,
        "시나리오": scenarios,
        "손익분기_월매출_만원": round(bep_sales) if bep_sales else None,
        "손익분기_하루_손님수": bep_daily_customers,
        "임대료_매출_비율": rent_ratio,
        "임대료_적정선": "매출의 10~15% 이하",
        "초기투자": {
            "보증금(나중에 돌려받음)": deposit,
            "권리금": premium,
            "인테리어·설비(추정)": interior_total,
            "초기_운영자금(2개월 고정비)": working_capital,
            "합계": total_invest,
        },
        "투자금_회수기간_개월(보통, 보증금 제외)": payback,
        "가정값": {
            "재료비율": cp["재료비"], "인건비율(사장 제외)": cp["인건비"], "공과금율": cp["공과금"],
            "배달수수료율": cp["배달"], "카드수수료율": CARD_FEE, "소모품율": SUPPLIES,
            "기타_고정비_만원": FIXED_OTHER, "관리비_만원": maintenance,
            "한달_영업일": open_days_per_month,
        },
    }


def _default_targets(svc):
    """업종의 주 고객 연령대·영업시간대 (수요 특성표에서 가중치가 높은 것)."""
    prof = CATEGORY_DEMAND_PROFILE.get(DEMAND_PROFILE_MAP.get(svc) or "")
    if not prof:
        return None, None
    ages = [k for k, v in prof["age"].items() if v >= 0.8] or None
    hours = [k for k, v in prof["time"].items() if v >= 0.7] or None
    return ages, hours


def predict_listing_profit(trdar_cd, business, area_pyeong, monthly_rent, deposit=0, maintenance=None,
                           premium=0, floor=1, interior_total=None, open_days_per_week=6):
    """상권코드 + 매물 조건 -> 수익 예측. 데이터 조회 포함."""
    std_ym = find_latest_quarter() or "20241"
    sales, stores, sales_q, msg = _load_sales_and_stores(std_ym)
    if not sales:
        return {"error": "서울시 추정매출 데이터를 불러오지 못했습니다 (" + str(msg) + ")"}

    names = {r.get("SVC_INDUTY_CD_NM") for r in sales if r.get("SVC_INDUTY_CD_NM")}
    svc = resolve_business(business, names)
    if not svc:
        return {"error": "'" + str(business) + "' 업종을 서울시 업종 목록에서 찾지 못했습니다.",
                "선택_가능한_업종": sorted(names)}

    meta = get_trade_area_meta()
    trdar_cd = str(trdar_cd)
    gu = (meta.get(trdar_cd) or {}).get("자치구")
    table, seoul_median = per_store_table(svc, sales, stores)

    def _median(vals):
        vals = sorted(v for v in vals if v)
        return vals[len(vals) // 2] if vals else None

    # 이 상권의 '서울 평균 대비 매출 지수'. 점포가 너무 적으면 같은 구 상권들의 중앙값을 쓴다.
    here = table.get(trdar_cd) or {}
    if here.get("raw") and (here.get("n") or 0) >= MIN_STORES:
        raw = here["raw"]
        source = "이 상권의 같은 업종 점포 평균 (점포 " + str(int(here["n"])) + "곳)"
    else:
        same_gu = [v["raw"] for cd, v in table.items()
                   if gu and (meta.get(cd) or {}).get("자치구") == gu and (v["n"] or 0) >= MIN_STORES]
        if same_gu:
            raw = _median(same_gu)
            source = gu + " 같은 업종 상권들의 중앙값 (이 상권은 같은 업종 점포가 " + str(MIN_STORES) + "곳 미만)"
        else:
            raw = seoul_median
            source = "서울 전체 같은 업종 중앙값"
    idx, base_raw = calibrated_sales(svc, raw, seoul_median)
    ticket = here.get("ticket") or _median(v["ticket"] for v in table.values())

    ff, sf = floor_factor(floor, svc), size_factor(area_pyeong, svc)
    base_sales = base_raw * ff * sf if base_raw else None
    open_days = round(open_days_per_week * 30 / 7)

    # 방법 2: 관심고객 기반 검산
    visitors_check = None
    flpop_rows, _, _ = load_quarter_table("VwsmTrdarFlpopQq", std_ym)
    fl = [r for r in flpop_rows if str(r.get("TRDAR_CD")) == trdar_cd]
    if fl and ticket:
        ages, hours = _default_targets(svc)
        tc = estimate_target_customers(summarize_population(fl), summarize_detailed_flpop(fl[0]),
                                       target_ages=ages, business_hours=hours)
        normal = list((tc.get("하루_예상_방문객_시나리오") or {}).values())
        if len(normal) == 3:
            daily = normal[1] * ff
            visitors_check = {
                "하루_예상_방문객(전환율 2% 가정)": round(daily),
                "객단가_원": round(ticket),
                "방문객_기준_월매출_만원": round(daily * ticket * open_days / 10000),
                "참고": "상권 평균 매출과 크게 다르면 전환율 가정이나 상권 범위를 의심해 볼 것",
            }
            if not base_sales:
                base_sales, source = visitors_check["방문객_기준_월매출_만원"], "관심고객 × 객단가 추정 (매출 데이터 없음)"

    # 예측 모델: 같은 업종 점포가 하나 늘어난 상황(내가 진입)의 점포당 월매출과 80% 범위
    scenario_sales, model_pred, scenario_meaning = None, None, "보수=평균의 70%, 보통=평균, 낙관=평균의 120% (임의 가정)"
    try:
        from sales_model import predict_entrant, model_summary_for_user
        pr = (predict_entrant(svc, [trdar_cd]) or {}).get(trdar_cd)
    except Exception as e:
        pr, model_error = None, str(e)
    if pr:
        scenario_sales = {"보수": pr["하위10%"] * ff * sf, "보통": pr["중앙"] * ff * sf, "낙관": pr["상위10%"] * ff * sf}
        base_sales = scenario_sales["보통"]
        source = ("매출 예측: " + pr["사용모델"] + " (같은 업종 " + str(pr["점포수_진입전"]) + "곳 → 내가 들어가면 "
                  + str(pr["점포수_진입후"]) + "곳 기준, 신뢰도 " + pr["신뢰도"] + ")")
        model_pred = {"점포당_월매출_예측_만원(층·면적 보정 전)": {k: pr[k] for k in ("하위10%", "중앙", "상위10%")},
                      "모델_검증": model_summary_for_user()}
        scenario_meaning = ("보수=예측 하위 10%, 보통=예측 중앙값, 낙관=예측 상위 10% (실제 서울시 데이터로 검증한 분기간 변동폭, "
                            "실제 값이 이 범위에 들어갈 확률 약 80%)")

    if not base_sales:
        return {"error": "이 업종의 매출 데이터를 찾지 못해 수익을 계산할 수 없습니다.", "업종": svc}

    result = compute_profit(svc, base_sales, ticket, area_pyeong, monthly_rent, deposit, maintenance,
                            premium, interior_total, open_days, scenario_sales=scenario_sales)
    result["시나리오_의미"] = scenario_meaning
    result["매출_예측_모델"] = model_pred
    result.update({
        "업종": svc,
        "매출_기준": source,
        "매출_기준_분기": sales_q,
        "매출_보정": {
            "서울_점포당_월매출_중앙값_만원": round(seoul_median) if seoul_median else None,
            "서울평균_대비_매출지수": idx,
            "동네_평균_점포_월매출_만원": round(base_raw) if base_raw else None,
            "층수_보정(가정값)": round(ff, 2), "면적_보정(가정값)": round(sf, 2),
            "표준평수": COST_PROFILES.get(svc, DEFAULT_COST)["표준평수"],
        },
        "검산_관심고객_기준": visitors_check,
        "매물": {"평수": area_pyeong, "층": floor, "보증금_만원": deposit, "월세_만원": monthly_rent,
               "관리비_만원": result["가정값"]["관리비_만원"], "권리금_만원": premium},
        "주의": "서울시 추정매출(카드결제 보정값)을 전체 매출로 보고 계산함. 세금(부가세·소득세)과 사장 본인 인건비는 빠져 있고, "
               "가맹점이면 로열티·물류비가 추가될 수 있음",
    })
    return result
