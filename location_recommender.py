"""
업종 → 위치 추천 모듈 (창업자용)
================================
"카페를 하고 싶은데 서울 어디가 좋을까?"에 답한다.

서울시가 지정한 상권(약 1,650곳) 전체를 대상으로, 원하는 업종에 대해 아래 세 가지를
비교해서 점수(0~100)를 매기고 순위를 낸다.

  1) 점포당 추정매출   - 이 상권에서 이 업종 가게 하나가 평균적으로 버는 돈 (높을수록 좋음)
                         → 매출 총액이 아니라 '가게 하나당'이라서 경쟁 정도가 자연히 반영된다
  2) 관심 유동인구     - 하루 유동인구 × 이 업종 주 고객층과의 궁합 (높을수록 좋음)
  3) 폐업률            - 이 상권에서 이 업종이 문 닫는 비율 (낮을수록 좋음)

점수 = 100 × (0.45 × 매출 순위 + 0.30 × 유동인구 순위 + 0.25 × 생존 순위)
(순위는 0~1 사이 백분위. 가중치는 CONFIG처럼 아래 WEIGHTS에서 바꿀 수 있다)

데이터 출처: 서울시 상권분석서비스 (추정매출·점포·길단위인구·상권영역)
"""

from foot_traffic import load_quarter_table, find_latest_quarter, previous_quarter
from compare_trade_area import summarize_population, compute_demand_score, SEOUL_GU_LAWD_CD

WEIGHTS = {"sales": 0.45, "traffic": 0.30, "survival": 0.25}
MIN_STORES = 3  # 점포가 이보다 적은 상권은 평균이 불안정해서 순위에서 뺀다

# 사람들이 흔히 말하는 업종 이름 → 서울시 서비스업종 이름
BUSINESS_ALIASES = {
    "카페": "커피-음료", "커피": "커피-음료", "음료": "커피-음료", "디저트카페": "커피-음료",
    "술집": "호프-간이주점", "호프": "호프-간이주점", "주점": "호프-간이주점", "이자카야": "호프-간이주점",
    "빵집": "제과점", "베이커리": "제과점", "제과": "제과점",
    "치킨": "치킨전문점", "분식": "분식전문점", "떡볶이": "분식전문점", "김밥": "분식전문점",
    "햄버거": "패스트푸드점", "패스트푸드": "패스트푸드점",
    "한식": "한식음식점", "고깃집": "한식음식점", "백반": "한식음식점", "국밥": "한식음식점",
    "중식": "중식음식점", "중국집": "중식음식점", "일식": "일식음식점", "초밥": "일식음식점",
    "라멘": "일식음식점", "양식": "양식음식점", "파스타": "양식음식점",
    "미용실": "미용실", "헤어": "미용실", "네일": "네일숍", "피부관리": "피부관리실",
    "편의점": "편의점", "슈퍼": "슈퍼마켓", "반찬": "반찬가게", "꽃집": "화초",
    "옷가게": "일반의류", "의류": "일반의류", "화장품": "화장품", "안경": "안경",
    "학원": "일반교습학원", "노래방": "노래방", "PC방": "PC방", "피시방": "PC방",
    "세탁소": "세탁소", "부동산": "부동산중개업", "약국": "의약품", "애견": "애완동물",
    "반려동물": "애완동물", "헬스": "스포츠클럽", "필라테스": "스포츠클럽",
}

# 서울시 서비스업종 이름 → 기존 수요 특성표(compare_trade_area.CATEGORY_DEMAND_PROFILE) 키
DEMAND_PROFILE_MAP = {
    "커피-음료": "비알코올", "호프-간이주점": "주점",
    "한식음식점": "한식", "중식음식점": "한식", "일식음식점": "한식", "양식음식점": "한식",
    "분식전문점": "기타 간이", "패스트푸드점": "기타 간이", "치킨전문점": "기타 간이", "제과점": "기타 간이",
    "미용실": "이용·미용", "네일숍": "이용·미용", "피부관리실": "이용·미용",
    "편의점": "종합 소매", "슈퍼마켓": "종합 소매",
    "반찬가게": "식료품 소매", "육류판매": "식료품 소매", "청과상": "식료품 소매", "수산물판매": "식료품 소매",
    "일반의류": "섬유·의복·신발 소매", "신발": "섬유·의복·신발 소매",
    "화장품": "의약·화장품 소매", "의약품": "의약·화장품 소매",
    "일반교습학원": "기타 교육", "외국어학원": "기타 교육", "예술학원": "기타 교육",
    "일반의원": "의원", "치과의원": "의원", "한의원": "의원",
    "부동산중개업": "부동산 서비스", "세무사사무소": "회계·세무", "회계사사무소": "회계·세무",
    "노래방": "유원지·오락", "PC방": "유원지·오락",
}

SALES_FIELDS = ["STDR_YYQU_CD", "TRDAR_CD", "TRDAR_CD_NM", "SVC_INDUTY_CD_NM",
                "THSMON_SELNG_AMT", "THSMON_SELNG_CO"]
STORE_FIELDS = ["STDR_YYQU_CD", "TRDAR_CD", "SVC_INDUTY_CD_NM", "STOR_CO",
                "SIMILR_INDUTY_STOR_CO", "FRC_STOR_CO", "CLSBIZ_RT", "OPBIZ_RT"]

_GU_BY_CODE = {v: k for k, v in SEOUL_GU_LAWD_CD.items()}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _percentile_ranks(values):
    """값 리스트 -> 0~1 백분위(가장 작은 값 0, 가장 큰 값 1). None은 0.5로 둔다."""
    valid = sorted(v for v in values if v is not None)
    n = len(valid)
    out = []
    for v in values:
        if v is None or n <= 1:
            out.append(0.5)
            continue
        below = sum(1 for x in valid if x < v)
        out.append(below / (n - 1))
    return out


def resolve_business(business, available_names):
    """사용자가 말한 업종 -> 서울시 서비스업종 이름. 못 찾으면 None."""
    b = (business or "").replace(" ", "")
    if b in available_names:
        return b
    for alias, name in BUSINESS_ALIASES.items():
        if alias in b and name in available_names:
            return name
    for name in available_names:
        if b and (b in name or name in b):
            return name
    return None


def get_trade_area_meta():
    """상권코드 -> {상권명, 상권구분, 자치구, 행정동, 위도, 경도}"""
    rows, _, _ = load_quarter_table("TbgisTrdarRelm", "")
    meta = {}
    try:
        from pyproj import Transformer
        to_latlng = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
    except Exception:
        to_latlng = None
    for r in rows:
        gu = r.get("SIGNGU_CD_NM") or _GU_BY_CODE.get(str(r.get("SIGNGU_CD") or "")[:5])
        lat = lng = None
        x, y = _num(r.get("XCNTS_VALUE")), _num(r.get("YDNTS_VALUE"))
        if to_latlng and x and y:
            lng, lat = to_latlng.transform(x, y)
        meta[str(r.get("TRDAR_CD"))] = {
            "상권명": r.get("TRDAR_CD_NM"),
            "상권구분": r.get("TRDAR_SE_CD_NM"),
            "자치구": gu,
            "행정동": r.get("ADSTRD_CD_NM"),
            "lat": round(lat, 6) if lat else None,
            "lng": round(lng, 6) if lng else None,
        }
    return meta


def _load_sales_and_stores(std_ym):
    sales, sales_q, msg = load_quarter_table("VwsmTrdarSelngQq", std_ym, SALES_FIELDS)
    if not sales:
        # 매출 데이터는 유동인구보다 늦게 올라오는 경우가 있어 직전 분기로 한 번 더
        sales, sales_q, msg = load_quarter_table("VwsmTrdarSelngQq", previous_quarter(std_ym), SALES_FIELDS)
    stores, _, _ = load_quarter_table("VwsmTrdhlStorQq", sales_q or std_ym, STORE_FIELDS)
    return sales, stores, sales_q, msg


def list_business_types():
    """추천에 쓸 수 있는 서울시 서비스업종 이름 목록."""
    std_ym = find_latest_quarter() or "20241"
    sales, _, _, _ = _load_sales_and_stores(std_ym)
    return sorted({r.get("SVC_INDUTY_CD_NM") for r in sales if r.get("SVC_INDUTY_CD_NM")})


def recommend_locations(business, gu=None, top_n=10, target_ages=None):
    """
    업종 하나에 대해 서울 상권 순위를 매긴다.
    business: '카페', '커피-음료', '치킨' 등
    gu: '마포구'처럼 자치구로 범위를 좁히고 싶을 때 (여러 개면 쉼표로)
    target_ages: 주 고객 연령대를 직접 지정하고 싶을 때 (예: ['20대','30대'])
    """
    std_ym = find_latest_quarter() or "20241"
    sales, stores, sales_q, msg = _load_sales_and_stores(std_ym)
    if not sales:
        return {"error": "서울시 추정매출 데이터를 불러오지 못했습니다 (" + str(msg) + ")"}

    names = {r.get("SVC_INDUTY_CD_NM") for r in sales if r.get("SVC_INDUTY_CD_NM")}
    svc = resolve_business(business, names)
    if not svc:
        return {"error": "'" + str(business) + "' 업종을 서울시 업종 목록에서 찾지 못했습니다.",
                "선택_가능한_업종": sorted(names)}

    flpop_rows, flpop_q, _ = load_quarter_table("VwsmTrdarFlpopQq", std_ym)
    flpop_by_cd = {str(r.get("TRDAR_CD")): r for r in flpop_rows}
    store_by_cd = {str(r.get("TRDAR_CD")): r for r in stores if r.get("SVC_INDUTY_CD_NM") == svc}
    meta = get_trade_area_meta()
    profile_key = DEMAND_PROFILE_MAP.get(svc)

    gu_filter = [g.strip() for g in (gu or "").replace("/", ",").split(",") if g.strip()]

    candidates = []
    for r in sales:
        if r.get("SVC_INDUTY_CD_NM") != svc:
            continue
        cd = str(r.get("TRDAR_CD"))
        info = meta.get(cd, {})
        if gu_filter and not any(g.replace("구", "") in (info.get("자치구") or "") for g in gu_filter):
            continue

        amt = _num(r.get("THSMON_SELNG_AMT"))
        st = store_by_cd.get(cd) or {}
        n_store = _num(st.get("STOR_CO")) or _num(st.get("SIMILR_INDUTY_STOR_CO"))
        if not amt or not n_store or n_store < MIN_STORES:
            continue

        fl = flpop_by_cd.get(cd)
        pop = {k: _num(v) for k, v in summarize_population([fl]).items()} if fl else {}
        daily = (_num(pop.get("총_유동인구")) or 0) / 91 if pop else None
        fit = None
        if pop and profile_key:
            fit = compute_demand_score(profile_key, pop)
        if pop and target_ages:
            age_total = sum(_num(pop.get(a)) or 0 for a in ["10대", "20대", "30대", "40대", "50대", "60대_이상"])
            share = (sum(_num(pop.get(a)) or 0 for a in target_ages) / age_total) if age_total else None
            fit = share if fit is None or share is None else (fit + share) / 2
        interested = daily * fit if (daily and fit is not None) else daily

        candidates.append({
            "trdar_cd": cd,
            "상권명": info.get("상권명") or r.get("TRDAR_CD_NM"),
            "상권구분": info.get("상권구분"),
            "자치구": info.get("자치구"),
            "행정동": info.get("행정동"),
            "lat": info.get("lat"), "lng": info.get("lng"),
            "점포수": int(n_store),
            "프랜차이즈_점포수": int(_num(st.get("FRC_STOR_CO")) or 0),
            "점포당_추정매출_만원": round(amt / n_store / 10000),
            "업종_전체_추정매출_만원": round(amt / 10000),
            "폐업률": _num(st.get("CLSBIZ_RT")),
            "개업률": _num(st.get("OPBIZ_RT")),
            "하루_유동인구": round(daily) if daily else None,
            "하루_관심_유동인구": round(interested) if interested else None,
        })

    if not candidates:
        where = (" (" + ", ".join(gu_filter) + ")") if gu_filter else ""
        return {"error": "조건에 맞는 상권이 없습니다" + where + ". 점포가 " + str(MIN_STORES)
                         + "곳 이상인 상권만 비교하기 때문에, 범위를 넓혀 보세요.", "업종": svc}

    pr_sales = _percentile_ranks([c["점포당_추정매출_만원"] for c in candidates])
    pr_traffic = _percentile_ranks([c["하루_관심_유동인구"] for c in candidates])
    pr_closure = _percentile_ranks([c["폐업률"] for c in candidates])
    for c, s, t, cl in zip(candidates, pr_sales, pr_traffic, pr_closure):
        c["점수"] = round(100 * (WEIGHTS["sales"] * s + WEIGHTS["traffic"] * t + WEIGHTS["survival"] * (1 - cl)), 1)
        c["점수_구성"] = {
            "점포당매출": "상위 " + str(max(1, round((1 - s) * 100))) + "%",
            "관심유동인구": "상위 " + str(max(1, round((1 - t) * 100))) + "%",
            "폐업률_낮은순": "상위 " + str(max(1, round(cl * 100))) + "%",
        }

    candidates.sort(key=lambda c: c["점수"], reverse=True)
    per_store = sorted(c["점포당_추정매출_만원"] for c in candidates)
    median = per_store[len(per_store) // 2]

    return {
        "업종": svc,
        "요청한_업종": business,
        "범위": ", ".join(gu_filter) if gu_filter else "서울 전체",
        "매출_기준_분기": sales_q,
        "유동인구_기준_분기": flpop_q,
        "비교한_상권_수": len(candidates),
        "업종_점포당_추정매출_중앙값_만원": median,
        "추천_상권": candidates[:top_n],
        "점수_계산법": "점포당 추정매출 45% + 관심 유동인구 30% + 낮은 폐업률 25% (각각 비교 상권 중 백분위)",
        "참고": "추정매출은 카드매출 기반 추정치이며, 점포 " + str(MIN_STORES) + "곳 미만 상권은 제외함",
    }
