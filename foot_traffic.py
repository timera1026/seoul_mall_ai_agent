"""
인적 동선(유동인구) 정밀 분석 모듈
==================================
기존 compare_trade_area.py는 '상권 전체 유동인구 합계'를 주로 썼다.
이 파일은 같은 상권을 더 잘게 쪼개서 본다.

  1) 요일별 · 성별×연령별 유동인구     (서울시 상권분석서비스 - 길단위인구)
  2) 직장인구 (낮에 일하러 오는 사람)  (서울시 상권분석서비스 - 직장인구)
  3) 집객시설 (지하철역·학교·관공서 등) (서울시 상권분석서비스 - 집객시설)
  4) 지하철역 시간대별 승·하차 인원     (서울시 지하철 호선별 역별 시간대별 승하차)
  5) 관심고객 수 추정                  (위 데이터를 업종의 주 고객층에 맞게 걸러냄)

서울시 API는 필드 이름이 바뀌는 일이 종종 있어서, 여기 함수들은 필드명을 하나로
고정하지 않고 '이름에 이런 글자가 들어 있으면 그 값'으로 찾아내도록 만들었다.
값을 못 찾으면 에러로 멈추지 않고 None을 돌려준다.
"""

import os
import re
import calendar
from datetime import date

import requests

SEOUL_API_BASE = "http://openapi.seoul.go.kr:8088"


def _seoul_key():
    # import 시점이 아니라 호출 시점에 읽어야 Streamlit secrets가 늦게 들어와도 동작한다.
    return os.getenv("SEOUL_API_KEY")


def _to_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seoul_get(service, start, end, *params):
    """서울 열린데이터광장 API 한 페이지 호출. (rows, 전체건수, 결과메시지) 반환."""
    key = _seoul_key()
    if not key:
        return [], 0, "SEOUL_API_KEY 없음"
    parts = [SEOUL_API_BASE, key, "json", service, str(start), str(end)]
    parts += [str(p) for p in params if p not in (None, "")]
    url = "/".join(parts) + "/"
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        payload = res.json()
    except Exception as e:
        return [], 0, "호출 실패: " + str(e)

    body = payload.get(service)
    if not body:
        # 데이터가 없을 때 서울시 API는 {"RESULT": {"CODE": "INFO-200", ...}} 형태로 준다
        msg = (payload.get("RESULT") or {}).get("MESSAGE") or "데이터 없음"
        return [], 0, msg
    rows = body.get("row", []) or []
    total = int(body.get("list_total_count", 0) or 0)
    msg = (body.get("RESULT") or {}).get("MESSAGE", "정상")
    return rows, total, msg


# ---------------------------------------------------------------------------
# 분기별 '서울 전체 상권 표' 불러오기 (+ 메모리 캐시)
#   서울시 API는 상권코드로 바로 검색이 안 돼서, 표 전체를 받은 뒤 원하는 상권을
#   골라야 한다. 한 페이지(1000행)씩 순서대로 받으면 느리므로 여러 페이지를 동시에
#   받고, 한 번 받은 표는 12시간 동안 기억해 두고 다시 쓴다.
#   또, 일부 API(직장인구·집객시설)는 분기 조건을 줘도 여러 분기가 섞여 오는
#   경우가 있어서, 행의 STDR_YYQU_CD(기준 분기)를 직접 확인해 원하는 분기만 남긴다.
# ---------------------------------------------------------------------------

import time
import threading

_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()
TABLE_TTL_SEC = 12 * 3600
MAX_TABLE_ROWS = 200000


def load_quarter_table(service, std_ym, keep_fields=None):
    """
    service 표에서 std_ym 분기 행들을 모두 가져온다.
    반환: (rows, 실제_사용된_분기, 메시지)
      - 원하는 분기가 표에 없으면, 표 안의 가장 최근 분기를 대신 쓴다.
    keep_fields를 주면 그 필드만 남겨 메모리를 아낀다.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache_key = (service, std_ym, tuple(keep_fields) if keep_fields else None)
    with _TABLE_LOCK:
        hit = _TABLE_CACHE.get(cache_key)
    if hit and time.time() - hit[0] < TABLE_TTL_SEC:
        return hit[1], hit[2], "캐시"

    first, total, msg = _seoul_get(service, 1, 1000, std_ym)
    if not first:
        return [], None, msg
    if total > MAX_TABLE_ROWS:
        return [], None, "표가 너무 큼(" + str(total) + "행) - 전체 조회를 건너뜀"

    pages = [(start, start + 999) for start in range(1001, total + 1, 1000)]
    all_rows = list(first)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows, _, _ in ex.map(lambda se: _seoul_get(service, se[0], se[1], std_ym), pages):
            all_rows.extend(rows)

    used = std_ym
    if all_rows and "STDR_YYQU_CD" in all_rows[0]:
        exact = [r for r in all_rows if str(r.get("STDR_YYQU_CD")) == str(std_ym)]
        if not exact:
            used = max(str(r.get("STDR_YYQU_CD")) for r in all_rows)
            exact = [r for r in all_rows if str(r.get("STDR_YYQU_CD")) == used]
        all_rows = exact

    if keep_fields:
        all_rows = [{k: r.get(k) for k in keep_fields} for r in all_rows]

    with _TABLE_LOCK:
        _TABLE_CACHE[cache_key] = (time.time(), all_rows, used)
    return all_rows, used, "정상"


def _fetch_rows_for_trdar(service, trdar_cd, std_ym):
    """std_ym 분기 표에서 trdar_cd 상권 행만 골라낸다."""
    rows, _, _ = load_quarter_table(service, std_ym)
    return [r for r in rows if str(r.get("TRDAR_CD")) == str(trdar_cd)]


# ---------------------------------------------------------------------------
# 0) 최신 분기 자동 찾기
#    예전 코드는 분석 기준 분기가 "20241"(2024년 1분기)로 고정돼 있어서,
#    시간이 지날수록 오래된 데이터로 분석하는 문제가 있었다.
# ---------------------------------------------------------------------------

def _recent_quarters(n=8):
    today = date.today()
    year, quarter = today.year, (today.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        out.append("{:04d}{:d}".format(year, quarter))
        quarter -= 1
        if quarter == 0:
            quarter, year = 4, year - 1
    return out  # 최신 -> 과거 순


def previous_quarter(std_ym):
    year, quarter = int(std_ym[:4]), int(std_ym[4:])
    quarter -= 1
    if quarter == 0:
        quarter, year = 4, year - 1
    return "{:04d}{:d}".format(year, quarter)


_LATEST_QUARTER_CACHE = {}


def find_latest_quarter(service="VwsmTrdarFlpopQq"):
    """해당 서비스에 실제로 데이터가 올라와 있는 가장 최근 분기 코드(예: '20262')."""
    # 성공한 결과만 기억한다 (실패를 기억하면 API가 잠깐 끊겼을 때 계속 실패로 남음)
    if service in _LATEST_QUARTER_CACHE:
        return _LATEST_QUARTER_CACHE[service]
    for std_ym in _recent_quarters(8):
        rows, _, _ = _seoul_get(service, 1, 1, std_ym)
        if rows:
            _LATEST_QUARTER_CACHE[service] = std_ym
            return std_ym
    return None


# ---------------------------------------------------------------------------
# 1) 요일별 · 성별×연령별 유동인구
# ---------------------------------------------------------------------------

DAY_FIELDS = {
    "월": "MON_FLPOP_CO", "화": "TUES_FLPOP_CO", "수": "WED_FLPOP_CO",
    "목": "THUR_FLPOP_CO", "금": "FRI_FLPOP_CO", "토": "SAT_FLPOP_CO", "일": "SUN_FLPOP_CO",
}
AGE_LABELS = ["10대", "20대", "30대", "40대", "50대", "60대_이상"]


def _find_field(row, *patterns):
    """row의 키 중 모든 패턴(정규식)에 맞는 첫 필드의 숫자값."""
    for k, v in row.items():
        if all(re.search(p, k) for p in patterns):
            return _to_num(v)
    return None


def _day_value(row, day_label):
    exact = _to_num(row.get(DAY_FIELDS[day_label]))
    if exact is not None:
        return exact
    prefix = DAY_FIELDS[day_label].split("_")[0][:3]  # MON, TUE, WED ...
    return _find_field(row, "^" + prefix, "FLPOP")


def _age_pattern(label):
    return "60_ABOVE" if label == "60대_이상" else "_" + label[:2] + "_"


def summarize_detailed_flpop(row):
    """VwsmTrdarFlpopQq 한 행 -> 요일별, 성별×연령별 유동인구 dict."""
    if not row:
        return {}
    by_day = {d: _day_value(row, d) for d in DAY_FIELDS}

    gender_age = {}
    for g_prefix, g_label in (("^MAG", "남"), ("^FAG", "여")):
        for label in AGE_LABELS:
            gender_age[g_label + "_" + label] = _find_field(row, g_prefix, _age_pattern(label), "FLPOP")

    weekday = [v for d, v in by_day.items() if d in "월화수목금" and v]
    weekend = [v for d, v in by_day.items() if d in "토일" and v]
    return {
        "요일별": by_day,
        # 요일별 값은 분기 동안 그 요일(약 13번)을 합친 값이라, 13으로 나눠 '그 요일 하루' 값으로 환산
        "주중_하루평균": round(sum(weekday) / len(weekday) / 13) if weekday else None,
        "주말_하루평균": round(sum(weekend) / len(weekend) / 13) if weekend else None,
        "성별_연령별": gender_age,
    }


# ---------------------------------------------------------------------------
# 2) 직장인구, 3) 집객시설
# ---------------------------------------------------------------------------

def summarize_worker_population(rows):
    if not rows:
        return None
    r = rows[0]
    out = {"총_직장인구": _find_field(r, "^TOT", "WRC_POPLTN")}
    out["남성"] = _find_field(r, "^ML", "WRC_POPLTN")
    out["여성"] = _find_field(r, "^FML", "WRC_POPLTN")
    for label in AGE_LABELS:
        out[label] = _find_field(r, "^AGRDE", _age_pattern(label), "WRC_POPLTN")
    return out


FACILITY_LABELS = {
    "VIATR_FCLTY_CO": "집객시설_수",
    "SUBWAY_STATN_CO": "지하철역_수",
    "BUS_STTN_CO": "버스정류장_수",
    "BUS_TRMINL_CO": "버스터미널_수",
    "RLROAD_STATN_CO": "철도역_수",
    "UNIV_CO": "대학교_수",
    "HGSCHL_CO": "고등학교_수",
    "MSKUL_CO": "중학교_수",
    "ELESCH_CO": "초등학교_수",
    "KNDRGR_CO": "유치원_수",
    "GEHSPT_CO": "종합병원_수",
    "GNRL_HSPTL_CO": "일반병원_수",
    "PBLOFC_CO": "관공서_수",
    "BANK_CO": "은행_수",
    "DRTS_CO": "백화점_수",
    "SUPMK_CO": "슈퍼마켓_수",
    "THEAT_CO": "극장_수",
    "STAYNG_FCLTY_CO": "숙박시설_수",
    "PARMACY_CO": "약국_수",
    "ARPRT_CO": "공항_수",
}


def summarize_facilities(rows):
    if not rows:
        return None
    r = rows[0]
    out = {}
    for field, label in FACILITY_LABELS.items():
        v = _to_num(r.get(field))
        if v is not None:
            out[label] = int(v)
    return out or None


# ---------------------------------------------------------------------------
# 4) 지하철역 시간대별 승·하차
#    하차가 많은 시간 = 그 시간에 이 동네로 '도착'하는 사람이 많다는 뜻.
#    아침 하차가 많으면 직장 밀집지, 저녁 하차가 많으면 주거지 성격.
# ---------------------------------------------------------------------------

_HOUR_WORDS = {
    "MIDNIGHT": 0, "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12,
    "THIRTEEN": 13, "FOURTEEN": 14, "FIFTEEN": 15, "SIXTEEN": 16, "SEVENTEEN": 17,
    "EIGHTEEN": 18, "NINETEEN": 19, "TWENTY": 20, "TWENTY_ONE": 21, "TWENTY_TWO": 22,
    "TWENTY_THREE": 23,
}


def _parse_hour_field(key):
    """필드명에서 (시각, '승차'|'하차')를 뽑는다. 해당 없으면 None."""
    k = key.upper()
    if re.search(r"RIDE|GET_ON|_ON_", k):
        kind = "승차"
    elif re.search(r"ALIGHT|GET_OFF|_OFF_", k):
        kind = "하차"
    else:
        return None
    m = re.search(r"(\d{1,2})", k)
    if m:
        hour = int(m.group(1))
    else:
        word = re.split(r"_(RIDE|ALIGHT|GET)", k)[0]
        if word not in _HOUR_WORDS:
            return None
        hour = _HOUR_WORDS[word]
    if hour > 24:
        return None
    return hour % 24, kind


def _station_name(row):
    for f in ("SUB_STA_NM", "STTN", "STTN_NM", "SBWY_STNS_NM"):
        if row.get(f):
            return str(row[f])
    return ""


def _clean_station(name):
    return re.sub(r"\(.*?\)|역$", "", name.replace(" ", "")).strip()


_SUBWAY_CACHE = {}


def _subway_month_rows(yyyymm):
    if yyyymm in _SUBWAY_CACHE:
        return _SUBWAY_CACHE[yyyymm]
    rows_all = []
    for page in range(3):
        start = page * 1000 + 1
        rows, total, _ = _seoul_get("CardSubwayTime", start, start + 999, yyyymm)
        if not rows:
            break
        rows_all.extend(rows)
        if start + 999 >= total:
            break
    if rows_all:
        _SUBWAY_CACHE[yyyymm] = rows_all
    return rows_all


def get_subway_hourly(station_name, months_back=4):
    """
    역 이름(예: '청량리', '홍대입구역')으로 가장 최근 달의 시간대별 승·하차 인원을 조회한다.
    여러 호선이 지나는 역은 호선별 값을 합산한다. 값은 '하루 평균'으로 환산해서 준다.
    """
    target = _clean_station(station_name)
    if not target:
        return {"station": station_name, "error": "역 이름이 비어 있음"}

    today = date.today()
    y, m = today.year, today.month
    for _ in range(months_back):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        yyyymm = "{:04d}{:02d}".format(y, m)
        rows = _subway_month_rows(yyyymm)
        if not rows:
            continue

        matched = [r for r in rows if _clean_station(_station_name(r)) == target]
        if not matched:
            matched = [r for r in rows if target in _clean_station(_station_name(r))]
        if not matched:
            return {"station": station_name, "month": yyyymm,
                    "error": "이 달 데이터에서 역 이름을 찾지 못함 (1~9호선 등 서울시 관할 노선만 제공)"}

        days = calendar.monthrange(y, m)[1]
        on, off = [0.0] * 24, [0.0] * 24
        lines = set()
        for r in matched:
            for line_field in ("LINE_NUM", "SBWY_ROUT_LN_NM"):
                if r.get(line_field):
                    lines.add(str(r[line_field]))
            for k, v in r.items():
                parsed = _parse_hour_field(k)
                val = _to_num(v)
                if not parsed or val is None:
                    continue
                hour, kind = parsed
                (on if kind == "승차" else off)[hour] += val

        if sum(on) + sum(off) == 0:
            return {"station": station_name, "month": yyyymm,
                    "error": "시간대 필드를 해석하지 못함 (데이터 점검 메뉴로 필드명을 확인하세요)"}

        hourly = {
            "{:02d}시".format(h): {"승차": round(on[h] / days), "하차": round(off[h] / days)}
            for h in range(24) if on[h] or off[h]
        }
        top_off = [h for h in sorted(range(24), key=lambda h: off[h], reverse=True) if off[h] > 0][:3]
        morning_off = sum(off[7:10])
        evening_off = sum(off[17:21])
        if morning_off > evening_off * 1.3:
            character = "출근 도착형 (아침 하차가 많음 → 직장·학교 밀집 지역 성격)"
        elif evening_off > morning_off * 1.3:
            character = "퇴근 귀가형 (저녁 하차가 많음 → 주거지 성격)"
        else:
            character = "혼합형 (아침·저녁 도착이 비슷함)"

        return {
            "station": _station_name(matched[0]),
            "lines": sorted(lines),
            "month": yyyymm,
            "하루평균_승차": round(sum(on) / days),
            "하루평균_하차": round(sum(off) / days),
            "하차_피크_시간": ["{:02d}시".format(h) for h in top_off],
            "역_성격": character,
            "시간대별_하루평균": hourly,
        }

    return {"station": station_name, "error": "최근 " + str(months_back) + "개월 지하철 데이터를 가져오지 못함"}


# ---------------------------------------------------------------------------
# 5) 관심고객 수 추정
# ---------------------------------------------------------------------------

TIME_KEYS = ["시간대_00~06", "시간대_06~11", "시간대_11~14", "시간대_14~17", "시간대_17~21", "시간대_21~24"]
DAYS_IN_QUARTER = 91


def estimate_target_customers(pop_summary, detailed, target_ages=None, target_gender="전체",
                              business_hours=None, open_days=None, capture_rates=(0.01, 0.02, 0.04)):
    """
    관심고객 수 = 하루 유동인구
                 × 주 고객층(연령·성별) 비율
                 × 영업시간대에 지나가는 비율
                 × 영업요일 보정
    잠재 방문객 = 관심고객 × 방문 전환율(낮음/보통/높음 세 가지 시나리오)

    주의: 서울시 길단위 유동인구는 '분기 합계' 값이라 91일로 나눠 하루 값으로 환산한다.
    전환율은 업계 경험치 기반의 가정값이며, 실측값이 아니다.
    """
    if not pop_summary or not pop_summary.get("총_유동인구"):
        return {"error": "유동인구 데이터가 없어 관심고객 수를 계산할 수 없음"}

    total = _to_num(pop_summary.get("총_유동인구")) or 0
    target_ages = target_ages or AGE_LABELS
    business_hours = business_hours or TIME_KEYS
    open_days = open_days or list("월화수목금토일")
    detailed = detailed or {}

    # (1) 주 고객층 비율: 성별×연령 교차값이 있으면 그걸 쓰고, 없으면 연령비율 × 성별비율로 근사
    ga = detailed.get("성별_연령별") or {}
    ga_total = sum(v for v in ga.values() if v)
    genders = ["남", "여"] if target_gender in (None, "", "전체") else [target_gender[0]]
    if ga_total:
        hit = sum(ga.get(g + "_" + a) or 0 for g in genders for a in target_ages)
        segment_share = hit / ga_total
        segment_method = "성별×연령 교차 데이터"
    else:
        age_total = sum(_to_num(pop_summary.get(a)) or 0 for a in AGE_LABELS)
        age_share = (sum(_to_num(pop_summary.get(a)) or 0 for a in target_ages) / age_total) if age_total else 1
        male = _to_num(pop_summary.get("남성_유동인구")) or 0
        female = _to_num(pop_summary.get("여성_유동인구")) or 0
        if len(genders) == 2 or not (male + female):
            g_share = 1
        else:
            g_share = (male if genders[0] == "남" else female) / (male + female)
        segment_share = age_share * g_share
        segment_method = "연령비율×성별비율 근사"

    # (2) 영업시간대 비율
    t_total = sum(_to_num(pop_summary.get(k)) or 0 for k in TIME_KEYS)
    time_share = (sum(_to_num(pop_summary.get(k)) or 0 for k in business_hours) / t_total) if t_total else 1

    # (3) 하루 유동인구 (영업일 기준). 요일 데이터가 있으면 영업요일의 평균을 쓴다.
    by_day = detailed.get("요일별") or {}
    day_vals = [by_day.get(d) for d in open_days if by_day.get(d)]
    all_days = [v for v in by_day.values() if v]
    daily_total = total / DAYS_IN_QUARTER
    if day_vals and all_days:
        # 요일별 값의 상대적 크기로 '영업하는 날'의 하루 유동인구를 보정
        daily_total = daily_total * (sum(day_vals) / len(day_vals)) / (sum(all_days) / len(all_days))

    interested = daily_total * segment_share * time_share
    scenarios = {
        "낮음({:.0%})".format(capture_rates[0]): round(interested * capture_rates[0]),
        "보통({:.0%})".format(capture_rates[1]): round(interested * capture_rates[1]),
        "높음({:.0%})".format(capture_rates[2]): round(interested * capture_rates[2]),
    }
    return {
        "영업일_하루_유동인구": round(daily_total),
        "주고객층_비율": round(segment_share, 3),
        "주고객층_계산방식": segment_method,
        "영업시간대_비율": round(time_share, 3),
        "하루_관심고객_수": round(interested),
        "하루_예상_방문객_시나리오": scenarios,
        "조건": {"연령": target_ages, "성별": target_gender or "전체",
                 "영업시간대": business_hours, "영업요일": open_days},
        "참고": "유동인구는 분기 합계를 91일로 나눈 추정치, 방문 전환율은 가정값임",
    }


# ---------------------------------------------------------------------------
# 한 번에 묶어서 조회 (챗봇 도구용)
# ---------------------------------------------------------------------------

def analyze_foot_traffic(trdar_cd, std_ym=None, station_names=None):
    """상권코드 하나에 대해 정밀 동선 데이터를 병렬로 모두 조회한다."""
    from concurrent.futures import ThreadPoolExecutor

    std_ym = std_ym or find_latest_quarter() or "20241"
    station_names = [s for s in (station_names or []) if s][:3]

    with ThreadPoolExecutor(max_workers=6) as ex:
        f_flpop = ex.submit(_fetch_rows_for_trdar, "VwsmTrdarFlpopQq", trdar_cd, std_ym)
        f_wrc = ex.submit(_fetch_rows_for_trdar, "VwsmTrdarWrcPopltnQq", trdar_cd, std_ym)
        f_fclty = ex.submit(_fetch_rows_for_trdar, "VwsmTrdarFcltyQq", trdar_cd, std_ym)
        f_subway = [ex.submit(get_subway_hourly, s) for s in station_names]

        flpop_rows = f_flpop.result()
        wrc_rows = f_wrc.result()
        fclty_rows = f_fclty.result()
        subway = [f.result() for f in f_subway]

    # 직장인구·집객시설은 유동인구보다 늦게 올라오는 경우가 있어 한 분기 전으로 재시도
    if not wrc_rows:
        wrc_rows = _fetch_rows_for_trdar("VwsmTrdarWrcPopltnQq", trdar_cd, previous_quarter(std_ym))
    if not fclty_rows:
        fclty_rows = _fetch_rows_for_trdar("VwsmTrdarFcltyQq", trdar_cd, previous_quarter(std_ym))

    return {
        "기준_분기": std_ym,
        "flpop_rows": flpop_rows,
        "유동인구_상세": summarize_detailed_flpop(flpop_rows[0] if flpop_rows else None),
        "직장인구": summarize_worker_population(wrc_rows),
        "집객시설": summarize_facilities(fclty_rows),
        "지하철_승하차": subway or None,
    }


# ---------------------------------------------------------------------------
# 데이터 점검용: 각 API가 실제로 어떤 필드명을 주는지 확인 (사이드바 '데이터 점검' 메뉴)
# ---------------------------------------------------------------------------

def inspect_new_apis():
    std_ym = find_latest_quarter()
    report = {"최신_유동인구_분기": std_ym}
    for service in ("VwsmTrdarFlpopQq", "VwsmTrdarWrcPopltnQq", "VwsmTrdarFcltyQq"):
        rows, total, msg = _seoul_get(service, 1, 1, std_ym or "")
        report[service] = {"결과": msg, "전체건수": total, "필드": list(rows[0].keys()) if rows else []}
        if service == "VwsmTrdarFlpopQq" and rows:
            # 값의 단위(분기 합계인지 하루 평균인지) 확인용 샘플
            r = rows[0]
            days = sum(_to_num(r.get(f)) or 0 for f in DAY_FIELDS.values())
            report["유동인구_샘플"] = {
                "상권명": r.get("TRDAR_CD_NM"),
                "총_유동인구": r.get("TOT_FLPOP_CO"),
                "월요일": r.get("MON_FLPOP_CO"),
                "요일합계": days,
            }
    for service in ("VwsmTrdarSelngQq", "VwsmTrdhlStorQq", "TbgisTrdarRelm"):
        rows, total, msg = _seoul_get(service, 1, 1, "" if service == "TbgisTrdarRelm" else (std_ym or ""))
        report[service] = {"결과": msg, "전체건수": total, "필드": list(rows[0].keys()) if rows else [],
                           "첫행_분기": rows[0].get("STDR_YYQU_CD") if rows else None}
    today = date.today()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    rows, total, msg = _seoul_get("CardSubwayTime", 1, 1, "{:04d}{:02d}".format(y, m))
    report["CardSubwayTime"] = {"결과": msg, "전체건수": total, "필드": list(rows[0].keys()) if rows else []}

    # 매출 단위 확인: 카페 한 곳의 보통 월매출은 1,000~2,000만원 수준.
    # '월 환산' 값이 이 근처면 SALES_AMT_MONTHS 설정(3)이 맞는 것이다.
    try:
        from location_recommender import recommend_locations
        from compare_trade_area import CONFIG
        rec = recommend_locations("커피-음료", top_n=1)
        report["매출단위_점검_카페"] = {
            "서울시_원자료_점포당_중앙값_만원": rec.get("서울시_원자료_점포당_중앙값_만원"),
            "현재_나누는_값": CONFIG["SALES_AMT_MONTHS"],
            "업종_표준_월매출_만원": rec.get("업종_표준_월매출_만원(서울 평균 점포 가정)"),
            "비교한_상권_수": rec.get("비교한_상권_수"),
            "error": rec.get("error"),
        }
    except Exception as e:
        report["매출단위_점검_카페"] = "확인 실패: " + str(e)
    return report
