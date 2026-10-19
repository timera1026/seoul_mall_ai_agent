"""
청량리역/경동시장 상권 분석 + 임대료 산정 + 챗봇 연동용 스크립트
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from collections import Counter
from dotenv import load_dotenv
from seoul_dong_codes import SEOUL_DONG_CODE

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
SEOUL_API_KEY = os.getenv("SEOUL_API_KEY")
DATA_GO_KR_KEY = os.getenv("DATA_GO_KR_KEY")
# 행정안전부 도로명주소(juso.go.kr) API 승인키. data.go.kr과는 완전히 다른 포털이라
# 별도로 발급받아야 한다. 주소 -> 법정동코드/번지 변환에 사용 (건축물대장 조회용).
JUSO_API_KEY = os.getenv("JUSO_API_KEY")

SEOUL_API_BASE = "http://openapi.seoul.go.kr:8088"

# ---------------------------------------------------------------------------
# 임대료/추천 로직에서 쓰는 모든 가중치·비율을 한 곳에 모아둔 설정값.
# 값을 바꾸고 싶으면 코드 여기저기를 뒤질 필요 없이 이 CONFIG만 수정하면 된다.
# (전부 경험적 추정치이며, 실제 시장 데이터로 검증된 값은 아니다)
# ---------------------------------------------------------------------------
CONFIG = {
    # 매매가 -> 임대료 환산 시 쓰는 연 환산 임대수익률 (예: 0.05 = 5%)
    "CAP_RATE": 0.05,

    # 층수별 임대료 가중치. 1층(기준)에서 위/아래로 멀어질수록 같은 폭으로 할인되고,
    # 3단계 이상 멀어지면 더 내려가지 않고 level_3_plus 값에서 멈춘다(최대 할인 고정).
    # - ground_1: 1층 (기준, 할인 없음)
    # - level_1 : 2층 / 지하1층 / 옥상층
    # - level_2 : 3층 / 지하2층
    # - level_3_plus: 4층 이상 / 지하3층 이하 (더 내려가도 이 값 밑으로는 안 내려감)
    "FLOOR_WEIGHT": {
        "ground_1": 1.0,
        "level_1": 0.85,
        "level_2": 0.65,
        "level_3_plus": 0.6,
    },

    # 업종 경쟁포화도 점수를 이 범위 안으로 제한 (너무 극단적인 값 방지)
    "SATURATION_SCORE_MIN": 0.85,
    "SATURATION_SCORE_MAX": 1.5,

    # 유동인구 시간대 적합도 점수를 이 범위 안으로 제한. 클램프가 없으면 특정
    # 시간대에 유동인구가 극단적으로 쏠린 상권에서 이 점수가 0에 가깝게 작아지거나
    # 3~4배씩 커져서, 최종 추천 임대료가 기준 매매가 기반 임대료를 터무니없이
    # 벗어나는 문제가 있었다 (예: 새벽 시간대 업종은 사실상 0원 취급되는 식).
    "POP_FIT_SCORE_MIN": 0.7,
    "POP_FIT_SCORE_MAX": 1.4,

    # 정주인구 비율이 수요 점수에 미치는 영향 범위
    # resident_factor = RESIDENT_FACTOR_BASE + RESIDENT_FACTOR_RANGE * (...)
    "RESIDENT_FACTOR_BASE": 0.7,
    "RESIDENT_FACTOR_RANGE": 0.6,

    # 업소 분포를 조회할 기본 반경(m)
    "DEFAULT_RADIUS_M": 500,
}

# 서울시 25개 자치구 법정동코드(5자리) 매핑표.
# 국토부 실거래가(LAWD_CD), KOSIS 주민등록인구(objL1) 조회에 공통으로 사용된다.
# 출처: 행정안전부 법정동코드 전체자료
SEOUL_GU_LAWD_CD = {
    "강남구": "11680", "강동구": "11740", "강북구": "11305", "강서구": "11500",
    "관악구": "11620", "광진구": "11215", "구로구": "11530", "금천구": "11545",
    "노원구": "11350", "도봉구": "11320", "동대문구": "11230", "동작구": "11590",
    "마포구": "11440", "서대문구": "11410", "서초구": "11650", "성동구": "11200",
    "성북구": "11290", "송파구": "11710", "양천구": "11470", "영등포구": "11560",
    "용산구": "11170", "은평구": "11380", "종로구": "11110", "중구": "11140",
    "중랑구": "11260",
}


def find_lawd_cd_by_gu_name(text):
    """
    주소나 키워드 문자열 안에서 서울 자치구 이름을 찾아 정확한 법정동코드를 반환.
    찾지 못하면 None을 반환한다 (이 경우 호출부에서 Claude가 준 lawd_cd로 대체 가능).
    """
    if not text:
        return None
    for gu_name, code in SEOUL_GU_LAWD_CD.items():
        if gu_name in text:
            return code
    return None


def find_dong_code_by_name(text):
    """
    주소나 키워드 문자열 '안에' 동 이름이 그대로 포함되어 있는 경우에만 찾는 방식.
    "서울역", "강남역" 같은 랜드마크명에는 동 이름이 안 들어있어서 못 찾는 경우가
    많다. 이 함수는 보조 수단으로만 쓰고, 가능하면 geocode_naver()가 돌려주는
    실제 법정동명(find_dong_code_by_emd_name)을 우선 사용한다.
    찾지 못하면 None을 반환한다.
    """
    if not text:
        return None
    matched = [(name, code) for name, code in SEOUL_DONG_CODE.items() if name in text]
    if not matched:
        return None
    # 가장 긴(구체적인) 이름으로 매칭된 것을 우선 (예: '청량리동'이 '리동' 등 짧은 오매칭보다 우선)
    matched.sort(key=lambda x: len(x[0]), reverse=True)
    return matched[0][1]


def _dong_key_base_name(key):
    """
    '역삼1동', '돈암제1동', '종로1.2.3.4가동' 같은 행정동 이름에서
    번호/'제' 표기를 제거해 법정동 기준 이름('역삼동', '돈암동', '종로가동')으로 만든다.
    법정동 하나가 여러 행정동으로 쪼개진 경우를 매칭하기 위한 보조 함수.
    """
    return re.sub(r"제?\d+(\.\d+)*", "", key)


def find_dong_code_by_emd_name(emd_name):
    """
    네이버 지오코딩이 돌려주는 법정동명(DONGMYUN, 예: '역삼동', '청량리동')을
    받아서 KOSIS 조회용 행정동 코드를 찾는다.

    1) 이름이 정확히 일치하면 (예: '청량리동' -> '청량리동') 그대로 사용.
    2) 법정동 하나가 여러 행정동으로 쪼개진 경우(예: '역삼동' -> '역삼1동'/'역삼2동')는
       번호를 뗀 기준 이름이 같은 후보들 중 번호가 가장 앞선(제1동 등) 곳을 근사치로 사용한다.
       (완벽하게 정확한 동까지는 아니지만, 구 단위보다는 훨씬 정밀함)

    반환: (코드, 정확히 일치했는지 여부) 튜플. 못 찾으면 (None, False).
    """
    if not emd_name:
        return None, False

    if emd_name in SEOUL_DONG_CODE:
        return SEOUL_DONG_CODE[emd_name], True

    candidates = []
    for key, code in SEOUL_DONG_CODE.items():
        if _dong_key_base_name(key) == emd_name:
            match = re.search(r"\d+", key)
            order = int(match.group()) if match else 0
            candidates.append((order, key, code))

    if not candidates:
        return None, False

    candidates.sort(key=lambda x: x[0])
    return candidates[0][2], False


def geocode_naver(address):
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()
    if not data.get("addresses"):
        raise ValueError("좌표를 찾을 수 없습니다: " + address)
    top = data["addresses"][0]

    dong_name = None
    for el in top.get("addressElements", []):
        if "DONGMYUN" in el.get("types", []):
            dong_name = el.get("longName") or el.get("shortName")
            break

    return {
        "lat": float(top["y"]),
        "lng": float(top["x"]),
        "address": top.get("roadAddress") or top.get("jibunAddress"),
        "dong_name": dong_name,
    }


def get_static_map_image(lat, lng, level=16, w=600, h=350, title=None):
    """
    네이버 Static Map API로 좌표 위치에 마커를 찍은 지도 '이미지'를 서버에서
    직접 받아온다. Dynamic Map(JS)과 달리 브라우저가 아니라 서버가 직접
    ID-KEY 인증으로 호출하기 때문에, Streamlit의 컴포넌트 iframe(about:srcdoc)이
    Referer 체크에 걸려 지도가 안 뜨는 문제 자체가 발생하지 않는다.
    반환: (이미지 바이트 또는 None, 에러 메시지 또는 None) 튜플.
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return None, "NAVER_CLIENT_ID/NAVER_CLIENT_SECRET이 설정되지 않았습니다."

    url = "https://maps.apigw.ntruss.com/map-static/v2/raster"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    params = {
        "w": w,
        "h": h,
        "center": str(lng) + "," + str(lat),
        "level": level,
        "format": "png",
        "markers": "type:d|size:mid|color:Red|pos:" + str(lng) + " " + str(lat),
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=15)
        content_type = res.headers.get("Content-Type", "")
        if res.status_code != 200 or "image" not in content_type:
            # 에러 응답은 보통 JSON/텍스트로 온다. 원문 그대로 보여줘서 정확한 원인 파악.
            error_body = res.text[:500]
            return None, "HTTP " + str(res.status_code) + " / " + content_type + " / " + error_body
        return res.content, None
    except Exception as e:
        return None, type(e).__name__ + ": " + str(e)


def find_trade_area_codes(keyword, max_pages=5):
    matched = []
    page_size = 1000
    for page in range(max_pages):
        start = page * page_size + 1
        end = start + page_size - 1
        url = SEOUL_API_BASE + "/" + SEOUL_API_KEY + "/json/TbgisTrdarRelm/" + str(start) + "/" + str(end) + "/"
        try:
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            payload = res.json()
        except Exception as e:
            print("[경고] 서울시 API 호출 실패, 건너뜁니다:", e)
            break
        rows = payload.get("TbgisTrdarRelm", {}).get("row", [])
        if not rows:
            break
        for row in rows:
            name = row.get("TRDAR_CD_NM", "")
            if keyword in name and "학교" not in name:
                matched.append({
                    "trdar_cd": row.get("TRDAR_CD"),
                    "trdar_cd_nm": name,
                    "trdar_se_cd_nm": row.get("TRDAR_SE_CD_NM"),
                })
        total = int(payload.get("TbgisTrdarRelm", {}).get("list_total_count", 0))
        if end >= total:
            break
    return matched


_LATLNG_TO_5181 = None


def _latlng_to_5181(lat, lng):
    """
    네이버가 주는 위도·경도(WGS84)를, 서울시 상권영역 데이터가 쓰는
    GRS80 중부원점(EPSG:5181, 미터 단위) 좌표로 변환한다.
    """
    global _LATLNG_TO_5181
    if _LATLNG_TO_5181 is None:
        from pyproj import Transformer
        _LATLNG_TO_5181 = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    x, y = _LATLNG_TO_5181.transform(lng, lat)
    return x, y


def find_nearest_trade_area(lat, lng, max_pages=3):
    """
    키워드로 상권을 못 찾았을 때(예: 랜드마크명이 없는 순수 건물 주소) 쓰는 대체 수단.
    좌표(위도·경도)를 EPSG:5181로 변환한 뒤, 서울시 상권영역 전체를 훑어서
    직선거리가 가장 가까운 상권을 찾는다. (경계 폴리곤이 아니라 상권 중심점 기준 근사치)
    """
    try:
        qx, qy = _latlng_to_5181(lat, lng)
    except Exception as e:
        print("[경고] 좌표 변환 실패(pyproj 설치 여부 확인 필요):", e)
        return None

    best = None
    best_dist = None
    page_size = 1000

    for page in range(max_pages):
        start = page * page_size + 1
        end = start + page_size - 1
        url = SEOUL_API_BASE + "/" + SEOUL_API_KEY + "/json/TbgisTrdarRelm/" + str(start) + "/" + str(end) + "/"
        try:
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            payload = res.json()
        except Exception as e:
            print("[경고] 상권영역 전체 조회 실패:", e)
            break

        rows = payload.get("TbgisTrdarRelm", {}).get("row", [])
        if not rows:
            break

        for row in rows:
            try:
                rx = float(row.get("XCNTS_VALUE"))
                ry = float(row.get("YDNTS_VALUE"))
            except (TypeError, ValueError):
                continue
            dist = ((rx - qx) ** 2 + (ry - qy) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = row

        total = int(payload.get("TbgisTrdarRelm", {}).get("list_total_count", 0))
        if end >= total:
            break

    if best is None:
        return None

    return {
        "trdar_cd": best.get("TRDAR_CD"),
        "trdar_cd_nm": best.get("TRDAR_CD_NM"),
        "trdar_se_cd_nm": best.get("TRDAR_SE_CD_NM"),
        "distance_m": round(best_dist, 1) if best_dist is not None else None,
    }


def get_floating_population(trdar_cd, std_ym):
    try:
        matched = []
        page_size = 1000
        for page in range(10):
            start = page * page_size + 1
            end = start + page_size - 1
            page_url = SEOUL_API_BASE + "/" + SEOUL_API_KEY + "/json/VwsmTrdarFlpopQq/" + str(start) + "/" + str(end) + "/" + std_ym
            page_res = requests.get(page_url, timeout=15)
            page_res.raise_for_status()
            page_payload = page_res.json()
            page_rows = page_payload.get("VwsmTrdarFlpopQq", {}).get("row", [])
            if not page_rows:
                break
            for r in page_rows:
                if r.get("TRDAR_CD") == trdar_cd:
                    matched.append(r)
            if matched:
                break
            total = int(page_payload.get("VwsmTrdarFlpopQq", {}).get("list_total_count", 0))
            if end >= total:
                break
        return matched
    except Exception as e:
        print("[경고] 유동인구 조회 실패:", e)
        return []


def _parse_sales_rows(raw_rows):
    """
    VwsmTrdarSelngQq(상권)와 VwsmAdstrdSelngW(행정동) 두 API 모두 같은 필드
    구조(SVC_INDUTY_CD_NM, THSMON_SELNG_AMT 등)를 쓰므로 파싱 로직을 공유한다.
    """
    results = []
    for row in raw_rows:
        results.append({
            "업종명": row.get("SVC_INDUTY_CD_NM"),
            "당월_추정매출_원": row.get("THSMON_SELNG_AMT"),
            "당월_추정매출_건수": row.get("THSMON_SELNG_CO"),
            "주중_매출_원": row.get("MDWK_SELNG_AMT"),
            "주말_매출_원": row.get("WKEND_SELNG_AMT"),
            "남성_매출_원": row.get("ML_SELNG_AMT"),
            "여성_매출_원": row.get("FML_SELNG_AMT"),
            "시간대_11~14_매출_원": row.get("TMZON_11_14_SELNG_AMT"),
            "시간대_17~21_매출_원": row.get("TMZON_17_21_SELNG_AMT"),
        })
    results.sort(key=lambda r: r.get("당월_추정매출_원") or 0, reverse=True)
    return results


def _fetch_sales_by_code(service_name, code_field, code_value, std_ym, max_pages=40):
    """
    추정매출 API(상권/행정동 공통)를 페이지네이션하며 code_field == code_value인
    행만 모은다. 한 지역 안에 업종별로 여러 행이 나오므로(유동인구 API와 달리
    첫 매치에서 멈추면 안 됨), 매치가 시작된 뒤 여러 페이지 연속으로 매치가
    없으면 그 지역 데이터는 다 모은 것으로 보고 중단해 불필요한 호출을 줄인다.
    """
    matched = []
    page_size = 1000
    empty_streak_after_match = 0
    for page in range(max_pages):
        start = page * page_size + 1
        end = start + page_size - 1
        page_url = SEOUL_API_BASE + "/" + SEOUL_API_KEY + "/json/" + service_name + "/" + str(start) + "/" + str(end) + "/" + std_ym
        page_res = requests.get(page_url, timeout=15)
        page_res.raise_for_status()
        page_payload = page_res.json()
        page_rows = page_payload.get(service_name, {}).get("row", [])
        if not page_rows:
            break

        page_matched = [r for r in page_rows if r.get(code_field) == code_value]
        if page_matched:
            matched.extend(page_matched)
            empty_streak_after_match = 0
        elif matched:
            empty_streak_after_match += 1
            if empty_streak_after_match >= 3:
                break

        total = int(page_payload.get(service_name, {}).get("list_total_count", 0))
        if end >= total:
            break

    return matched


def _fetch_sales_by_dong_name(dong_name, std_ym, max_pages=40):
    """
    추정매출-행정동 API(VwsmAdstrdSelngW)를 행정동 '이름'(ADSTRD_CD_NM)으로
    매칭한다. 이 API의 행정동코드(ADSTRD_CD, 8자리)는 KOSIS 정주인구 조회에
    쓰는 10자리 코드(seoul_dong_codes.py)와 체계가 달라 코드로 직접 변환할
    방법이 마땅치 않으므로, 이름으로 대신 매칭한다.

    이름 매칭은 코드 매칭과 달리 정렬 순서를 가정할 수 없어 "매치 후 연속
    실패시 중단" 최적화를 쓰기 어렵다. 대신 max_pages로 상한을 두어 무한정
    호출되지 않게 한다. (행정동 이름이 정확히 일치하는 첫 몇십 페이지 안에서
    보통 찾아지며, 못 찾으면 그 지역은 데이터가 없는 것으로 처리한다)
    """
    matched = []
    page_size = 1000
    for page in range(max_pages):
        start = page * page_size + 1
        end = start + page_size - 1
        page_url = SEOUL_API_BASE + "/" + SEOUL_API_KEY + "/json/VwsmAdstrdSelngW/" + str(start) + "/" + str(end) + "/" + std_ym
        page_res = requests.get(page_url, timeout=15)
        page_res.raise_for_status()
        page_payload = page_res.json()
        page_rows = page_payload.get("VwsmAdstrdSelngW", {}).get("row", [])
        if not page_rows:
            break

        matched.extend(r for r in page_rows if r.get("ADSTRD_CD_NM") == dong_name)

        total = int(page_payload.get("VwsmAdstrdSelngW", {}).get("list_total_count", 0))
        if end >= total:
            break
    return matched


def get_estimated_sales(trdar_cd, std_ym, dong_name=None):
    """
    서울시 상권분석서비스의 카드매출 기반 업종별 추정매출을 조회한다.

    1) 먼저 상권 단위(VwsmTrdarSelngQq, trdar_cd)로 시도한다. 상권코드는 서울시가
       지정한 발달상권/골목상권/전통시장/관광특구 구역에만 존재하므로, 이 구역
       밖의 일반 주소는 매치되는 행이 아예 없을 수 있다.
    2) 상권 단위에서 못 찾았고 dong_name(법정동/행정동명)이 있으면, 행정동 단위
       (VwsmAdstrdSelngW)로 재시도한다. 행정동은 서울 전역에 빠짐없이 존재하므로
       상권 밖 지역까지 커버할 수 있다.

    반환: [{"업종명": "한식음식점", "당월_추정매출_원": ..., ...}, ...] (매출액 큰 순),
          어느 쪽으로도 못 찾으면 빈 리스트.
    """
    try:
        matched = _fetch_sales_by_code("VwsmTrdarSelngQq", "TRDAR_CD", trdar_cd, std_ym)
        if matched:
            return _parse_sales_rows(matched)
    except Exception as e:
        print("[경고] 상권 단위 추정매출 조회 실패:", e)

    if not dong_name:
        return []

    try:
        matched = _fetch_sales_by_dong_name(dong_name, std_ym)
        return _parse_sales_rows(matched)
    except Exception as e:
        print("[경고] 행정동 단위 추정매출 조회 실패:", e)
        return []


def summarize_estimated_sales(sales_rows, category_name=None):
    """
    get_estimated_sales() 결과에서 요약 정보를 뽑는다.
    category_name이 있으면 그 업종명이 포함된 행만 골라 매출 정보를 반환하고
    (예: '한식' -> '한식음식점' 매치), 없으면 전체 업종의 매출 합계와
    업종별 상위 목록을 반환한다.

    당월 매출은 API 특성상 실제 매출이 아니라 카드매출 기반 추정치이므로,
    호출부에서 반드시 "추정치"라는 점을 함께 안내해야 한다.
    """
    if not sales_rows:
        return None

    if category_name:
        matched = [r for r in sales_rows if r.get("업종명") and category_name in r["업종명"]]
        if not matched:
            return None
        row = matched[0]
        amt = row.get("당월_추정매출_원") or 0
        count = row.get("당월_추정매출_건수") or 0
        return {
            "업종명": row["업종명"],
            "당월_추정매출_원": round(amt),
            "당월_추정매출_건수": round(count),
            "건당_평균_추정매출_원": round(amt / count) if count else None,
            "주중_매출_비중": round((row.get("주중_매출_원") or 0) / amt, 3) if amt else None,
            "주말_매출_비중": round((row.get("주말_매출_원") or 0) / amt, 3) if amt else None,
        }

    total_amt = sum((r.get("당월_추정매출_원") or 0) for r in sales_rows)
    top5 = [
        {"업종명": r["업종명"], "당월_추정매출_원": round(r.get("당월_추정매출_원") or 0)}
        for r in sales_rows[:5]
    ]
    return {
        "상권_전체_추정매출_합계_원": round(total_amt),
        "업종수": len(sales_rows),
        "매출_상위5_업종": top5,
    }


def _parse_store_stats_rows(raw_rows):
    """
    서울시 상권분석서비스(점포-상권) API(VwsmTrdhlStorQq) 응답을 파싱한다.
    이 API도 매출 API처럼 한 상권 안에 업종별로 여러 행이 나온다.

    개업률/폐업률(OPBIZ_RT, CLSBIZ_RT)은 API가 이미 계산해서 주는 값으로,
    "해당 분기에 (개업/폐업)한 점포 수 / 전체 유사업종 점포 수" 비율로 추정된다.
    표본이 적은 업종(점포 수 자체가 몇 개 안 되는 경우)은 비율 하나의 변화가
    커 보일 수 있으므로, 호출부에서 similir_induty_stor_co(유사업종 점포 수)도
    함께 참고해 표본 규모를 알 수 있게 남겨둔다.
    """
    results = []
    for row in raw_rows:
        results.append({
            "업종명": row.get("SVC_INDUTY_CD_NM"),
            "유사업종_점포수": row.get("SIMILR_INDUTY_STOR_CO"),
            "점포수": row.get("STOR_CO"),
            "프랜차이즈_점포수": row.get("FRC_STOR_CO"),
            "개업률": row.get("OPBIZ_RT"),
            "개업_점포수": row.get("OPBIZ_STOR_CO"),
            "폐업률": row.get("CLSBIZ_RT"),
            "폐업_점포수": row.get("CLSBIZ_STOR_CO"),
        })
    # 폐업률이 높은 순으로 정렬해두면, 위험 업종을 바로 눈에 띄게 보여줄 수 있다.
    results.sort(key=lambda r: r.get("폐업률") or 0, reverse=True)
    return results


def get_store_stats(trdar_cd, std_ym):
    """
    서울시 상권분석서비스(점포-상권) API(VwsmTrdhlStorQq)로 특정 상권의
    업종별 점포 수, 프랜차이즈 비율, 개업률/폐업률을 조회한다.

    반환: [{"업종명": "제과점", "점포수": 6, "프랜차이즈_점포수": 7,
            "개업률": 8, "폐업률": 0, ...}, ...] (폐업률 높은 순),
          데이터가 없으면 빈 리스트.
    """
    try:
        matched = _fetch_sales_by_code("VwsmTrdhlStorQq", "TRDAR_CD", trdar_cd, std_ym)
        return _parse_store_stats_rows(matched)
    except Exception as e:
        print("[경고] 점포/폐업 통계 조회 실패:", e)
        return []


def summarize_store_stats(store_stats_rows, category_name=None):
    """
    get_store_stats() 결과를 요약한다. category_name이 있으면 그 업종명이
    포함된 행 하나를 찾아 상세 정보를 반환하고, 없으면 상권 전체에서
    폐업률이 높은 업종 상위 목록(위험 신호)을 반환한다.

    개업률/폐업률은 카드사·통신사 등 원천 데이터를 서울신용보증재단이
    가공한 분기별 추정치이며, 실제 사업자등록 개폐업 통계와는 집계 방식이
    다를 수 있다는 점을 호출부에서 함께 안내해야 한다.
    """
    if not store_stats_rows:
        return None

    if category_name:
        matched = [r for r in store_stats_rows if r.get("업종명") and category_name in r["업종명"]]
        if not matched:
            return None
        row = matched[0]
        return {
            "업종명": row["업종명"],
            "점포수": row.get("점포수"),
            "프랜차이즈_점포수": row.get("프랜차이즈_점포수"),
            "프랜차이즈_비율": (
                round(row["프랜차이즈_점포수"] / row["점포수"], 3)
                if row.get("점포수") else None
            ),
            "개업률": row.get("개업률"),
            "폐업률": row.get("폐업률"),
            "표본_유사업종_점포수": row.get("유사업종_점포수"),
        }

    # 표본이 너무 적은 업종(점포 수 3개 미만)은 폐업률 비율이 통계적으로
    # 불안정해서(예: 1개 폐업 = 100% 폐업률처럼 극단적으로 보임) 상위 목록에서
    # 제외하고, 어느 정도 표본이 있는 업종들 중에서만 폐업률 순위를 매긴다.
    reliable = [r for r in store_stats_rows if (r.get("유사업종_점포수") or 0) >= 3]
    top5_closure = [
        {
            "업종명": r["업종명"],
            "폐업률": r.get("폐업률"),
            "점포수": r.get("점포수"),
        }
        for r in reliable[:5]
    ]
    total_stores = sum((r.get("점포수") or 0) for r in store_stats_rows)
    total_franchise = sum((r.get("프랜차이즈_점포수") or 0) for r in store_stats_rows)
    return {
        "상권_전체_점포수": total_stores,
        "상권_전체_프랜차이즈_비율": round(total_franchise / total_stores, 3) if total_stores else None,
        "폐업률_상위5_업종": top5_closure,
    }


def get_population_trend(trdar_cd, quarters=8):
    """
    특정 상권코드(trdar_cd)의 최근 quarters개 분기치 유동인구를 조회해서
    분기별 추이를 반환한다. get_price_trend()와 동일한 패턴.

    분기 코드 형식은 서울 열린데이터광장 기준 "YYYYQ" (예: "20241" = 2024년 1분기,
    "20242" = 2024년 2분기 ... "20244" = 2024년 4분기). 분기 데이터는 보통
    한 분기 정도 늦게 올라오므로, 오늘 날짜 기준 최신 분기가 아직 없을 수 있다
    (그 경우 해당 분기는 값이 비어서 빠지고, 있는 분기만 반환된다).

    반환: [{"std_ym": "20231", "총_유동인구": 12345, ...}, ...] (오래된 분기 -> 최신 분기 순)
    """
    from datetime import date
    from concurrent.futures import ThreadPoolExecutor

    today = date.today()
    year, quarter = today.year, (today.month - 1) // 3 + 1

    std_yms = []
    for _ in range(quarters):
        std_yms.append("{:04d}{:d}".format(year, quarter))
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1
    std_yms.reverse()  # 오래된 -> 최신 순

    def fetch_one(std_ym):
        rows = get_floating_population(trdar_cd, std_ym)
        summary = summarize_population(rows)
        if not summary:
            return None
        summary["std_ym"] = std_ym
        return summary

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch_one, std_yms))

    # 데이터가 없는 분기(None)는 제외하고 반환
    return [r for r in results if r is not None]


def summarize_population(rows):
    if not rows:
        return {}
    latest = rows[0]
    return {
        "총_유동인구": latest.get("TOT_FLPOP_CO"),
        "남성_유동인구": latest.get("ML_FLPOP_CO"),
        "여성_유동인구": latest.get("FML_FLPOP_CO"),
        "10대": latest.get("AGRDE_10_FLPOP_CO"),
        "20대": latest.get("AGRDE_20_FLPOP_CO"),
        "30대": latest.get("AGRDE_30_FLPOP_CO"),
        "40대": latest.get("AGRDE_40_FLPOP_CO"),
        "50대": latest.get("AGRDE_50_FLPOP_CO"),
        "60대_이상": latest.get("AGRDE_60_ABOVE_FLPOP_CO"),
        "시간대_00~06": latest.get("TMZON_00_06_FLPOP_CO"),
        "시간대_06~11": latest.get("TMZON_06_11_FLPOP_CO"),
        "시간대_11~14": latest.get("TMZON_11_14_FLPOP_CO"),
        "시간대_14~17": latest.get("TMZON_14_17_FLPOP_CO"),
        "시간대_17~21": latest.get("TMZON_17_21_FLPOP_CO"),
        "시간대_21~24": latest.get("TMZON_21_24_FLPOP_CO"),
    }


def get_store_breakdown_in_radius(lat, lng, radius_m=500):
    url = "http://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"
    all_items = []
    total_count = None
    page_no = 1
    num_of_rows = 1000

    try:
        while True:
            params = {
                "serviceKey": DATA_GO_KR_KEY,
                "type": "json",
                "radius": radius_m,
                "cx": lng,
                "cy": lat,
                "pageNo": page_no,
                "numOfRows": num_of_rows,
            }
            res = requests.get(url, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()

            body = data.get("body", {})
            items = body.get("items", [])
            if total_count is None:
                total_count = body.get("totalCount")

            if not items:
                break

            all_items.extend(items)

            if total_count is not None and len(all_items) >= total_count:
                break
            if page_no > 20:
                break
            page_no = page_no + 1

        large_category = Counter()
        mid_category = Counter()
        for item in all_items:
            large_name = item.get("indsLclsNm")
            mid_name = item.get("indsMclsNm")
            if large_name:
                large_category[large_name] += 1
            if mid_name:
                mid_category[mid_name] += 1

        result = {
            "total_count": total_count if total_count is not None else len(all_items),
            "collected_count": len(all_items),
            "large_category": large_category,
            "mid_category": mid_category,
        }
        return result
    except Exception as e:
        print("[경고] 업소 목록 조회 실패:", e)
        return None


SEOUL_BUS_API_BASE = "http://ws.bus.go.kr/api/rest/stationinfo"


def get_nearby_bus_stations(lat, lng, radius_m=300):
    """
    서울시 정류소정보조회 서비스(getStationByPos)로 좌표 반경 내 버스정류장
    목록을 조회한다. 이 API의 tmX/tmY 파라미터는 이름과 달리 실제로는
    WGS84 좌표(보통 쓰는 위도/경도)라서 별도 좌표 변환이 필요 없다.
    반경은 API 제약상 최대 1500m까지만 허용된다.

    이 API는 type=json을 줘도 예전 XML 시절 흔적으로 응답 최상위 태그가
    <ServiceResult>인 XML 구조를 그대로 JSON으로 감싸서 준다
    (다른 서울시 API의 {서비스명: {...}} 형태와 다름). 그래서 XML로 받아
    파싱하는 편이 더 안정적이다.

    반환: [{"정류소번호": "03737", "정류소명": "...", "거리_m": 48}, ...],
          실패 시 빈 리스트.
    """
    if not DATA_GO_KR_KEY:
        return []
    if radius_m > 1500:
        radius_m = 1500

    url = SEOUL_BUS_API_BASE + "/getStationByPos"
    params = {
        "ServiceKey": DATA_GO_KR_KEY,
        "tmX": lng,
        "tmY": lat,
        "radius": radius_m,
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        root = ET.fromstring(res.content)

        header_msg = root.findtext(".//headerMsg", default="")
        if header_msg and "정상" not in header_msg:
            print("[경고] 버스정류소 조회 API 오류:", header_msg)
            return []

        stations = []
        for item in root.findall(".//itemList"):
            ars_id = item.findtext("arsId")
            station_nm = item.findtext("stationNm")
            dist = item.findtext("dist")
            if not ars_id:
                continue
            stations.append({
                "정류소번호": ars_id,
                "정류소명": station_nm,
                "거리_m": int(dist) if dist and dist.isdigit() else None,
            })
        return stations
    except Exception as e:
        print("[경고] 버스정류소 조회 실패:", e)
        return []


def get_route_count_for_station(ars_id):
    """
    특정 정류소(arsId)를 경유하는 버스 노선 개수를 조회한다(getRouteByStation).
    반환: 정수(경유 노선 수). 실패 시 None.
    """
    if not DATA_GO_KR_KEY or not ars_id:
        return None

    url = SEOUL_BUS_API_BASE + "/getRouteByStation"
    params = {
        "ServiceKey": DATA_GO_KR_KEY,
        "arsId": ars_id,
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        root = ET.fromstring(res.content)
        return len(root.findall(".//itemList"))
    except Exception as e:
        print("[경고] 정류소 경유노선 조회 실패:", e)
        return None


def get_transit_accessibility(lat, lng, radius_m=300):
    """
    특정 좌표 주변의 버스 접근성을 종합해서 반환한다.
    반경 내 정류장을 찾고, 정류장별로 경유 노선 수를 병렬 조회해서
    (중복 제거한) 총 노선 수와 정류장별 상세 목록을 함께 준다.

    지하철은 이미 유동인구 데이터(역세권 상권코드)로 간접 반영되고 있으므로,
    여기서는 버스만 다룬다. 정류장이 많아도 다니는 노선이 겹치면 실질
    접근성은 낮을 수 있어, 노선 수를 정류장별 합산이 아니라 노선ID 기준
    중복 제거한 값으로 계산하려 했으나, getRouteByStation 응답에 노선ID가
    포함되어 있지 않은 축약형 조회라 여기서는 "노선 수 합산"(정류장 간 중복
    가능성 있음)으로 근사한다. 참고용 지표로만 취급해야 한다.
    """
    stations = get_nearby_bus_stations(lat, lng, radius_m)
    if not stations:
        return {
            "정류장_수": 0,
            "총_경유노선_수_합산": 0,
            "정류장_목록": [],
        }

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        route_counts = list(executor.map(
            lambda s: get_route_count_for_station(s["정류소번호"]), stations
        ))

    total_routes = 0
    station_list = []
    for station, route_count in zip(stations, route_counts):
        station_list.append({
            **station,
            "경유노선_수": route_count,
        })
        if route_count:
            total_routes += route_count

    # 가까운 정류장부터 보여주는 게 실무적으로 더 유용하므로 거리순 정렬.
    station_list.sort(key=lambda s: s["거리_m"] if s["거리_m"] is not None else 9999)

    return {
        "정류장_수": len(stations),
        "총_경유노선_수_합산": total_routes,
        "정류장_목록": station_list[:10],  # 상위 10개까지만 (응답 크기 제한)
    }


def get_commercial_sale_prices(lawd_cd, deal_ym):
    """
    국토교통부 상업업무용 부동산 매매 실거래가 조회.
    이 API는 type=json을 줘도 XML로만 응답하므로 XML로 파싱한다.
    lawd_cd: 법정동코드 앞5자리 (동대문구 = 11230)
    deal_ym: 계약년월 6자리 (예: 202412)
    """
    url = "https://apis.data.go.kr/1613000/RTMSDataSvcNrgTrade/getRTMSDataSvcNrgTrade"
    params = {
        "serviceKey": DATA_GO_KR_KEY,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": 1,
        "numOfRows": 1000,
    }
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()

        root = ET.fromstring(res.text)
        items = []
        for item_el in root.iter("item"):
            item_dict = {}
            for child in item_el:
                item_dict[child.tag] = (child.text or "").strip()
            items.append(item_dict)

        return items
    except Exception as e:
        print("[경고] 매매실거래가 조회 실패:", e)
        return []


def search_juso(keyword):
    """
    행정안전부 도로명주소 API(juso.go.kr)로 주소를 검색해서, 건축물대장 조회에
    필요한 법정동코드(10자리, 시군구5자리+법정동5자리)와 지번(본번/부번)을 얻는다.
    도로명주소든 지번주소든 둘 다 검색어로 넣을 수 있다.
    반환: {"road_addr":..., "jibun_addr":..., "adm_cd":..., "bun":..., "ji":..., "bd_nm":...}
    또는 못 찾으면 None.
    """
    if not JUSO_API_KEY:
        return None

    url = "https://www.juso.go.kr/addrlink/addrLinkApi.do"
    params = {
        "confmKey": JUSO_API_KEY,
        "currentPage": 1,
        "countPerPage": 1,
        "keyword": keyword,
        "resultType": "json",
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        results = data.get("results", {})
        common = results.get("common", {})
        error_code = common.get("errorCode")
        if error_code and error_code != "0":
            print("[경고] juso.go.kr 검색 실패:", common.get("errorMessage"))
            return None

        juso_list = results.get("juso", [])
        if not juso_list:
            return None

        top = juso_list[0]
        return {
            "road_addr": top.get("roadAddr"),
            "jibun_addr": top.get("jibunAddr"),
            "adm_cd": top.get("admCd"),
            "bun": top.get("lnbrMnnm"),
            "ji": top.get("lnbrSlno"),
            "bd_nm": top.get("bdNm"),
        }
    except Exception as e:
        print("[경고] juso.go.kr 호출 실패:", e)
        return None


def get_building_register_title(sigungu_cd, bjdong_cd, bun, ji):
    """
    국토교통부 건축HUB_건축물대장정보 서비스의 '표제부' 조회.
    sigungu_cd(5자리) + bjdong_cd(5자리) + 본번/부번으로 특정 건물을 지정한다.
    반환: item 딕셔너리 리스트 (보통 동별로 여러 개일 수 있어 첫 항목을 대표로 사용)
    """
    url = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
    bun_str = str(bun or "0").zfill(4)
    ji_str = str(ji or "0").zfill(4)
    params = {
        "serviceKey": DATA_GO_KR_KEY,
        "sigunguCd": sigungu_cd,
        "bjdongCd": bjdong_cd,
        "platGbCd": "0",
        "bun": bun_str,
        "ji": ji_str,
        "numOfRows": 5,
        "pageNo": 1,
    }
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()

        root = ET.fromstring(res.text)

        # 오류 응답인지 먼저 확인 (resultCode가 있고 00이 아니면 에러)
        result_code = root.findtext(".//resultCode")
        if result_code and result_code != "00":
            print("[경고] 건축물대장 API 오류:", root.findtext(".//resultMsg"))
            return []

        items = []
        for item_el in root.iter("item"):
            item_dict = {}
            for child in item_el:
                item_dict[child.tag] = (child.text or "").strip()
            items.append(item_dict)
        return items
    except Exception as e:
        print("[경고] 건축물대장 조회 실패:", e)
        return []


def get_building_info(address):
    """
    주소(도로명 또는 지번) 하나로 juso.go.kr -> 법정동코드/번지 변환 -> 건축HUB 표제부 조회까지
    한 번에 처리하는 챗봇용 함수.
    """
    if not JUSO_API_KEY:
        return {"error": "JUSO_API_KEY가 설정되지 않았습니다. juso.go.kr에서 승인키를 발급받아 Secrets에 추가하세요."}
    if not DATA_GO_KR_KEY:
        return {"error": "DATA_GO_KR_KEY가 설정되지 않았습니다."}

    juso = search_juso(address)
    if not juso or not juso.get("adm_cd"):
        return {"error": "주소를 찾을 수 없습니다: " + address}

    adm_cd = juso["adm_cd"]
    if len(adm_cd) < 10:
        return {"error": "법정동코드 형식이 예상과 다릅니다: " + adm_cd, "juso": juso}

    sigungu_cd = adm_cd[:5]
    bjdong_cd = adm_cd[5:10]

    items = get_building_register_title(sigungu_cd, bjdong_cd, juso.get("bun"), juso.get("ji"))
    if not items:
        return {"error": "건축물대장 정보를 찾을 수 없습니다 (신축/미등록 건물이거나 조회 조건 불일치일 수 있음): " + address, "juso": juso}

    b = items[0]
    return {
        "juso": juso,
        "건물명": b.get("bldNm") or juso.get("bd_nm"),
        "주용도": b.get("mainPurpsCdNm"),
        "구조": b.get("strctCdNm"),
        "연면적_㎡": b.get("totArea"),
        "건축면적_㎡": b.get("archArea"),
        "대지면적_㎡": b.get("platArea"),
        "건폐율_%": b.get("bcRat"),
        "용적률_%": b.get("vlRat"),
        "지상층수": b.get("grndFlrCnt"),
        "지하층수": b.get("ugrndFlrCnt"),
        "승용승강기수": b.get("rideUseElvtCnt"),
        "비상용승강기수": b.get("emgenUseElvtCnt"),
        "사용승인일": b.get("useAprDay"),
        "허가일": b.get("pmsDay"),
        "높이_m": b.get("heit"),
        "세대수": b.get("hhldCnt"),
    }


def _remove_price_outliers(values):
    """
    평당가 표본 중 극단치를 걸러낸다. 실거래가는 표본 수가 몇 건 안 되는 경우가
    많아서, 단순 산술평균을 쓰면 유난히 싸거나 비싼 거래 1건이 평균 전체를
    끌고 가버리는 문제가 있었다 (예: 나머지는 평당 800만원인데 특수거래 1건이
    평당 3000만원이면 평균이 확 튀어서 임대료 추정치도 같이 튀어버림).

    중앙값(median) 기준 절대편차(MAD)를 이용해 중앙값에서 너무 멀리 떨어진
    값을 제거한다. 표본이 4건 미만이면 통계적으로 이상치 판단이 무의미하므로
    그대로 반환한다.
    """
    if len(values) < 4:
        return values

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = (sorted_vals[n // 2] if n % 2 == 1
              else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2)

    deviations = sorted([abs(v - median) for v in values])
    mad = (deviations[n // 2] if n % 2 == 1
           else (deviations[n // 2 - 1] + deviations[n // 2]) / 2)

    if mad == 0:
        # 편차가 전부 0(모든 값이 동일)이면 걸러낼 이상치가 없음
        return values

    # 표준정규분포 환산 계수(0.6745)를 적용한 modified z-score가 3.5를 넘으면
    # 이상치로 간주하는 통계학의 일반적인 관례를 따른다.
    threshold = 3.5
    filtered = [v for v in values if 0.6745 * abs(v - median) / mad <= threshold]

    # 필터링 후 남은 값이 너무 적으면(전부 걸러졌거나 1개만 남으면) 신뢰하기
    # 어려우므로 원본 값을 그대로 사용한다.
    if len(filtered) < 2:
        return values
    return filtered


def estimate_price_per_pyeong(items, area_field="buildingAr", price_field="dealAmount", dong_name=None):
    """
    dong_name을 주면 그 동(umdNm)의 거래 건만 필터링해서 평균을 낸다.
    (구 전체 평균이 아니라 동네별로 다른 시세를 반영하기 위함)

    극단치(이상 거래) 때문에 평균이 왜곡되는 것을 막기 위해, 표본이 4건 이상이면
    _remove_price_outliers로 먼저 이상치를 거른 뒤 평균을 낸다.
    """
    if dong_name:
        filtered = [item for item in items if (item.get("umdNm") or "").strip() == dong_name]
        if filtered:
            items = filtered
        # 해당 동 거래가 하나도 없으면 items를 그대로(구 전체) 둬서 값이라도 나오게 한다.

    values = []
    for item in items:
        try:
            area_sqm = float(str(item.get(area_field, 0)).replace(",", "").strip())
            price_str = str(item.get(price_field, "0")).replace(",", "").strip()
            price_manwon = float(price_str)
            if area_sqm <= 0:
                continue
            pyeong = area_sqm / 3.3058
            price_per_pyeong = price_manwon / pyeong
            values.append(price_per_pyeong)
        except Exception:
            continue

    if not values:
        return None

    values = _remove_price_outliers(values)
    return sum(values) / len(values)


def get_average_price_recent(lawd_cd, dong_name=None, max_months_back=6, cap_rate=None):
    """
    이번 달부터 거슬러 올라가며(최대 max_months_back개월), 실거래가 있는
    가장 최근 달의 '평당 평균 매매가'를 찾고, 그걸 바탕으로
    '평당 월 임대료 시세'까지 함께 계산해서 반환한다.
    (임대료 시세 = 평당 매매가 x 연 임대수익률(cap_rate) / 12)

    dong_name을 주면 그 동(umdNm) 거래 건만으로 평균을 내려고 시도한다. 이래야
    같은 구 안의 서로 다른 동/역세권(예: 오목교역 vs 목동역)이 똑같은 시세로
    뭉뚱그려지지 않는다. 해당 동에 거래가 없으면 구 전체 평균으로 자동 대체된다.

    반환: {"avg_price_per_pyeong_manwon": ..., "avg_monthly_rent_per_pyeong_manwon": ...,
           "deal_ym_used": ..., "sample_count": ..., "price_level": "동 단위"|"구 단위"} 또는 None
    """
    if cap_rate is None:
        cap_rate = CONFIG["CAP_RATE"]

    from datetime import date

    today = date.today()
    year = today.year
    month = today.month

    # 동 단위 데이터를 몇 달치까지 우선 탐색하고, 그동안 못 찾으면 구 전체 평균으로
    # 대체한다. 최근 달의 구 데이터를 먼저 써버리면 더 과거의 동 단위 데이터를
    # 놓칠 수 있어서, 구 단위 대체값은 일단 저장해두고 동 단위를 끝까지 우선 찾는다.
    gu_fallback = None

    for _ in range(max_months_back):
        deal_ym = "{:04d}{:02d}".format(year, month)
        items = get_commercial_sale_prices(lawd_cd, deal_ym)

        dong_items = None
        if dong_name:
            dong_items = [item for item in items if (item.get("umdNm") or "").strip() == dong_name]

        if dong_items:
            avg_price = estimate_price_per_pyeong(dong_items)
            if avg_price is not None:
                monthly_rent_per_pyeong = (avg_price * cap_rate) / 12
                return {
                    "avg_price_per_pyeong_manwon": round(avg_price, 1),
                    "avg_monthly_rent_per_pyeong_manwon": round(monthly_rent_per_pyeong, 2),
                    "deal_ym_used": deal_ym,
                    "sample_count": len(dong_items),
                    "price_level": "동 단위",
                }

        if gu_fallback is None:
            avg_price = estimate_price_per_pyeong(items)
            if avg_price is not None:
                monthly_rent_per_pyeong = (avg_price * cap_rate) / 12
                gu_fallback = {
                    "avg_price_per_pyeong_manwon": round(avg_price, 1),
                    "avg_monthly_rent_per_pyeong_manwon": round(monthly_rent_per_pyeong, 2),
                    "deal_ym_used": deal_ym,
                    "sample_count": len(items),
                    "price_level": "구 단위" + ("(해당 동 거래 없어 대체)" if dong_name else ""),
                }

        month -= 1
        if month == 0:
            month = 12
            year -= 1

    return gu_fallback


def get_price_trend(lawd_cd, dong_name=None, months=6):
    """
    최근 months개월치 상업업무용 실거래가를 월별로 조회해서, 각 달의 평당 평균
    매매가 추이를 반환한다. dong_name이 있으면 그 동 거래를 우선 쓰고, 없는 달은
    구 전체로 자동 대체된다 (get_average_price_recent와 동일한 원칙).
    반환: [{"deal_ym": "202501", "avg_price_per_pyeong_manwon": 1234.5,
            "sample_count": 5, "price_level": "동 단위"}, ...] (오래된 달 -> 최신 달 순)
    거래가 없는 달은 avg_price_per_pyeong_manwon이 None으로 들어간다(그래프에서 빈 구간).
    """
    from datetime import date
    from concurrent.futures import ThreadPoolExecutor

    today = date.today()
    year, month = today.year, today.month
    deal_yms = []
    for _ in range(months):
        deal_yms.append("{:04d}{:02d}".format(year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    deal_yms.reverse()  # 오래된 -> 최신 순으로 정렬

    def fetch_one(deal_ym):
        items = get_commercial_sale_prices(lawd_cd, deal_ym)

        dong_items = None
        if dong_name:
            dong_items = [item for item in items if (item.get("umdNm") or "").strip() == dong_name]

        if dong_items:
            avg = estimate_price_per_pyeong(dong_items)
            level = "동 단위"
            count = len(dong_items)
        else:
            avg = estimate_price_per_pyeong(items)
            level = "구 단위" + ("(해당 동 거래 없어 대체)" if dong_name else "")
            count = len(items)

        return {
            "deal_ym": deal_ym,
            "avg_price_per_pyeong_manwon": round(avg, 1) if avg is not None else None,
            "sample_count": count,
            "price_level": level,
        }

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch_one, deal_yms))

    return results


def derive_price_info_from_trend(trend, cap_rate=None):
    """
    get_price_trend()이 이미 조회해 둔 월별 데이터에서, 값이 있는 가장 최근 달을
    골라 요약(평당 매매가/월임대료 시세)을 만든다. get_average_price_recent()와
    똑같은 실거래가 API를 또 호출하지 않기 위한 용도 (중복 호출 제거로 속도 개선).
    """
    if cap_rate is None:
        cap_rate = CONFIG["CAP_RATE"]
    if not trend:
        return None

    for row in reversed(trend):  # 최신 -> 과거 순으로 값이 있는 가장 최근 달을 찾는다
        avg_price = row.get("avg_price_per_pyeong_manwon")
        if avg_price is not None:
            monthly_rent_per_pyeong = (avg_price * cap_rate) / 12
            return {
                "avg_price_per_pyeong_manwon": avg_price,
                "avg_monthly_rent_per_pyeong_manwon": round(monthly_rent_per_pyeong, 2),
                "deal_ym_used": row["deal_ym"],
                "sample_count": row["sample_count"],
                "price_level": row["price_level"],
            }
    return None


def get_resident_population(lawd_cd, max_months_back=6):
    """
    KOSIS 주민등록인구(DT_1B04005N)로 정주인구(실제 거주 등록인구)를 조회.
    lawd_cd: 법정동코드 5자리(구 단위) 또는 행정동 10자리 코드(동 단위).

    예전엔 이번 달부터 한 달씩 거슬러 올라가며 최대 6번 순차 호출했는데,
    KOSIS API는 startPrdDe~endPrdDe로 기간을 한 번에 요청할 수 있어서
    한 번의 호출로 바꿔 속도를 개선했다. 응답에는 그 기간 내 여러 달 데이터가
    올 수 있으니, 그중 가장 최근 달(prdDe가 가장 큰 값)을 골라 쓴다.
    반환: {"total_population": ..., "prd_de_used": ...} 또는 None
    """
    from datetime import date

    kosis_api_key = os.getenv("KOSIS_API_KEY")
    if not kosis_api_key:
        return None

    url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

    today = date.today()
    end_prd_de = "{:04d}{:02d}".format(today.year, today.month)

    start_year, start_month = today.year, today.month - (max_months_back - 1)
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    start_prd_de = "{:04d}{:02d}".format(start_year, start_month)

    params = {
        "method": "getList",
        "apiKey": kosis_api_key,
        "itmId": "T2",
        "objL1": lawd_cd,
        "objL2": "0",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": start_prd_de,
        "endPrdDe": end_prd_de,
        "orgId": "101",
        "tblId": "DT_1B04005N",
    }
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        data = res.json()
        if not isinstance(data, list) or not data:
            return None

        latest = max(data, key=lambda row: row.get("PRD_DE", ""))
        return {
            "total_population": int(latest.get("DT", 0)),
            "prd_de_used": latest.get("PRD_DE"),
        }
    except Exception as e:
        print("[경고] 정주인구 조회 실패:", e)
        return None


def compute_population_fit_score(pop_summary, business_hours):
    """
    영업시간대에 유동인구가 얼마나 몰리는지를 0.7~1.4 사이 점수로 반환한다
    (CONFIG의 POP_FIT_SCORE_MIN/MAX). 클램프가 없으면 특정 시간대(예: 새벽)에
    유동인구가 극단적으로 적거나 많은 상권에서 점수가 0에 가깝게 떨어지거나
    3~4배씩 튀어서, 최종 임대료가 기준 매매가 기반 임대료를 비상식적으로
    벗어나는 문제가 있었다. 경쟁포화도 점수(compute_saturation_score)와
    동일하게 상하한을 두어, "참고용 보정"이지 "몇 배씩 뒤집는 요인"은
    되지 않도록 한다.
    """
    if not pop_summary:
        return 1.0

    time_keys = [
        "시간대_00~06", "시간대_06~11", "시간대_11~14",
        "시간대_14~17", "시간대_17~21", "시간대_21~24",
    ]
    total = sum(pop_summary.get(k, 0) or 0 for k in time_keys)
    if total == 0:
        return 1.0

    target_sum = sum(pop_summary.get(k, 0) or 0 for k in business_hours)
    target_ratio = target_sum / total
    even_ratio = len(business_hours) / len(time_keys)

    if even_ratio == 0:
        return 1.0

    score = target_ratio / even_ratio
    score = max(CONFIG["POP_FIT_SCORE_MIN"], min(score, CONFIG["POP_FIT_SCORE_MAX"]))
    return score


def _get_category_supply_count(counter, category_name):
    """
    정부 원본 데이터의 업종명 표기가 지역/레코드마다 조금씩 다른 경우가 많다
    (예: '비알코올', '비알코올음료', '비알코올 음료점업' 등 제각각).
    그래서 정확히 일치하는 이름만 찾지 않고, category_name 글자가 포함된
    모든 업종명의 개수를 합산해 좀 더 안정적으로 잡는다.
    """
    if not counter:
        return 0
    total = 0
    for name, count in counter.items():
        if category_name in name:
            total += count
    return total


def compute_saturation_score(store_breakdown, category_name, category_level="mid"):
    if not store_breakdown:
        return 1.0

    counter = store_breakdown["mid_category"] if category_level == "mid" else store_breakdown["large_category"]
    total_count = store_breakdown["total_count"] or 1

    # category_name이 표시명(예: '카페')으로 들어올 수 있으므로,
    # 원래 정부 분류명(예: '비알코올')으로도 매칭을 시도한다.
    lookup_name = category_name
    if not any(category_name in name for name in counter):
        for raw_name, disp_name in CATEGORY_DISPLAY_NAMES.items():
            if disp_name == category_name:
                lookup_name = raw_name
                break

    count = _get_category_supply_count(counter, lookup_name)
    share = count / total_count

    average_share = 1.0 / max(len(counter), 1)
    if average_share == 0:
        return 1.0

    saturation_ratio = share / average_share
    if saturation_ratio == 0:
        return CONFIG["SATURATION_SCORE_MAX"]
    score = 1.0 / saturation_ratio
    score = max(CONFIG["SATURATION_SCORE_MIN"], min(score, CONFIG["SATURATION_SCORE_MAX"]))
    return score


# ---------------------------------------------------------------------------
# 정부 데이터의 딱딱한 업종 분류명을 사람이 이해하기 쉬운 이름으로 바꿔주는 표.
# 내부 계산(공급량 매칭 등)은 원래 이름을 그대로 쓰고, 화면/답변에 보여줄 때만
# 이 표를 거쳐서 변환한다.
# ---------------------------------------------------------------------------
CATEGORY_DISPLAY_NAMES = {
    "비알코올": "카페",
}


def display_category_name(name):
    # 정확히 일치하는 이름 먼저 시도
    if name in CATEGORY_DISPLAY_NAMES:
        return CATEGORY_DISPLAY_NAMES[name]
    # 표기가 조금 다른 경우(예: '비알코올음료')를 대비해 포함 여부로도 시도
    for raw_name, disp_name in CATEGORY_DISPLAY_NAMES.items():
        if raw_name in name:
            return disp_name
    return name


# ---------------------------------------------------------------------------
# 수요 기반 업종 추천 로직
# ---------------------------------------------------------------------------
# 주의: 아래 가중치는 실제 매출/소비 데이터가 아니라, 일반적으로 알려진 소비
# 패턴(연령대별 선호, 시간대별 이용 패턴, 유동/정주인구 의존도)을 바탕으로 설계한
# 휴리스틱(경험적 추정) 값이다. "정답"이 아니라 "단순 업종 개수 비교보다 나은
# 참고 지표" 정도로 취급해야 한다.
CATEGORY_DEMAND_PROFILE = {
    "한식": {
        "age": {"10대": 0.5, "20대": 0.7, "30대": 0.9, "40대": 1.0, "50대": 1.0, "60대_이상": 0.9},
        "time": {"시간대_00~06": 0.1, "시간대_06~11": 0.3, "시간대_11~14": 1.0,
                 "시간대_14~17": 0.4, "시간대_17~21": 1.0, "시간대_21~24": 0.5},
        "resident_pref": 0.6,
    },
    "주점": {
        "age": {"10대": 0.1, "20대": 1.0, "30대": 0.9, "40대": 0.7, "50대": 0.5, "60대_이상": 0.3},
        "time": {"시간대_00~06": 0.3, "시간대_06~11": 0.05, "시간대_11~14": 0.2,
                 "시간대_14~17": 0.2, "시간대_17~21": 1.0, "시간대_21~24": 1.0},
        "resident_pref": 0.3,
    },
    "비알코올": {
        "age": {"10대": 0.6, "20대": 1.0, "30대": 0.9, "40대": 0.7, "50대": 0.5, "60대_이상": 0.4},
        "time": {"시간대_00~06": 0.1, "시간대_06~11": 0.7, "시간대_11~14": 0.9,
                 "시간대_14~17": 1.0, "시간대_17~21": 0.6, "시간대_21~24": 0.3},
        "resident_pref": 0.4,
    },
    "의원": {
        "age": {"10대": 0.3, "20대": 0.4, "30대": 0.5, "40대": 0.7, "50대": 0.9, "60대_이상": 1.0},
        "time": {"시간대_00~06": 0.05, "시간대_06~11": 0.8, "시간대_11~14": 0.9,
                 "시간대_14~17": 0.8, "시간대_17~21": 0.4, "시간대_21~24": 0.05},
        "resident_pref": 0.8,
    },
    "의약·화장품 소매": {
        "age": {"10대": 0.4, "20대": 0.7, "30대": 0.7, "40대": 0.7, "50대": 0.8, "60대_이상": 0.9},
        "time": {"시간대_00~06": 0.1, "시간대_06~11": 0.6, "시간대_11~14": 0.7,
                 "시간대_14~17": 0.7, "시간대_17~21": 0.7, "시간대_21~24": 0.3},
        "resident_pref": 0.7,
    },
    "섬유·의복·신발 소매": {
        "age": {"10대": 0.6, "20대": 1.0, "30대": 0.8, "40대": 0.5, "50대": 0.4, "60대_이상": 0.3},
        "time": {"시간대_00~06": 0.05, "시간대_06~11": 0.3, "시간대_11~14": 0.8,
                 "시간대_14~17": 1.0, "시간대_17~21": 0.7, "시간대_21~24": 0.2},
        "resident_pref": 0.2,
    },
    "식료품 소매": {
        "age": {"10대": 0.4, "20대": 0.6, "30대": 0.8, "40대": 0.9, "50대": 0.9, "60대_이상": 0.9},
        "time": {"시간대_00~06": 0.1, "시간대_06~11": 0.7, "시간대_11~14": 0.7,
                 "시간대_14~17": 0.8, "시간대_17~21": 0.9, "시간대_21~24": 0.3},
        "resident_pref": 0.9,
    },
    "종합 소매": {
        "age": {"10대": 0.7, "20대": 0.9, "30대": 0.8, "40대": 0.7, "50대": 0.6, "60대_이상": 0.5},
        "time": {"시간대_00~06": 0.4, "시간대_06~11": 0.7, "시간대_11~14": 0.7,
                 "시간대_14~17": 0.7, "시간대_17~21": 0.8, "시간대_21~24": 0.6},
        "resident_pref": 0.6,
    },
    "이용·미용": {
        "age": {"10대": 0.5, "20대": 0.9, "30대": 0.9, "40대": 0.8, "50대": 0.7, "60대_이상": 0.6},
        "time": {"시간대_00~06": 0.02, "시간대_06~11": 0.4, "시간대_11~14": 0.8,
                 "시간대_14~17": 1.0, "시간대_17~21": 0.7, "시간대_21~24": 0.1},
        "resident_pref": 0.6,
    },
    "기타 간이": {
        "age": {"10대": 0.9, "20대": 0.9, "30대": 0.7, "40대": 0.6, "50대": 0.5, "60대_이상": 0.4},
        "time": {"시간대_00~06": 0.1, "시간대_06~11": 0.4, "시간대_11~14": 1.0,
                 "시간대_14~17": 0.7, "시간대_17~21": 0.7, "시간대_21~24": 0.3},
        "resident_pref": 0.4,
    },
    "일반 숙박": {
        "age": {"10대": 0.2, "20대": 0.8, "30대": 0.9, "40대": 0.6, "50대": 0.4, "60대_이상": 0.3},
        "time": {"시간대_00~06": 0.6, "시간대_06~11": 0.2, "시간대_11~14": 0.2,
                 "시간대_14~17": 0.3, "시간대_17~21": 0.7, "시간대_21~24": 1.0},
        "resident_pref": 0.05,
    },
    "유원지·오락": {
        "age": {"10대": 0.9, "20대": 1.0, "30대": 0.7, "40대": 0.4, "50대": 0.2, "60대_이상": 0.1},
        "time": {"시간대_00~06": 0.2, "시간대_06~11": 0.1, "시간대_11~14": 0.4,
                 "시간대_14~17": 0.7, "시간대_17~21": 1.0, "시간대_21~24": 0.8},
        "resident_pref": 0.2,
    },
    "기타 교육": {
        "age": {"10대": 1.0, "20대": 0.6, "30대": 0.5, "40대": 0.7, "50대": 0.3, "60대_이상": 0.1},
        "time": {"시간대_00~06": 0.02, "시간대_06~11": 0.2, "시간대_11~14": 0.3,
                 "시간대_14~17": 1.0, "시간대_17~21": 0.9, "시간대_21~24": 0.1},
        "resident_pref": 0.9,
    },
    "부동산 서비스": {
        "age": {"10대": 0.05, "20대": 0.4, "30대": 0.9, "40대": 1.0, "50대": 0.9, "60대_이상": 0.6},
        "time": {"시간대_00~06": 0.02, "시간대_06~11": 0.6, "시간대_11~14": 0.8,
                 "시간대_14~17": 0.8, "시간대_17~21": 0.4, "시간대_21~24": 0.05},
        "resident_pref": 0.9,
    },
    "회계·세무": {
        "age": {"10대": 0.02, "20대": 0.3, "30대": 0.7, "40대": 1.0, "50대": 0.9, "60대_이상": 0.5},
        "time": {"시간대_00~06": 0.02, "시간대_06~11": 0.7, "시간대_11~14": 0.8,
                 "시간대_14~17": 0.7, "시간대_17~21": 0.3, "시간대_21~24": 0.02},
        "resident_pref": 0.4,
    },
}


def compute_demand_score(category_name, pop_summary, resident_population=None):
    """
    이 업종의 수요 특성표와 실제 지역 인구 데이터(연령대/시간대/정주인구 비율)를
    비교해 0~1 부근의 '예상 수요 점수'를 계산한다. 알려지지 않은 업종은 None 반환.
    """
    profile = CATEGORY_DEMAND_PROFILE.get(category_name)
    if not profile or not pop_summary:
        return None

    age_keys = ["10대", "20대", "30대", "40대", "50대", "60대_이상"]
    total_age = sum((pop_summary.get(k, 0) or 0) for k in age_keys)
    age_score = 0.0
    if total_age > 0:
        for k in age_keys:
            share = (pop_summary.get(k, 0) or 0) / total_age
            age_score += share * profile["age"].get(k, 0.5)

    time_keys = [
        "시간대_00~06", "시간대_06~11", "시간대_11~14",
        "시간대_14~17", "시간대_17~21", "시간대_21~24",
    ]
    total_time = sum((pop_summary.get(k, 0) or 0) for k in time_keys)
    time_score = 0.0
    if total_time > 0:
        for k in time_keys:
            share = (pop_summary.get(k, 0) or 0) / total_time
            time_score += share * profile["time"].get(k, 0.5)

    resident_factor = 1.0
    if resident_population and resident_population.get("total_population"):
        total_floating = pop_summary.get("총_유동인구", 0) or 0
        resident_count = resident_population["total_population"]
        if total_floating > 0:
            resident_ratio = resident_count / (resident_count + total_floating)
            pref = profile.get("resident_pref", 0.5)
            resident_factor = 0.7 + 0.6 * (resident_ratio * pref + (1 - resident_ratio) * (1 - pref))

    return (age_score * 0.5 + time_score * 0.5) * resident_factor


def recommend_categories_by_demand(pop_summary, store_breakdown, resident_population=None, top_n=8):
    """
    미리 정의된 업종별 수요 특성표와 실제 이 지역 인구 데이터를 비교해서,
    '수요는 있는데 공급(기존 업소수 비중)이 부족한' 순서로 업종을 추천한다.
    단순히 '개수가 적은 업종'을 추천하던 기존 방식과 달리, 실제 인구 구성에
    기반한 수요-공급 격차(gap_score)를 기준으로 정렬한다.
    """
    if not store_breakdown:
        return []

    mid_counter = store_breakdown["mid_category"]
    total_stores = store_breakdown["total_count"] or 1

    results = []
    for category_name in CATEGORY_DEMAND_PROFILE:
        demand_score = compute_demand_score(category_name, pop_summary, resident_population)
        if demand_score is None:
            continue

        supply_count = _get_category_supply_count(mid_counter, category_name)
        supply_share = supply_count / total_stores

        results.append({
            "category": display_category_name(category_name),
            "demand_score": round(demand_score, 3),
            "supply_count": supply_count,
            "supply_share": round(supply_share, 3),
            "gap_score": round(demand_score - supply_share, 3),
        })

    results.sort(key=lambda x: x["gap_score"], reverse=True)
    return results[:top_n]


def estimate_rent(base_price_per_pyeong, area_pyeong, pop_summary, store_breakdown,
                   business_hours, category_name, category_level="mid",
                   cap_rate=None, floor_weight=1.0, estimated_sales_rows=None,
                   store_stats_rows=None):
    if cap_rate is None:
        cap_rate = CONFIG["CAP_RATE"]

    base_sale_value = base_price_per_pyeong * area_pyeong
    base_annual_rent = base_sale_value * cap_rate
    base_monthly_rent = base_annual_rent / 12

    pop_score = compute_population_fit_score(pop_summary, business_hours)
    sat_score = compute_saturation_score(store_breakdown, category_name, category_level)

    adjusted_monthly_rent = base_monthly_rent * pop_score * sat_score * floor_weight

    result = {
        "기준_평당매매가_만원": base_price_per_pyeong,
        "기준_월임대료_만원": round(base_monthly_rent, 1),
        "유동인구_적합도": round(pop_score, 2),
        "경쟁포화도_점수": round(sat_score, 2),
        "층수가중치": floor_weight,
        "최종_추천_월임대료_만원": round(adjusted_monthly_rent, 1),
        "예상_매출": None,
        "폐업_위험도": None,
    }

    # 이 상권·업종의 카드매출 기반 추정 매출이 있으면 함께 반환해서, 임차인이
    # 이 임대료를 감당 가능한 수준인지(매출 대비 임대료 비율) 참고할 수 있게 한다.
    # display_category_name 표를 거쳐도 실제 정부 업종명(예: '한식음식점')과는
    # 표기가 다를 수 있어 category_name 원문으로 부분일치 매칭을 시도한다.
    sales_summary = summarize_estimated_sales(estimated_sales_rows, category_name) if estimated_sales_rows else None
    if sales_summary and sales_summary.get("당월_추정매출_원"):
        monthly_sales_manwon = sales_summary["당월_추정매출_원"] / 10000
        rent_to_sales_ratio = None
        if monthly_sales_manwon > 0:
            rent_to_sales_ratio = round((adjusted_monthly_rent / monthly_sales_manwon) * 100, 1)
        result["예상_매출"] = {
            "매칭된_업종명": sales_summary["업종명"],
            "당월_추정매출_만원": round(monthly_sales_manwon, 1),
            "당월_추정매출_건수": sales_summary["당월_추정매출_건수"],
            "임대료_매출_비율_퍼센트": rent_to_sales_ratio,
            "참고": "카드매출 데이터 기반 추정치이며 실제 매출과 다를 수 있음",
        }

    # 이 업종의 점포 수·프랜차이즈 비율·개업률·폐업률 정보가 있으면 함께 반환해서,
    # 임차인을 들여도 되는 업종인지(생존 가능성) 임대인이 참고할 수 있게 한다.
    store_stats_summary = summarize_store_stats(store_stats_rows, category_name) if store_stats_rows else None
    if store_stats_summary:
        result["폐업_위험도"] = {
            "매칭된_업종명": store_stats_summary.get("업종명"),
            "폐업률": store_stats_summary.get("폐업률"),
            "개업률": store_stats_summary.get("개업률"),
            "프랜차이즈_비율": store_stats_summary.get("프랜차이즈_비율"),
            "표본_점포수": store_stats_summary.get("표본_유사업종_점포수"),
            "참고": "분기별 카드사·통신사 데이터 기반 추정치이며, 표본(점포수)이 적을수록 신뢰도가 낮음",
        }

    return result


def analyze_single_area(keyword, address, target_std_ym=None, radius_m=None, lawd_cd=None):
    """
    지역 키워드 하나를 받아서 좌표, 상권코드, 유동인구, 업종분포, (가능하면) 평균 매매가·정주인구까지
    한 번에 조회하는 챗봇용 함수.
    lawd_cd: 법정동코드 5자리(호출부가 넘겨줄 수 있음). 단, 주소/키워드 안에서
    서울 자치구 이름이 확인되면 검증된 SEOUL_GU_LAWD_CD 표의 코드를 우선 사용해
    잘못된 추측값을 덮어쓴다. (매매실거래가는 구 단위로만 제공되어 이 코드를 그대로 사용)

    정주인구는 가능하면 '동' 단위로 조회한다. 우선순위:
    1) 지오코딩 결과에 실제로 찍힌 좌표의 법정동명(가장 정확, 랜드마크 검색에도 동작)
    2) 그래도 못 찾으면 주소/키워드 문자열 안에 동 이름이 그대로 들어있는 경우
    3) 그것도 안 되면 구 단위로 대체
    """
    if radius_m is None:
        radius_m = CONFIG["DEFAULT_RADIUS_M"]
    if target_std_ym is None:
        # 예전에는 "20241"(2024년 1분기)로 고정돼 있었다. 이제는 실제로 데이터가
        # 올라와 있는 가장 최근 분기를 자동으로 찾아서 쓴다.
        from foot_traffic import find_latest_quarter
        target_std_ym = find_latest_quarter() or "20241"
    verified_lawd_cd = find_lawd_cd_by_gu_name(address) or find_lawd_cd_by_gu_name(keyword)
    if verified_lawd_cd:
        lawd_cd = verified_lawd_cd

    result = {
        "keyword": keyword,
        "geo": None,
        "trdar_cd": None,
        "trdar_cd_nm": None,
        "population": {},
        "population_trend": None,
        "estimated_sales": None,
        "store_stats": None,
        "transit_accessibility": None,
        "store_breakdown": None,
        "price_info": None,
        "price_trend": None,
        "resident_population": None,
        "resident_population_level": None,
        "demand_recommendations": None,
        "std_ym": target_std_ym,
        "error": None,
    }

    # keyword(역명/랜드마크명 등 잘 알려진 지명)를 먼저 지오코딩 시도하고,
    # 실패할 때만 address(Claude가 추측해서 채운 행정구역 텍스트)로 대체한다.
    # address만 쓰면, Claude가 추측한 주소가 틀렸을 때(예: "까치산역"인데 엉뚱한
    # 동으로 추측) 전혀 다른 위치가 나오는 문제가 있었다.
    geo = None
    geocode_error = None
    for candidate in [keyword, address]:
        if not candidate:
            continue
        try:
            geo = geocode_naver(candidate)
            break
        except Exception as e:
            geocode_error = e
            continue

    if not geo:
        result["error"] = "지오코딩 실패 [" + (type(geocode_error).__name__ if geocode_error else "Unknown") + "]: " + str(geocode_error)
        return result

    result["geo"] = geo

    # 1) 지오코딩이 알려준 실제 법정동명으로 우선 매칭 시도
    dong_code, exact_match = find_dong_code_by_emd_name(geo.get("dong_name"))

    if dong_code:
        result["resident_population_level"] = "동 단위" if exact_match else "동 단위(근사)"
    else:
        # 2) 주소/키워드 문자열 자체에 동 이름이 포함된 경우로 대체 시도
        dong_code = find_dong_code_by_name(address) or find_dong_code_by_name(keyword)
        result["resident_population_level"] = "동 단위" if dong_code else "구 단위"

    resident_pop_code = dong_code or lawd_cd

    areas = find_trade_area_codes(keyword)
    if areas:
        priority_order = {"발달상권": 0, "전통시장": 1, "골목상권": 2, "관광특구": 3}
        areas_sorted = sorted(areas, key=lambda a: priority_order.get(a["trdar_se_cd_nm"], 99))
        primary = areas_sorted[0]
        result["trade_area_match_method"] = "키워드 매칭"
    else:
        # 키워드로 매칭되는 상권이 없는 경우(순수 건물 주소 등) -> 좌표 기준 최근접 상권으로 대체
        nearest = find_nearest_trade_area(geo["lat"], geo["lng"])
        if not nearest:
            result["error"] = "상권을 찾지 못했습니다: " + keyword
            return result
        primary = nearest
        result["trade_area_match_method"] = (
            "좌표 기준 최근접 상권(약 " + str(nearest.get("distance_m")) + "m) - 키워드 '" + keyword + "'로 직접 매칭되는 상권 없음"
        )

    result["trdar_cd"] = primary["trdar_cd"]
    result["trdar_cd_nm"] = primary["trdar_cd_nm"]

    from concurrent.futures import ThreadPoolExecutor

    tasks = {}
    with ThreadPoolExecutor(max_workers=7) as executor:
        tasks["population"] = executor.submit(get_floating_population, primary["trdar_cd"], target_std_ym)
        tasks["population_trend"] = executor.submit(get_population_trend, primary["trdar_cd"])
        tasks["estimated_sales"] = executor.submit(
            get_estimated_sales, primary["trdar_cd"], target_std_ym, geo.get("dong_name")
        )
        tasks["store_stats"] = executor.submit(get_store_stats, primary["trdar_cd"], target_std_ym)
        tasks["store_breakdown"] = executor.submit(get_store_breakdown_in_radius, geo["lat"], geo["lng"], radius_m)
        tasks["transit_accessibility"] = executor.submit(get_transit_accessibility, geo["lat"], geo["lng"])
        if lawd_cd:
            # price_trend 하나만 조회하고, price_info(요약)는 그 결과에서 뽑아 쓴다.
            # (예전엔 get_average_price_recent를 따로 또 호출해서 같은 API를 중복 조회했음)
            tasks["price_trend"] = executor.submit(get_price_trend, lawd_cd, geo.get("dong_name"))
        if resident_pop_code:
            tasks["resident_population"] = executor.submit(get_resident_population, resident_pop_code)

        pop_rows = tasks["population"].result()
        result["population"] = summarize_population(pop_rows)
        result["population_trend"] = tasks["population_trend"].result()
        result["estimated_sales"] = tasks["estimated_sales"].result()
        result["store_stats"] = tasks["store_stats"].result()
        result["store_breakdown"] = tasks["store_breakdown"].result()
        result["transit_accessibility"] = tasks["transit_accessibility"].result()
        if lawd_cd:
            result["price_trend"] = tasks["price_trend"].result()
            result["price_info"] = derive_price_info_from_trend(result["price_trend"])
        if resident_pop_code:
            result["resident_population"] = tasks["resident_population"].result()

    # 추정매출·점포통계는 유동인구보다 한 분기 늦게 올라오는 경우가 있어서, 비어 있으면 직전 분기로 재시도
    from foot_traffic import previous_quarter
    if not result["estimated_sales"]:
        result["estimated_sales"] = get_estimated_sales(
            primary["trdar_cd"], previous_quarter(target_std_ym), geo.get("dong_name")
        )
    if not result["store_stats"]:
        result["store_stats"] = get_store_stats(primary["trdar_cd"], previous_quarter(target_std_ym))

    result["demand_recommendations"] = recommend_categories_by_demand(
        result["population"], result["store_breakdown"], result["resident_population"]
    )

    return result


def store_breakdown_to_plain_dict(store_breakdown):
    """
    Counter 객체가 들어있는 store_breakdown을 JSON으로 바꿀 수 있는
    평범한 dict/list 형태로 변환 (챗봇 도구 응답용).
    업종명은 사람이 읽기 쉬운 이름(예: 비알코올 -> 카페)으로 바꿔서 내보낸다.
    """
    if not store_breakdown:
        return None

    large_top = [(display_category_name(name), count)
                 for name, count in store_breakdown["large_category"].most_common(10)]
    mid_top = [(display_category_name(name), count)
               for name, count in store_breakdown["mid_category"].most_common(15)]

    return {
        "total_count": store_breakdown["total_count"],
        "large_category_top10": large_top,
        "mid_category_top15": mid_top,
    }


def compare_trade_areas(target_std_ym="20241"):
    targets = {
        "청량리역 상권": {"keyword": "청량리역", "address": "서울특별시 동대문구 청량리동"},
        "경동시장 상권": {"keyword": "경동시장", "address": "서울특별시 동대문구 제기동 620"},
    }

    results = {}

    for label, target_info in targets.items():
        keyword = target_info["keyword"]
        address = target_info["address"]
        print("\n=== " + label + " 분석 시작 (검색 키워드: '" + keyword + "') ===")

        area_result = analyze_single_area(keyword, address, target_std_ym)

        if area_result["error"]:
            print("[경고]", area_result["error"])

        if area_result["geo"]:
            print("좌표:", area_result["geo"])

        if not area_result["trdar_cd"]:
            results[label] = None
            continue

        print("매칭된 상권:", area_result["trdar_cd_nm"], "[" + str(area_result["trdar_cd"]) + "]")

        results[label] = {
            "trdar_cd": area_result["trdar_cd"],
            "trdar_cd_nm": area_result["trdar_cd_nm"],
            "population": area_result["population"],
            "store_breakdown": area_result["store_breakdown"],
        }

    print("\n\n================ 최종 비교 ================")
    for label, r in results.items():
        print("\n[" + label + "]")
        if not r:
            print("  데이터 없음")
            continue

        print("  상권명: " + r["trdar_cd_nm"] + " (" + str(r["trdar_cd"]) + ")")

        sb = r.get("store_breakdown")
        if sb:
            print("  반경500m 전체 업소수:", sb["total_count"])
            print("  --- 업종 대분류 TOP 10 ---")
            for name, count in sb["large_category"].most_common(10):
                print("    " + name + ":", count)
            print("  --- 업종 중분류 TOP 15 ---")
            for name, count in sb["mid_category"].most_common(15):
                print("    " + name + ":", count)
        else:
            print("  업소 데이터 없음")

        print("  --- 유동인구 ---")
        for k, v in r["population"].items():
            print("  " + k + ":", v)

    return results


if __name__ == "__main__":
    compare_trade_areas(target_std_ym="20241")