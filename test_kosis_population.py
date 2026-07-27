"""
KOSIS 주민등록인구현황 API 테스트 스크립트
==========================================
실행: python test_kosis_population.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")


def test_kosis_population():
    url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    params = {
        "method": "getList",
        "apiKey": KOSIS_API_KEY,
        "itmId": "T2",  # 총인구수
        "objL1": "ALL",  # 전체 지역, 이름으로 나중에 필터링
        "objL2": "0",
        "objL3": "",
        "objL4": "",
        "objL5": "",
        "objL6": "",
        "objL7": "",
        "objL8": "",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": "202412",
        "endPrdDe": "202412",
        "orgId": "101",
        "tblId": "DT_1B04005N",
    }

    print("요청 URL:", url)
    print()

    res = requests.get(url, params=params, timeout=20)
    print("HTTP 상태 코드:", res.status_code)
    print()
    print("=== 원본 응답 (앞부분 1000자) ===")
    print(res.text[:1000])
    print()

    try:
        data = res.json()
    except Exception as e:
        print("[오류] JSON 파싱 실패:", e)
        return

    if isinstance(data, dict):
        print("=== 오류 응답 ===")
        print(data)
        return

    print("총 항목 수:", len(data))
    print()
    print("=== 첫 번째 항목 필드명 ===")
    print(list(data[0].keys()))
    print()

    print("=== '동대문' 포함 지역 검색 결과 ===")
    found = [d for d in data if "동대문" in d.get("C1_NM", "")]
    for d in found:
        print(d)


def test_kosis_population_dong(objl1_code):
    url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    params = {
        "method": "getList",
        "apiKey": KOSIS_API_KEY,
        "itmId": "T2",
        "objL1": objl1_code,
        "objL2": "0",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": "202412",
        "endPrdDe": "202412",
        "orgId": "101",
        "tblId": "DT_1B04005N",
    }
    res = requests.get(url, params=params, timeout=20)
    print("HTTP 상태 코드:", res.status_code)
    print(res.text[:1000])


if __name__ == "__main__":
    if not KOSIS_API_KEY:
        print("[오류] .env에 KOSIS_API_KEY가 없습니다.")
    else:
        print("### 1. 구 단위(11230) ###")
        test_kosis_population()

        print()
        print("### 2. 동 단위(청량리동=1123070500)로 시도 ###")
        test_kosis_population_dong("1123070500")