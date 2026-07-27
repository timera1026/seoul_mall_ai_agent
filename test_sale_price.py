"""
국토부 상업업무용 부동산 매매 실거래가 API 테스트 스크립트
=========================================================
실제 응답 구조와 필드명을 확인하기 위한 진단용 스크립트.
실행: python test_sale_price.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_GO_KR_KEY = os.getenv("DATA_GO_KR_KEY")


def test_sale_price_api(lawd_cd="11230", deal_ym="202412"):
    url = "https://apis.data.go.kr/1613000/RTMSDataSvcNrgTrade/getRTMSDataSvcNrgTrade"
    params = {
        "serviceKey": DATA_GO_KR_KEY,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": 1,
        "numOfRows": 100,
        "type": "json",
    }

    print("요청 URL:", url)
    print("요청 파라미터:", {k: v for k, v in params.items() if k != "serviceKey"})
    print()

    res = requests.get(url, params=params, timeout=20)
    print("HTTP 상태 코드:", res.status_code)
    print()
    print("=== 원본 응답 (앞부분 2000자) ===")
    print(res.text[:2000])
    print()

    try:
        data = res.json()
    except Exception as e:
        print("[오류] JSON 파싱 실패:", e)
        print("응답이 JSON이 아닐 수 있습니다 (XML 응답이거나 인증 오류 페이지일 가능성)")
        return

    print("=== 최상위 키 ===")
    print(list(data.keys()))

    body = data.get("body")
    if body is None:
        response = data.get("response", {})
        body = response.get("body", {})
        print("[안내] 'response' 안에 body가 있는 구조입니다.")

    print("=== body 내용 ===")
    print(body if not isinstance(body, dict) else list(body.keys()))

    items_container = body.get("items") if isinstance(body, dict) else None
    if isinstance(items_container, dict):
        items = items_container.get("item", [])
    else:
        items = items_container

    if isinstance(items, dict):
        items = [items]

    if not items:
        print("[안내] items가 비어있습니다. 다른 계약년월/지역코드로 시도해보세요.")
        return

    print()
    print("=== 첫 번째 거래 항목 필드명 ===")
    print(list(items[0].keys()))
    print()
    print("=== 첫 번째 거래 항목 전체 내용 ===")
    print(items[0])
    print()
    print("총 항목 수:", len(items))


if __name__ == "__main__":
    if not DATA_GO_KR_KEY:
        print("[오류] .env에 DATA_GO_KR_KEY가 없습니다.")
    else:
        test_sale_price_api(lawd_cd="11230", deal_ym="202412")