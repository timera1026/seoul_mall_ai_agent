"""
행정안전부 행정동별 주민등록 인구 및 세대현황 API 테스트 스크립트
================================================================
실행: python test_resident_population.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_GO_KR_KEY = os.getenv("DATA_GO_KR_KEY")


def test_resident_population(admm_cd, srch_fr_ym, srch_to_ym, lv=2, response_type="json"):
    url = "https://apis.data.go.kr/1741000/admmPpltnHhStus"
    params = {
        "serviceKey": DATA_GO_KR_KEY,
        "admmCd": admm_cd,
        "srchFrYm": srch_fr_ym,
        "srchToYm": srch_to_ym,
        "lv": lv,
        "regSeCd": 1,
        "type": response_type,
        "numOfRows": 10,
        "pageNo": 1,
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
        return

    print("=== 최상위 키 ===")
    print(list(data.keys()))

    # 공공데이터포털 표준 응답 구조 시도
    body = data.get("body")
    if body is None:
        response = data.get("response", {})
        body = response.get("body", {})

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
        print("[안내] items가 비어있습니다. admmCd나 조회기간을 다시 확인해보세요.")
        return

    print()
    print("=== 첫 번째 항목 필드명 ===")
    print(list(items[0].keys()))
    print()
    print("=== 첫 번째 항목 전체 내용 ===")
    print(items[0])
    print()
    print("총 항목 수:", len(items))


if __name__ == "__main__":
    if not DATA_GO_KR_KEY:
        print("[오류] .env에 DATA_GO_KR_KEY가 없습니다.")
    else:
        print("### 문서에 있던 예시값 그대로 (lv=4 기본값, type=XML) ###")
        test_resident_population(
            admm_cd="1111054000",
            srch_fr_ym="202210",
            srch_to_ym="202210",
            lv=4,
            response_type="xml",
        )
        print()
        print("### 같은 조건, type=json만 변경 ###")
        test_resident_population(
            admm_cd="1111054000",
            srch_fr_ym="202210",
            srch_to_ym="202210",
            lv=4,
            response_type="json",
        )