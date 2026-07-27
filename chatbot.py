"""
상권 분석 AI 챗봇
=================
compare_trade_area.py 에 있는 함수들을 "도구"로 등록해서,
자연어 질문에 Claude가 알아서 필요한 도구를 호출하고 답변하는 챗봇.

실행 전: pip install anthropic
.env에 ANTHROPIC_API_KEY 필요
"""

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

from compare_trade_area import (
    analyze_single_area,
    store_breakdown_to_plain_dict,
    get_commercial_sale_prices,
    estimate_price_per_pyeong,
    estimate_rent,
)

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_NAME = "claude-haiku-4-5-20251001"

TOOLS = [
    {
        "name": "analyze_trade_area",
        "description": (
            "특정 지역/상권의 유동인구(연령대별, 시간대별)와 반경 500m 내 "
            "업종별 업소 분포를 조회한다. 상권 분석 질문에 사용."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "상권 검색 키워드 (예: '청량리역', '경동시장')",
                },
                "address": {
                    "type": "string",
                    "description": "지오코딩용 정확한 주소 (예: '서울특별시 동대문구 청량리동')",
                },
            },
            "required": ["keyword", "address"],
        },
    },
    {
        "name": "estimate_shop_rent",
        "description": (
            "특정 상권, 특정 업종, 특정 평수의 상가에 대해 추천 월 임대료를 계산한다. "
            "먼저 analyze_trade_area로 그 지역 데이터를 조회한 뒤 사용해야 한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "상권 검색 키워드"},
                "address": {"type": "string", "description": "지오코딩용 주소"},
                "lawd_cd": {
                    "type": "string",
                    "description": "법정동코드 5자리 (예: 동대문구=11230)",
                },
                "deal_ym": {
                    "type": "string",
                    "description": "매매실거래가 조회할 계약년월 6자리 (예: 202412)",
                },
                "category_name": {
                    "type": "string",
                    "description": "업종 중분류명 (예: '한식', '주점', '기타 간이')",
                },
                "area_pyeong": {
                    "type": "number",
                    "description": "상가 면적(평)",
                },
                "business_hours": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "해당 업종의 주 영업시간대. 다음 중 선택: "
                        "시간대_00~06, 시간대_06~11, 시간대_11~14, "
                        "시간대_14~17, 시간대_17~21, 시간대_21~24"
                    ),
                },
                "floor_weight": {
                    "type": "number",
                    "description": "층수 가중치 (1층=1.0, 2층=0.7, 지하=0.5 등, 기본값 1.0)",
                },
            },
            "required": ["keyword", "address", "lawd_cd", "deal_ym", "category_name", "area_pyeong", "business_hours"],
        },
    },
]


def run_tool(tool_name, tool_input):
    if tool_name == "analyze_trade_area":
        result = analyze_single_area(tool_input["keyword"], tool_input["address"])
        plain = {
            "keyword": result["keyword"],
            "trdar_cd_nm": result["trdar_cd_nm"],
            "population": result["population"],
            "store_breakdown": store_breakdown_to_plain_dict(result["store_breakdown"]),
            "error": result["error"],
        }
        return plain

    if tool_name == "estimate_shop_rent":
        area_result = analyze_single_area(tool_input["keyword"], tool_input["address"])
        if area_result["error"]:
            return {"error": area_result["error"]}

        sale_items = get_commercial_sale_prices(tool_input["lawd_cd"], tool_input["deal_ym"])
        price_per_pyeong = estimate_price_per_pyeong(sale_items)

        if price_per_pyeong is None:
            return {"error": "해당 지역/기간의 매매실거래가 데이터를 찾을 수 없습니다."}

        rent_result = estimate_rent(
            base_price_per_pyeong=price_per_pyeong,
            area_pyeong=tool_input["area_pyeong"],
            pop_summary=area_result["population"],
            store_breakdown=area_result["store_breakdown"],
            business_hours=tool_input["business_hours"],
            category_name=tool_input["category_name"],
            category_level="mid",
            cap_rate=0.05,
            floor_weight=tool_input.get("floor_weight", 1.0),
        )
        return rent_result

    return {"error": "알 수 없는 도구: " + tool_name}


def chat_loop():
    print("=== 상권 분석 챗봇 ===")
    print("예시 질문: '청량리역 상권 유동인구 알려줘', '경동시장에 한식당 20평 내면 임대료 얼마가 적당해?'")
    print("종료하려면 'exit' 입력\n")

    messages = []

    while True:
        user_input = input("나: ")
        if user_input.strip().lower() == "exit":
            print("챗봇을 종료합니다.")
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1500,
                tools=TOOLS,
                messages=messages,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            for t in text_blocks:
                print("\n챗봇:", t.text)

            if not tool_use_blocks:
                assistant_content = []
                for b in response.content:
                    if b.type == "text":
                        assistant_content.append({"type": "text", "text": b.text})
                messages.append({"role": "assistant", "content": assistant_content})
                break

            assistant_content = []
            for b in response.content:
                if b.type == "text":
                    assistant_content.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for tool_use in tool_use_blocks:
                print("  [도구 호출]", tool_use.name, tool_use.input)
                result = run_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            messages.append({"role": "user", "content": tool_results})

        print()


if __name__ == "__main__":
    chat_loop()