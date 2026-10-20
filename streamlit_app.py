"""
상권 분석 AI 챗봇 - 웹 버전 (Streamlit)
=======================================
실행: streamlit run streamlit_app.py
배포: Streamlit Community Cloud (streamlit.io) 무료 호스팅
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic, AuthenticationError

load_dotenv()

from compare_trade_area import (
    analyze_single_area,
    store_breakdown_to_plain_dict,
    get_commercial_sale_prices,
    estimate_price_per_pyeong,
    estimate_rent,
    get_building_info,
    get_static_map_image,
    summarize_estimated_sales,
    summarize_store_stats,
    summarize_population,
    geocode_naver,
    find_trade_area_codes,
    find_nearest_trade_area,
    CONFIG,
)
from location_recommender import recommend_locations
from profit_model import predict_listing_profit
import sales_model
from foot_traffic import (
    analyze_foot_traffic,
    estimate_target_customers,
    inspect_new_apis,
)

MODEL_NAME = "claude-sonnet-4-5"

SYSTEM_PROMPT = (
    "당신은 건설사/디벨로퍼(임대인) 내부 실무자를 돕는 상권·임대 전략 전문가입니다. "
    "이 도구를 쓰는 사람은 상가를 소유하고 임차인을 유치해야 하는 임대인 측 담당자이며, "
    "목표는 '어떤 임대료와 업종 구성이 임차인에게 매력적으로 보여서 공실 없이 빠르게 "
    "채울 수 있는가'를 판단하는 것입니다. 임차인에게 직접 안내하듯 말하지 말고, "
    "항상 임대인이 의사결정에 참고할 수 있는 내부 분석 보고서처럼 답변하세요.\n\n"
    "도구 호출 결과를 받으면, 숫자를 나열만 하지 말고 다음을 반드시 포함해 상세히 설명하세요:\n"
    "1) 핵심 수치 요약 (총 유동인구, 정주인구/등록인구, 주요 연령대, 피크 시간대, 업종 경쟁 상황)\n"
    "2) 그 수치가 이 호실/상가의 임차인 유치 경쟁력에 어떤 의미인지 해석. "
    "특히 유동인구 대비 정주인구(resident_population) 비율이 어떤지도 함께 짚어서, "
    "이 상권이 '스쳐 지나가는 유동인구 의존형'인지 '실거주 배후수요가 탄탄한 곳'인지 판단해 설명하세요.\n"
    "3) 업종 추천 시 절대 '업소 개수가 적어서 추천'이라는 논리만으로 말하지 마세요. "
    "analyze_trade_area 결과의 demand_based_recommendations 배열을 반드시 활용하세요. "
    "이 배열의 각 항목은 demand_score(연령대·시간대·정주인구 특성을 반영한 예상 수요), "
    "supply_share(현재 그 업종의 실제 공급 비중 = 경쟁강도), gap_score(수요-공급 격차, "
    "클수록 유망)를 담고 있습니다. 표를 만들 때 supply_share를 '현재 경쟁강도 OO%' 형태로 "
    "반드시 함께 표시해 신뢰도를 높이세요. gap_score 상위 항목들을 근거로 업종을 추천하되, "
    "**표는 상위 4~5개까지만** 다루고 각 근거는 한 문장 이내로 간결하게 쓰세요 (응답이 길어지면 "
    "잘릴 수 있으므로 간결함이 중요합니다). 왜 그 업종의 수요가 이 지역에서 클 것으로 보이는지"
    "(연령대/시간대/정주인구 특성과 연결해서) 설명하되, 이 수요 점수는 실제 매출 데이터가 아니라 "
    "일반적 소비패턴 기반 추정치라는 점은 표 아래 한 줄로만 짧게 언급하세요.\n"
    "4) analyze_trade_area 결과의 average_sale_price 안에 평당 평균 매매가와 "
    "평당 월 임대료 시세(avg_monthly_rent_per_pyeong_manwon)가 있다면, 이 숫자를 반드시 "
    "명시적으로 언급하고, 대표적인 평수(예: 10평, 20평) 기준으로 '월 임대료 약 OOO만원 수준' "
    "처럼 구체적인 예상 총액도 계산해서 제시하세요. average_sale_price가 null이거나 데이터가 "
    "없다면 '이 지역은 최근 매매 실거래 데이터가 부족해 시세 추정이 어렵다'고 명확히 밝히세요. "
    "절대 임대료 관련 수치를 언급 없이 넘어가지 마세요. price_trend 배열(최근 수개월 평당 매매가 "
    "추이)이 있으면, 그래프는 화면에 자동으로 표시되니 텍스트에서는 '최근 상승세/하락세/보합세'인지 "
    "한두 문장으로만 짧게 해석해 주세요. 데이터가 부족한 달은 결측일 수 있다는 점도 참고하세요.\n"
    "4-1) population_trend 배열(최근 분기별 총 유동인구 추이)이 있으면, 이 상권이 최근 뜨는 "
    "중인지 지는 중인지를 반드시 짚어주세요. 가장 오래된 분기 대비 가장 최근 분기의 증감률을 "
    "계산해서 '최근 O개 분기간 유동인구 OO% 증가/감소' 형태로 명시하고, 이게 임차인 유치 "
    "경쟁력 판단에 왜 중요한지(예: 유동인구가 꾸준히 느는 지역은 향후 임대료 인상 여력이 있고, "
    "줄고 있다면 공실 리스크 신호일 수 있음) 한두 문장으로 짧게 해석하세요. 그래프는 화면에 "
    "자동 표시되니 텍스트에서는 해석만 간결하게 하세요. 분기가 2개 미만이면 추세를 논하지 말고 "
    "'추세 판단에 충분한 분기 데이터가 없다'고만 밝히세요.\n"
    "5) estimate_shop_rent 결과가 있다면, 그 임대료 수준이 임차인 입장에서 왜 합리적/매력적인지, "
    "그리고 임대인 입장에서는 이 가격으로 얼마나 빨리 임차인을 유치할 수 있을지 함께 설명\n"
    "5-1) analyze_trade_area 결과의 estimated_sales_summary, 또는 estimate_shop_rent 결과의 "
    "예상_매출 필드가 있으면 반드시 활용하세요. 이건 카드매출 데이터 기반 '이 업종이 이 상권에서 "
    "실제로 올리는 추정 매출'입니다(demand_score 같은 휴리스틱 추정치와는 다른, 실측에 가까운 "
    "데이터). estimate_shop_rent의 예상_매출.임대료_매출_비율_퍼센트가 있으면 '월 매출 대비 "
    "임대료 비중 OO%' 형태로 명시하고, 일반적으로 임대료가 매출의 10~15%를 넘으면 임차인에게 "
    "부담스러운 수준이라는 점을 참고해 이 임대료가 임차인에게 감당 가능한 수준인지 판단해주세요. "
    "estimated_sales_summary만 있는 경우(특정 업종을 안 정했을 때)는 매출_상위5_업종을 통해 "
    "이 상권에서 어떤 업종이 실제로 잘 되고 있는지 짚어주세요. 이 매출 데이터는 카드매출 기반 "
    "추정치이며 현금 결제나 특수 상황은 반영되지 않을 수 있다는 점을 표 아래 한 줄로 짧게 "
    "언급하세요. 매출 데이터가 없는 지역(estimated_sales_summary가 null)이면 '이 지역은 "
    "서울시 상권분석서비스의 상권 범위에 포함되지 않아 추정매출 데이터를 제공하지 않는다'고만 "
    "짧게 밝히고 넘어가세요.\n"
    "5-2) analyze_trade_area 결과의 store_stats_summary, 또는 estimate_shop_rent 결과의 "
    "폐업_위험도 필드가 있으면 반드시 활용하세요. 이건 '공급이 부족해서 유망하다'는 gap_score "
    "추정과는 별개로, 실제 그 업종이 이 상권에서 얼마나 안정적으로 살아남는지(폐업률)를 보여주는 "
    "지표입니다. 폐업률이 높은 업종은 gap_score가 높더라도(공급 부족처럼 보여도) 실제로는 "
    "'들어와도 오래 못 버티는 업종'일 수 있으니, 이 두 지표를 함께 봐서 모순되는 경우(공급은 "
    "부족한데 폐업률도 높은 경우) 그 이유를 추정해 설명하세요(예: 초기 진입은 쉽지만 상권 "
    "특성상 매출이 안 나와 오래 못 버티는 업종일 수 있음). 폐업_위험도.표본_점포수가 3개 미만이면 "
    "'표본이 적어 폐업률 수치의 신뢰도가 낮다'고 반드시 짧게 밝히세요. store_stats_summary만 "
    "있는 경우(특정 업종 미지정)는 폐업률_상위5_업종을 통해 이 상권에서 특히 리스크가 큰 업종이 "
    "무엇인지 짚어주고, 프랜차이즈_비율이 높으면 '개인 창업보다 프랜차이즈 의존도가 높은 상권'이라는 "
    "점도 함께 언급하세요. 이 데이터는 분기별 카드사·통신사 기반 추정치이며 실제 사업자등록 "
    "통계와 집계 방식이 다를 수 있다는 점을 표 아래 한 줄로 짧게 언급하세요.\n"
    "5-3) analyze_trade_area 결과의 transit_accessibility 필드가 있으면 활용하세요. "
    "정류장_수와 총_경유노선_수_합산으로 도보 대중교통 접근성을 판단할 수 있습니다. "
    "이건 지하철 유동인구와는 별개 지표이므로, 지하철역 유동인구가 많더라도 버스 접근성이 "
    "낮으면(정류장이 1~2개뿐이거나 노선 수가 적으면) '역세권이지만 버스 배후 접근성은 약한 "
    "곳'처럼 구분해서 설명하세요. 반대로 지하철 유동인구는 적어도 버스 정류장·노선이 많으면 "
    "'버스 환승 거점형 상권'일 수 있다는 점도 짚어주세요. 정류장_수가 0이면 '반경 300m 내 "
    "버스정류장이 없어 대중교통 접근성이 매우 낮다'고 명확히 밝히세요. 총_경유노선_수_합산은 "
    "정류장 간 노선이 겹칠 수 있어 참고용 근사치라는 점을 표 아래 한 줄로 짧게 언급하세요. "
    "이 항목은 짧게 한두 문장으로만 다루고 과도하게 비중을 두지 마세요.\n"
    "5-4) analyze_foot_traffic 결과가 있으면 '사람이 언제, 누가, 왜 오는지'를 설명하세요. "
    "주중_하루평균과 주말_하루평균을 비교해 '평일 직장인형/주말 나들이형' 상권인지 판단하고, "
    "직장인구가 많으면 점심·퇴근 수요를, 집객시설(학교·병원·관공서 등)이 있으면 그 시설이 "
    "끌어오는 고객층을 짚어주세요. 지하철_승하차의 역_성격(출근 도착형/퇴근 귀가형)과 "
    "하차_피크_시간을 활용해 '몇 시에 이 동네로 사람이 들어오는지' 설명하세요. 지하철 데이터가 "
    "error면 '해당 역 승하차 데이터를 가져오지 못했다'고만 짧게 밝히세요.\n"
    "5-5) estimate_target_customers 결과가 있으면 '하루 관심고객 약 O명, 예상 방문객 하루 "
    "O~O명(전환율 1~4% 가정)' 형태로 반드시 숫자를 제시하세요. 이 숫자는 유동인구 분기합계를 "
    "하루로 나누고 방문 전환율을 가정한 추정치라는 점을 한 줄로 짧게 밝히세요.\n"
    "6) 필요하다면 임대료 조정안(할인/인센티브)이나 업종 전환 시나리오도 제안\n"
)

FORMAT_RULES = (
    "공통) 사용자가 특정 건물 주소를 주면서 그 건물 자체를 물어보면 analyze_building 도구도 "
    "함께 호출하세요. 이 도구 결과(연면적, 대지면적, 건폐율/용적률, 지상·지하 층수, 승강기 수, "
    "준공일, 주용도, 구조)가 있으면 답변에 반드시 반영하고, error가 있으면 '이 주소는 건축물대장 "
    "조회에 실패했다'고 명확히 밝히세요. 이 도구는 상권/유동인구 분석과는 별개이므로, 상권 질문과 "
    "함께 나오면 analyze_trade_area도 같이 호출하세요.\n"
    "표 안에서 한 셀에 여러 줄을 적어야 할 때도 '<br>' 같은 HTML 태그는 절대 쓰지 마세요. "
    "대신 같은 셀 안에서는 가운뎃점(·)이나 쉼표로 구분해 한 줄로 쓰거나, "
    "필요하면 표 대신 글머리 기호(-) 목록으로 표현하세요. 답변 전체에서 HTML 태그를 "
    "일절 사용하지 말고 순수 마크다운 문법만 사용하세요.\n"
    "마크다운 표를 쓸 때는 반드시 다음 형식을 정확히 지키세요: 헤더 행, 구분선 행(|---|---|...), "
    "데이터 행 각각에서 '|'로 나뉘는 칸의 개수가 컬럼 개수와 정확히 똑같아야 합니다. "
    "예를 들어 컬럼이 5개(업종, 수요점수, 경쟁강도, 갭, 근거)라면 구분선도 반드시 "
    "'|------|------|------|------|------|'처럼 5칸으로 맞추세요. 구분선 칸 수가 헤더나 "
    "데이터 행보다 적으면 표가 아예 깨져서 렌더링되지 않으니, 표를 쓰기 전에 각 행의 '|' 개수를 "
    "스스로 세어서 모두 동일한지 확인한 뒤 출력하세요.\n"
    "표나 목록 형태를 적극적으로 활용해 가독성 있게 답변하세요. "
    "그래프/차트는 화면에 별도로 자동 표시되니, 텍스트에서는 해석과 전략에 집중하세요. "
    "전체 답변은 각 항목을 핵심 위주로 간결하게 정리하고, 불필요하게 늘어지지 않게 하세요 "
    "(응답 길이 제한으로 문장이 중간에 잘리는 것을 방지하기 위함입니다)."
)

SYSTEM_PROMPT_LANDLORD = SYSTEM_PROMPT + FORMAT_RULES

SYSTEM_PROMPT_FOUNDER = (
    "당신은 처음 가게를 열려는 개인 예비 창업자를 돕는 친절한 상권 분석 상담가입니다. "
    "사용자는 상권 분석이나 부동산 용어를 잘 모르는 일반인이라고 생각하고, 전문용어를 쓰면 "
    "괄호로 쉽게 풀어서 설명하세요. 말투는 존댓말로 따뜻하지만 군더더기 없이, 결론을 먼저 말하세요. "
    "숫자는 모두 공공데이터 기반 추정치이므로 '확정'처럼 말하지 말고, 그렇다고 겁을 주지도 마세요.\n\n"
    "질문 유형별로 다음처럼 도구를 쓰세요.\n"
    "A) '이 위치(주소/역)에서 뭘 하면 좋을까?' 같은 위치 기반 질문: analyze_trade_area와 "
    "analyze_foot_traffic을 함께 호출하세요. 그 다음 demand_based_recommendations의 gap_score 상위 "
    "업종 중 2~3개를 골라 각각 estimate_target_customers를 호출해 업종별 관심고객 수를 구하세요. "
    "답변은 ① 이 동네 한 줄 요약(누가, 언제 오는 곳인지) ② 추천 업종 표(업종, 추천 이유 한 문장, "
    "하루 관심고객 수, 예상 방문객 범위, 현재 경쟁 정도) ③ 피해야 할 업종(폐업률이 높은 업종) 순서로 쓰세요.\n"
    "B) '카페 하고 싶은데 어디가 좋을까?' 같은 업종 기반 질문: recommend_locations_for_business를 "
    "호출하세요. 사용자가 원하는 지역(구)을 말했으면 gu에 넣으세요. 답변은 상위 5곳 표(순위, 상권명, "
    "자치구, 점수, 점포당 월매출 추정, 서울평균 대비 지수, 점포 수, 폐업률, 하루 유동인구)와 각 상권을 추천하는 이유 한 문장을 쓰고, "
    "서울평균_대비_매출지수로 상위 상권이 서울 평균 가게보다 몇 배 버는지 짚어주세요. "
    "추천 상권에 진입시_예측_월매출_만원이 있으면 '내가 들어갔을 때 예상 월매출 O~O만원(중앙 O만원)'도 표에 넣고, "
    "예측모델_검증의 오차 중앙값을 한 줄로 밝히세요. "
    "점수가 높아도 점포당 매출이 높은 곳은 보통 임대료도 비싸다는 점을 한 줄로 알려주세요. "
    "결과의 error에 선택_가능한_업종 목록이 있으면 그중 가장 가까운 업종으로 다시 호출하세요. "
    "1위 상권에 대해서는 analyze_foot_traffic을 추가로 호출해 언제 사람이 많은지도 짧게 설명하면 좋습니다.\n"
    "C) 특정 매물(보증금, 월세, 평수 등)을 주며 '얼마나 남을까?'를 묻는 질문: predict_listing_profit을 "
    "호출하세요. 업종이나 월세·평수 중 빠진 게 있으면 추측하지 말고 먼저 짧게 물어보세요(보증금·관리비·권리금·층은 "
    "모르면 비워도 됨). 답변은 ① 판정 한 줄 ② 시나리오 표(보수/보통/낙관: 월매출, 월비용, 월순이익) "
    "③ 보통 시나리오 비용 내역에서 큰 항목 3~4개 ④ 손익분기 월매출과 손익분기 하루 손님 수 "
    "⑤ 임대료_매출_비율이 15%를 넘으면 경고 ⑥ 초기투자 합계와 투자금 회수 기간 순서로 쓰세요. "
    "매출_예측_모델이 있으면 시나리오는 학습·검증된 예측 범위(하위10%~상위10%)이므로 '실제 매출이 이 범위에 들어갈 "
    "확률 약 80%'라고 설명하고, 모델_검증의 오차 중앙값(예: '이 모델은 보통 ±20% 정도 틀립니다')을 한 줄로 반드시 "
    "밝히세요. 매출_예측_모델이 null이면 '예측 모델을 아직 쓸 수 없어 동네 평균으로 대신 계산했다'고 밝히세요. "
    "신뢰도가 '낮음'이면 그 이유도 짚으세요. 층수·면적 보정과 비용 비율은 데이터가 아닌 가정값이라는 점도 짧게 알려주세요. "
    "검산_관심고객_기준의 월매출이 보통 시나리오와 2배 이상 차이 나면 그 점도 짚으세요. 순이익은 세금 전이고 사장 본인 인건비가 빠진 금액이라는 "
    "점을 반드시 한 줄로 알려주세요. 사용자가 위치는 정했지만 업종을 못 정했다면 A) 흐름으로 추천 업종 2~3개를 "
    "고른 뒤 각각 predict_listing_profit을 호출해 비교표를 만들어 주세요.\n\n"
    "도구 결과 해석 원칙:\n"
    "- estimate_target_customers 결과는 '하루 관심고객 약 O명, 하루 예상 방문객 O~O명(전환율 1~4% 가정)'처럼 "
    "반드시 숫자로 말하세요.\n"
    "- analyze_foot_traffic 결과로 평일형/주말형, 출근 도착형/퇴근 귀가형 같은 동네 성격을 쉬운 말로 설명하세요.\n"
    "- 폐업률 표본(점포 수)이 3개 미만이면 신뢰도가 낮다고 짧게 밝히세요.\n"
    "- 답변 마지막에는 '직접 확인해 볼 것' 체크리스트를 2~3개 주세요 "
    "(예: 평일 점심·주말 저녁에 현장에 가서 사람 흐름 보기, 주변 같은 업종 가게 손님 수 세어보기, "
    "실제 임대료 시세 부동산에 물어보기).\n"
    "- 임대인(건물주) 입장의 말투나 '임차인 유치' 같은 표현은 쓰지 마세요.\n\n"
) + FORMAT_RULES



def get_api_key(key_name):
    """
    로컬 실행시에는 .env, Streamlit Cloud 배포시에는 st.secrets에서
    키를 읽어온다. 로컬에는 secrets.toml이 없을 수 있으므로 안전하게 처리.
    """
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name)


ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
NAVER_CLIENT_ID_FOR_MAP = get_api_key("NAVER_CLIENT_ID")

TOOLS = [
    {
        "name": "analyze_trade_area",
        "description": (
            "특정 지역/상권의 유동인구(연령대별, 시간대별), 반경 500m 내 "
            "업종별 업소 분포, 해당 법정동이 속한 시군구의 평균 상가(평당) 매매가, "
            "서울시 상권분석서비스의 카드매출 기반 업종별 추정매출(estimated_sales_summary), "
            "업종별 점포 수·프랜차이즈 비율·개업률·폐업률(store_stats_summary), 그리고 "
            "반경 300m 내 버스정류장 수·경유노선 수(transit_accessibility)까지 "
            "함께 조회한다. 상권 분석 질문에는 항상 이 도구를 사용하고, lawd_cd는 "
            "질문에 나온 지역이 속한 시군구의 법정동코드 5자리를 스스로 판단해서 채운다 "
            "(예: 서울 동대문구=11230, 서울 강남구=11680 등 널리 알려진 코드 지식을 활용)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "상권 검색 키워드 (예: '청량리역', '경동시장')"},
                "address": {"type": "string", "description": "지오코딩용 정확한 주소 (예: '서울특별시 동대문구 청량리동')"},
                "lawd_cd": {"type": "string", "description": "이 지역이 속한 시군구의 법정동코드 5자리"},
            },
            "required": ["keyword", "address", "lawd_cd"],
        },
    },
    {
        "name": "estimate_shop_rent",
        "description": (
            "특정 상권, 특정 업종, 특정 평수의 상가에 대해 추천 월 임대료를 계산한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "address": {"type": "string"},
                "lawd_cd": {"type": "string", "description": "법정동코드 5자리 (동대문구=11230)"},
                "deal_ym": {"type": "string", "description": "계약년월 6자리 (예: 202412)"},
                "category_name": {"type": "string", "description": "업종 중분류명 (예: '한식', '주점')"},
                "area_pyeong": {"type": "number"},
                "business_hours": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "시간대_00~06, 시간대_06~11, 시간대_11~14, 시간대_14~17, 시간대_17~21, 시간대_21~24 중 선택",
                },
                "floor_weight": {"type": "number", "description": "층수 가중치 (1층=1.0, 2층=0.7 등)"},
            },
            "required": ["keyword", "address", "lawd_cd", "deal_ym", "category_name", "area_pyeong", "business_hours"],
        },
    },
    {
        "name": "analyze_building",
        "description": (
            "특정 건물의 실제 건축물대장 정보(연면적, 대지면적, 건폐율/용적률, 지상·지하 층수, "
            "승강기 수, 준공일/사용승인일, 주용도, 구조 등)를 조회한다. "
            "사용자가 특정 건물 주소(도로명주소 또는 지번주소)를 주면서 '이 건물', '이 빌딩' "
            "등으로 건물 자체의 실제 스펙을 물어볼 때 사용한다. 상권/유동인구 분석과는 "
            "별개이므로, 건물 정보가 필요하면 analyze_trade_area와 함께 호출해도 된다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "건물의 정확한 도로명주소 또는 지번주소"},
            },
            "required": ["address"],
        },
    },
]


TIME_SLOT_ENUM = ["시간대_00~06", "시간대_06~11", "시간대_11~14", "시간대_14~17", "시간대_17~21", "시간대_21~24"]
AGE_ENUM = ["10대", "20대", "30대", "40대", "50대", "60대_이상"]
DAY_ENUM = ["월", "화", "수", "목", "금", "토", "일"]

TOOLS += [
    {
        "name": "analyze_foot_traffic",
        "description": (
            "특정 위치/상권의 '인적 동선'을 정밀하게 분석한다. 요일별 유동인구(주중·주말 하루평균), "
            "성별×연령별 유동인구, 직장인구, 집객시설(지하철역·학교·병원·관공서 수 등), "
            "그리고 주변 지하철역의 시간대별 하루평균 승·하차 인원과 역 성격(출근 도착형/퇴근 귀가형)을 "
            "조회한다. 유동인구가 '언제, 누가' 오는지 자세히 묻거나 창업 입지를 판단할 때 사용한다. "
            "station_names에는 그 위치에서 걸어갈 수 있는 가까운 지하철역 이름을 스스로 판단해 1~3개 넣는다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "상권 검색 키워드 (예: '청량리역')"},
                "address": {"type": "string", "description": "지오코딩용 주소 (예: '서울특별시 동대문구 청량리동')"},
                "station_names": {
                    "type": "array", "items": {"type": "string"},
                    "description": "근처 지하철역 이름 1~3개 (예: ['청량리', '회기'])",
                },
            },
            "required": ["keyword", "address"],
        },
    },
    {
        "name": "estimate_target_customers",
        "description": (
            "창업하려는 업종의 주 고객층(연령·성별)과 영업시간·영업요일을 받아서, 이 위치에서 "
            "하루에 지나가는 '관심고객 수'와 실제 매장에 들를 '예상 방문객 수'(전환율 1%/2%/4% 세 가지 "
            "시나리오)를 추정한다. 사용자가 '손님이 몇 명쯤 올까', '관심고객이 얼마나 될까'를 묻거나 "
            "특정 업종의 입지를 평가할 때 사용한다. 업종만 주어지면 그 업종의 일반적인 주 고객층과 "
            "영업시간을 스스로 판단해서 채운다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "address": {"type": "string"},
                "target_ages": {"type": "array", "items": {"type": "string", "enum": AGE_ENUM}},
                "target_gender": {"type": "string", "enum": ["전체", "남성", "여성"]},
                "business_hours": {"type": "array", "items": {"type": "string", "enum": TIME_SLOT_ENUM}},
                "open_days": {"type": "array", "items": {"type": "string", "enum": DAY_ENUM}},
            },
            "required": ["keyword", "address", "target_ages", "business_hours"],
        },
    },
]


TOOLS.append({
    "name": "recommend_locations_for_business",
    "description": (
        "창업하려는 업종 하나를 받아서, 서울 전체(또는 지정한 자치구) 상권 중 그 업종을 하기 좋은 곳을 "
        "점수 순으로 추천한다. 점수는 점포당 추정매출(45%), 주 고객층에 맞는 하루 유동인구(30%), "
        "낮은 폐업률(25%)로 계산한다. '카페 하려면 어디가 좋아?', '마포구에서 치킨집 하기 좋은 곳' 같은 "
        "업종 기반 위치 질문에 사용한다. 처음 호출할 때는 서울 전체 데이터를 불러와서 30초 정도 걸릴 수 있다."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "business": {"type": "string", "description": "업종 (예: '카페', '치킨', '미용실', '커피-음료')"},
            "gu": {"type": "string", "description": "범위를 좁힐 자치구. 여러 개면 쉼표로 (예: '마포구, 서대문구'). 없으면 서울 전체"},
            "target_ages": {"type": "array", "items": {"type": "string", "enum": AGE_ENUM},
                            "description": "사용자가 주 고객 연령대를 말한 경우에만 넣는다"},
            "top_n": {"type": "integer", "description": "추천 개수 (기본 10)"},
        },
        "required": ["business"],
    },
})


TOOLS.append({
    "name": "predict_listing_profit",
    "description": (
        "실제 상가 매물(위치, 평수, 월세, 보증금 등)에서 특정 업종을 할 때 한 달에 얼마가 남을지 예측한다. "
        "같은 상권의 같은 업종 점포 평균 매출을 층수·면적으로 보정해 보수/보통/낙관 3가지 월매출을 만들고, "
        "업종별 비용 구조(재료비·인건비·공과금·배달·카드수수료)와 월세·관리비를 빼서 월 순이익, "
        "손익분기 매출, 손익분기 하루 손님 수, 초기투자와 회수 기간을 계산한다. 금액 단위는 모두 만원."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "매물 근처 상권 키워드 (예: '망원역')"},
            "address": {"type": "string", "description": "매물 주소 또는 지오코딩용 주소"},
            "business": {"type": "string", "description": "하려는 업종 (예: '카페', '치킨')"},
            "area_pyeong": {"type": "number", "description": "전용면적(평). ㎡로 받으면 3.3으로 나눠서 넣는다"},
            "monthly_rent_manwon": {"type": "number", "description": "월세(만원)"},
            "deposit_manwon": {"type": "number", "description": "보증금(만원), 모르면 0"},
            "maintenance_manwon": {"type": "number", "description": "월 관리비(만원). 모르면 생략 (평당 1만원으로 가정)"},
            "premium_manwon": {"type": "number", "description": "권리금(만원), 없으면 0"},
            "floor": {"type": "integer", "description": "층 (1층=1, 2층=2, 지하1층=-1). 모르면 1"},
            "interior_manwon": {"type": "number", "description": "인테리어·설비 예산(만원). 모르면 생략 (업종 평균으로 추정)"},
            "open_days_per_week": {"type": "number", "description": "주 영업일 수. 모르면 6"},
        },
        "required": ["keyword", "address", "business", "area_pyeong", "monthly_rent_manwon"],
    },
})


def resolve_trade_area(keyword, address):
    """키워드/주소 -> (상권코드, 상권명, 좌표). analyze_single_area보다 가벼운 버전."""
    geo = None
    for candidate in [keyword, address]:
        if not candidate:
            continue
        try:
            geo = geocode_naver(candidate)
            break
        except Exception:
            continue
    areas = find_trade_area_codes(keyword) if keyword else []
    if areas:
        priority_order = {"발달상권": 0, "전통시장": 1, "골목상권": 2, "관광특구": 3}
        primary = sorted(areas, key=lambda a: priority_order.get(a["trdar_se_cd_nm"], 99))[0]
    elif geo:
        primary = find_nearest_trade_area(geo["lat"], geo["lng"])
    else:
        primary = None
    return primary, geo


def run_tool(tool_name, tool_input):
    if tool_name == "analyze_trade_area":
        result = analyze_single_area(
            tool_input["keyword"],
            tool_input["address"],
            lawd_cd=tool_input.get("lawd_cd"),
        )

        return {
            "keyword": result["keyword"],
            "trdar_cd_nm": result["trdar_cd_nm"],
            "population": result["population"],
            "population_trend": result.get("population_trend"),
            "estimated_sales_summary": summarize_estimated_sales(result.get("estimated_sales")),
            "store_stats_summary": summarize_store_stats(result.get("store_stats")),
            "transit_accessibility": result.get("transit_accessibility"),
            "store_breakdown": store_breakdown_to_plain_dict(result["store_breakdown"]),
            "average_sale_price": result["price_info"],
            "price_trend": result.get("price_trend"),
            "resident_population": result.get("resident_population"),
            "resident_population_level": result.get("resident_population_level"),
            "trade_area_match_method": result.get("trade_area_match_method"),
            "demand_based_recommendations": result.get("demand_recommendations"),
            "geo": result.get("geo"),
            "error": result["error"],
        }

    if tool_name == "estimate_shop_rent":
        area_result = analyze_single_area(tool_input["keyword"], tool_input["address"])
        if area_result["error"]:
            return {"error": area_result["error"]}

        sale_items = get_commercial_sale_prices(tool_input["lawd_cd"], tool_input["deal_ym"])
        dong_name = (area_result.get("geo") or {}).get("dong_name")
        # dong_name을 주면 그 동 거래만 우선 쓰고, 없으면 함수 내부에서 구 전체로 자동 대체된다.
        price_per_pyeong = estimate_price_per_pyeong(sale_items, dong_name=dong_name)

        if price_per_pyeong is None:
            return {"error": "해당 지역/기간의 매매실거래가 데이터를 찾을 수 없습니다."}

        return estimate_rent(
            base_price_per_pyeong=price_per_pyeong,
            area_pyeong=tool_input["area_pyeong"],
            pop_summary=area_result["population"],
            store_breakdown=area_result["store_breakdown"],
            business_hours=tool_input["business_hours"],
            category_name=tool_input["category_name"],
            category_level="mid",
            cap_rate=0.05,
            floor_weight=tool_input.get("floor_weight", 1.0),
            estimated_sales_rows=area_result.get("estimated_sales"),
            store_stats_rows=area_result.get("store_stats"),
        )

    if tool_name == "analyze_building":
        return get_building_info(tool_input["address"])

    if tool_name == "predict_listing_profit":
        primary, geo = resolve_trade_area(tool_input.get("keyword"), tool_input.get("address"))
        if not primary:
            return {"error": "매물 위치의 상권을 찾지 못했습니다: " + str(tool_input.get("keyword"))}
        result = predict_listing_profit(
            primary["trdar_cd"],
            tool_input["business"],
            area_pyeong=tool_input["area_pyeong"],
            monthly_rent=tool_input["monthly_rent_manwon"],
            deposit=tool_input.get("deposit_manwon") or 0,
            maintenance=tool_input.get("maintenance_manwon"),
            premium=tool_input.get("premium_manwon") or 0,
            floor=tool_input.get("floor") or 1,
            interior_total=tool_input.get("interior_manwon"),
            open_days_per_week=tool_input.get("open_days_per_week") or 6,
        )
        result["상권명"] = primary.get("trdar_cd_nm")
        result["geo"] = geo
        return result

    if tool_name == "recommend_locations_for_business":
        return recommend_locations(
            tool_input["business"],
            gu=tool_input.get("gu"),
            top_n=tool_input.get("top_n") or 10,
            target_ages=tool_input.get("target_ages"),
        )

    if tool_name in ("analyze_foot_traffic", "estimate_target_customers"):
        primary, geo = resolve_trade_area(tool_input.get("keyword"), tool_input.get("address"))
        if not primary:
            return {"error": "상권을 찾지 못했습니다: " + str(tool_input.get("keyword"))}
        detail = analyze_foot_traffic(primary["trdar_cd"], station_names=tool_input.get("station_names"))
        pop_summary = summarize_population(detail.pop("flpop_rows", None))
        base = {"keyword": tool_input.get("keyword"), "trdar_cd_nm": primary.get("trdar_cd_nm"), "geo": geo}

        if tool_name == "analyze_foot_traffic":
            base.update(detail)
            base["유동인구_요약"] = pop_summary
            return base

        base["기준_분기"] = detail.get("기준_분기")
        base["관심고객_추정"] = estimate_target_customers(
            pop_summary,
            detail.get("유동인구_상세"),
            target_ages=tool_input.get("target_ages"),
            target_gender=tool_input.get("target_gender", "전체"),
            business_hours=tool_input.get("business_hours"),
            open_days=tool_input.get("open_days"),
        )
        return base

    return {"error": "알 수 없는 도구: " + tool_name}


def build_charts_from_tool_call(tool_name, result):
    """
    도구 실행 결과에서 화면에 표시할 차트 데이터 목록을 만든다.
    각 차트는 {"title": ..., "data": {라벨: 값, ...}} 형태.
    단일 값 하나짜리 막대그래프(매매가, 정주인구 등)는 시각적으로 의미가 적어
    제외하고, 실제 분포를 보여주는 시간대/연령대/성별 그래프만 유지한다.
    (매매가/임대료/정주인구/업종 추천 수치는 차트 대신 텍스트 답변에서 설명된다.)
    """
    charts = []
    if result.get("error"):
        return charts

    if tool_name == "analyze_trade_area":
        area_name = result.get("trdar_cd_nm") or result.get("keyword") or ""
        pop = result.get("population") or {}

        age_keys = ["10대", "20대", "30대", "40대", "50대", "60대_이상"]
        age_data = {k: pop.get(k) for k in age_keys if pop.get(k) is not None}
        if age_data:
            charts.append({"title": area_name + " - 연령대별 유동인구", "data": age_data})

        gender_data = {}
        if pop.get("남성_유동인구") is not None:
            gender_data["남성"] = pop["남성_유동인구"]
        if pop.get("여성_유동인구") is not None:
            gender_data["여성"] = pop["여성_유동인구"]
        if gender_data:
            charts.append({"title": area_name + " - 성별 유동인구", "data": gender_data})

        time_keys = [
            "시간대_00~06", "시간대_06~11", "시간대_11~14",
            "시간대_14~17", "시간대_17~21", "시간대_21~24",
        ]
        time_data = {k: pop.get(k) for k in time_keys if pop.get(k) is not None}
        if time_data:
            charts.append({"title": area_name + " - 시간대별 유동인구", "data": time_data})

        population_trend = result.get("population_trend")
        if population_trend:
            pop_trend_data = {
                row["std_ym"]: row["총_유동인구"]
                for row in population_trend
                if row.get("총_유동인구") is not None
            }
            if len(pop_trend_data) >= 2:  # 값이 1개뿐이면 추이라고 보기 어려우니 생략
                charts.append({
                    "title": area_name + " - 총 유동인구 추이(최근 " + str(len(population_trend)) + "개 분기)",
                    "data": pop_trend_data,
                    "type": "line",
                })

        sales_summary = result.get("estimated_sales_summary")
        if sales_summary and sales_summary.get("매출_상위5_업종"):
            sales_data = {
                row["업종명"]: round((row.get("당월_추정매출_원") or 0) / 10000)
                for row in sales_summary["매출_상위5_업종"]
            }
            if sales_data:
                charts.append({
                    "title": area_name + " - 업종별 추정매출 상위 5(당월, 만원)",
                    "data": sales_data,
                })

        store_stats_summary = result.get("store_stats_summary")
        if store_stats_summary and store_stats_summary.get("폐업률_상위5_업종"):
            closure_data = {
                row["업종명"]: row.get("폐업률") or 0
                for row in store_stats_summary["폐업률_상위5_업종"]
            }
            if closure_data:
                charts.append({
                    "title": area_name + " - 업종별 폐업률 상위 5(%, 표본 3개 이상 업종만)",
                    "data": closure_data,
                })

        price_trend = result.get("price_trend")
        if price_trend:
            trend_data = {
                row["deal_ym"]: row["avg_price_per_pyeong_manwon"]
                for row in price_trend
                if row.get("avg_price_per_pyeong_manwon") is not None
            }
            if len(trend_data) >= 2:  # 값이 1개뿐이면 추이라고 보기 어려우니 생략
                charts.append({
                    "title": area_name + " - 평당 매매가 추이(최근 " + str(len(price_trend)) + "개월, 만원)",
                    "data": trend_data,
                    "type": "line",
                })

    if tool_name == "predict_listing_profit" and result.get("시나리오"):
        label = (result.get("상권명") or "") + " " + (result.get("업종") or "")
        charts.append({
            "title": label + " - 시나리오별 월 순이익(만원)",
            "data": {k: v["월순이익_만원"] for k, v in result["시나리오"].items()},
        })
        costs = result["시나리오"]["보통"]["비용_내역_만원"]
        charts.append({
            "title": label + " - 보통 시나리오 월 비용 내역(만원)",
            "data": dict(sorted(costs.items(), key=lambda kv: kv[1], reverse=True)),
        })

    if tool_name == "recommend_locations_for_business":
        top = (result.get("추천_상권") or [])[:7]
        if top:
            charts.append({
                "title": (result.get("업종") or "") + " - 추천 상권 점포당 월 추정매출(만원)",
                "data": {str(i + 1) + ". " + str(c["상권명"]): c["점포당_월매출_추정_만원"] for i, c in enumerate(top)},
            })

    if tool_name == "analyze_foot_traffic":
        area_name = result.get("trdar_cd_nm") or result.get("keyword") or ""
        detail = result.get("유동인구_상세") or {}
        by_day = {k: v for k, v in (detail.get("요일별") or {}).items() if v is not None}
        if by_day:
            charts.append({"title": area_name + " - 요일별 유동인구(분기 합계)", "data": by_day})
        for sw in result.get("지하철_승하차") or []:
            hourly = sw.get("시간대별_하루평균")
            if hourly:
                charts.append({
                    "title": sw.get("station", "") + "역 - 시간대별 하루평균 하차 인원(" + str(sw.get("month")) + ")",
                    "data": {h: v["하차"] for h, v in hourly.items()},
                })

    return charts


def render_charts(charts):
    for chart in charts:
        st.write("**" + chart["title"] + "**")
        df = pd.DataFrame({"값": list(chart["data"].values())}, index=list(chart["data"].keys()))
        if chart.get("type") == "line":
            st.line_chart(df)
        else:
            st.bar_chart(df)


import base64
import re


def _is_markdown_table_row(line):
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_markdown_table_separator(line):
    stripped = line.strip()
    if not _is_markdown_table_row(stripped):
        return False
    inner = stripped.strip("|")
    return all(ch in "-: |" for ch in inner) and "-" in inner


def fix_markdown_tables(text):
    """
    Claude 답변에 나오는 마크다운 표에서, 구분선 행(|---|---|...)의 칸 수가
    헤더/데이터 행의 칸 수와 안 맞으면 표 전체가 깨져서 렌더링되지 않는
    문제가 있었다 (예: 컬럼 5개인데 구분선은 4칸만 있는 경우). 시스템
    프롬프트로 형식을 지키라고 안내는 했지만 100% 보장은 안 되므로, 화면에
    그리기 직전에 각 표 블록의 구분선 칸 수를 헤더 행 칸 수에 맞춰 자동으로
    다시 만들어 표가 항상 정상적으로 렌더링되게 한다.
    """
    lines = text.split("\n")
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_markdown_table_row(line):
            # 표의 첫 줄(헤더)만 기준으로 칸 수를 정하고, 구분선은 헤더 바로
            # 다음에 딱 한 번만 넣는다. 그 뒤로 이어지는 데이터 행들은 원래
            # 있던 구분선(있다면)만 건너뛰고 그대로 옮긴다 - 데이터 행마다
            # 구분선을 새로 끼워넣으면 표가 여러 조각으로 쪼개져 버린다.
            header_cols = line.strip().strip("|").count("|") + 1
            fixed_lines.append(line)
            i += 1
            if i < len(lines) and _is_markdown_table_separator(lines[i]):
                i += 1  # 원래 있던(칸 수가 틀렸을 수도 있는) 구분선은 버린다.
            fixed_lines.append("|" + "|".join(["---"] * header_cols) + "|")

            while i < len(lines) and _is_markdown_table_row(lines[i]):
                if not _is_markdown_table_separator(lines[i]):
                    fixed_lines.append(lines[i])
                i += 1
        else:
            fixed_lines.append(line)
            i += 1
    return "\n".join(fixed_lines)


def encode_image_base64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")


def extract_rooms_from_image(client, file_bytes, media_type):
    """
    도면 이미지를 Claude 비전으로 읽어서 호실 목록(호실명, 층)을 JSON으로 추출.
    이미지 상단에 '전용: OOO평 (OO개 호실)' 같은 표기가 있으면 함께 읽어서
    평균 평수를 자동 계산해 기본값으로 채운다.
    """
    b64 = encode_image_base64(file_bytes)
    instruction = (
        "이 이미지는 상가 건물의 층별 임대 현황 도면입니다. "
        "다음 두 가지를 읽어주세요.\n\n"
        "1) 이미지 상단에 '지상 O층 (OO개 호실) 전용 : OOO.OO평' 같은 표기가 있으면, "
        "그 층 이름(floor_label), 호실수(total_room_count), 전체 전용면적(total_area_pyeong)을 숫자로 추출\n"
        "2) 이미지 안에서 보이는 모든 호실(구획) 번호/이름과, 그 호실이 있는 층을 추출\n\n"
        "다음 JSON 형식으로만 답하세요. 다른 설명은 절대 추가하지 마세요.\n\n"
        "{\n"
        '  "floor_label": "지상 3층",\n'
        '  "total_room_count": 23,\n'
        '  "total_area_pyeong": 601.79,\n'
        '  "rooms": [{"room_name": "301", "floor": "지상 3층"}, {"room_name": "302", "floor": "지상 3층"}]\n'
        "}\n\n"
        "층 정보가 없으면 floor_label/total_room_count/total_area_pyeong은 null로 두세요. "
        "확실하지 않은 호실은 rooms 목록에서 제외하세요."
    )

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
    )

    text = "".join(b.text for b in response.content if b.type == "text")

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else json.loads(text)
    except Exception as e:
        st.error("호실 목록 추출 중 JSON 파싱 실패: " + str(e))
        st.code(text)
        return [], None

    rooms = parsed.get("rooms", [])
    total_count = parsed.get("total_room_count")
    total_area = parsed.get("total_area_pyeong")

    avg_pyeong = None
    if total_count and total_area and total_count > 0:
        avg_pyeong = round(total_area / total_count, 2)

    for room in rooms:
        room["평수"] = avg_pyeong if avg_pyeong is not None else 10.0

    floor_summary = {
        "floor_label": parsed.get("floor_label"),
        "total_room_count": total_count,
        "total_area_pyeong": total_area,
        "avg_pyeong": avg_pyeong,
    }

    return rooms, floor_summary


def floor_weight_from_label(floor_label):
    """
    층 표기 문자열에서 층수 가중치를 추정한다. 1층을 기준으로 위/아래로
    한 단계씩 멀어질 때마다 CONFIG["FLOOR_WEIGHT"]의 level_1 -> level_2 ->
    level_3_plus 순서로 할인폭이 커지고, 3단계 이상부터는 더 내려가지 않고
    level_3_plus 값에서 고정된다(최대 할인 고정).

    - 1층: ground_1 (할인 없음)
    - 2층 / 지하1층 / 옥상층: level_1
    - 3층 / 지하2층: level_2
    - 4층 이상 / 지하3층 이하: level_3_plus (최대 할인, 더 안 내려감)
    """
    weights = CONFIG["FLOOR_WEIGHT"]
    label = floor_label or ""

    # 옥상층은 2층과 동일한 등급(level_1) 취급
    if "옥상" in label:
        return weights["level_1"]

    is_basement = ("지하" in label) or bool(re.search(r"B\s*\d", label.upper()))

    match = re.search(r"(\d+)", label)
    floor_num = int(match.group(1)) if match else None

    if is_basement:
        depth = floor_num if floor_num else 1  # 지하1층/B1 등, 숫자 없으면 1층으로 취급
        if depth <= 1:
            return weights["level_1"]
        if depth == 2:
            return weights["level_2"]
        return weights["level_3_plus"]

    # 지상층
    if not floor_num or floor_num <= 1:
        return weights["ground_1"]
    level = floor_num - 1  # 2층->1단계, 3층->2단계, 4층->3단계...
    if level == 1:
        return weights["level_1"]
    if level == 2:
        return weights["level_2"]
    return weights["level_3_plus"]


def recommend_categories_for_rooms(client, rooms, area_result):
    """
    여러 호실에 대해 한 번의 호출로 적합 업종을 추천받는다.
    area_result: analyze_single_area()의 반환값 (유동인구 + 업종분포 포함)
    """
    sb = area_result.get("store_breakdown")
    sb_plain = store_breakdown_to_plain_dict(sb) if sb else None
    context = {
        "trdar_cd_nm": area_result.get("trdar_cd_nm"),
        "population": area_result.get("population"),
        "store_breakdown": sb_plain,
    }

    instruction = (
        "당신은 상권 분석 전문가입니다. 아래 상권 데이터를 참고하여, "
        "주어진 각 호실(층 정보 포함)에 가장 적합한 업종(중분류 수준, 예: '한식', '카페(비알코올)', "
        "'의원', '기타 간이' 등)과 그 업종의 주 영업시간대를 추천하세요. "
        "이미 업종 중분류에 개수가 많이 몰려있는 업종(포화 업종)은 가능한 피하고, "
        "층수 특성(1층=접근성 좋은 업종, 상층/지하=목적성 방문 업종)도 고려하세요.\n\n"
        "상권 데이터:\n" + json.dumps(context, ensure_ascii=False, default=str) + "\n\n"
        "호실 목록:\n" + json.dumps(rooms, ensure_ascii=False) + "\n\n"
        "다음 JSON 배열 형식으로만 답하세요. 다른 설명은 추가하지 마세요.\n"
        '[{"room_name": "101호", "floor": "1층", "recommended_category": "카페(비알코올)", '
        '"business_hours": ["시간대_06~11", "시간대_11~14"], "reasoning": "간단한 추천 이유"}]'
    )

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=3000,
        messages=[{"role": "user", "content": instruction}],
    )

    text = "".join(b.text for b in response.content if b.type == "text")

    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        st.error("업종 추천 결과 JSON 파싱 실패: " + str(e))
        st.code(text)
        return []



def describe_tool_call(tool_name, tool_input):
    if tool_name == "analyze_trade_area":
        return "🔍 '" + tool_input.get("keyword", "") + "' 상권 분석 중"
    if tool_name == "estimate_shop_rent":
        return "💰 '" + tool_input.get("keyword", "") + "' 임대료 계산 중"
    if tool_name == "predict_listing_profit":
        return "💵 '" + tool_input.get("keyword", "") + "' 매물 수익 계산 중"
    if tool_name == "recommend_locations_for_business":
        return "🗺️ '" + tool_input.get("business", "") + "' 하기 좋은 상권 찾는 중"
    if tool_name == "analyze_foot_traffic":
        return "🚶 '" + tool_input.get("keyword", "") + "' 인적 동선 분석 중"
    if tool_name == "estimate_target_customers":
        return "🎯 '" + tool_input.get("keyword", "") + "' 관심고객 수 추정 중"
    return "🔧 " + tool_name + " 실행 중"


def get_claude_response(client, messages):
    """
    도구 호출이 필요 없을 때까지 반복해서 Claude를 호출하고,
    최종 텍스트 답변, 업데이트된 메시지 목록, 이번 턴 도구 호출 기록(차트용)을 반환.
    """
    tool_call_records = []

    while True:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system=st.session_state.get("system_prompt", SYSTEM_PROMPT_FOUNDER),
            tools=TOOLS,
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        assistant_content = []
        for b in response.content:
            if b.type == "text":
                assistant_content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                assistant_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_use_blocks:
            final_text = "\n".join(t.text for t in text_blocks)
            return final_text, messages, tool_call_records

        tool_results = []
        for tool_use in tool_use_blocks:
            label = describe_tool_call(tool_use.name, tool_use.input)

            with st.status(label, expanded=True) as status:
                if tool_use.name == "analyze_trade_area":
                    st.write("📍 위치(좌표) 확인 중...")
                    st.write("📊 유동인구 · 업종 데이터 조회 중... (병렬 처리)")
                    st.write("💰 인근 매매가/임대료 시세 조회 중... (병렬 처리)")
                elif tool_use.name == "predict_listing_profit":
                    st.write("📍 매물이 속한 상권 찾는 중...")
                    st.write("📊 같은 업종 가게들의 매출 불러오는 중...")
                    st.write("🧠 매출 예측 모델 실행 중... (앱을 처음 켰다면 학습이 끝날 때까지 2~4분 걸려요)")
                    st.write("🧮 비용·손익분기·투자금 회수 기간 계산 중...")
                elif tool_use.name == "recommend_locations_for_business":
                    st.write("📥 서울 전체 상권의 매출·점포·유동인구 데이터 불러오는 중... (처음 한 번은 30초쯤 걸려요)")
                    st.write("🧮 점포당 매출 · 관심 유동인구 · 폐업률로 순위 매기는 중...")
                elif tool_use.name == "estimate_shop_rent":
                    st.write("📍 상권 데이터 확인 중...")
                    st.write("💰 매매실거래가 조회 중...")
                    st.write("🧮 유동인구·경쟁밀도 반영해 임대료 계산 중...")
                else:
                    st.write("처리 중...")

                result = run_tool(tool_use.name, tool_use.input)
                status.update(label=label + " - 완료", state="complete", expanded=False)

            tool_call_records.append({"name": tool_use.name, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
        messages.append({"role": "user", "content": tool_results})


def render_naver_map(lat, lng, title="분석 위치"):
    """
    네이버 Static Map API로 서버에서 받아온 지도 이미지를 화면에 표시한다.

    예전에는 Dynamic Map(JS)을 components.html()로 띄웠는데, Streamlit이 이 HTML을
    샌드박스 iframe(about:srcdoc) 안에서 렌더링하다 보니 네이버 서버가 요청 출처를
    "about:srcdoc"으로 인식해서 등록된 Web 서비스 URL과 안 맞아 지도가 계속 거부됐다
    (브라우저 콘솔에 "URI: about:srcdoc" 에러로 확인됨).
    Static Map은 브라우저가 아니라 이 서버가 직접 ID-KEY로 인증해서 이미지를
    받아오기 때문에 그 문제 자체가 생기지 않는다.
    """
    image_bytes, error = get_static_map_image(lat, lng, title=title)
    if not image_bytes:
        st.warning("지도 이미지를 불러오지 못했습니다. 원인: " + str(error))
        return
    st.image(image_bytes, caption=title, use_container_width=True)


@st.cache_resource
def warm_up_sales_model():
    """앱이 켜질 때 예측 모델을 뒤에서 미리 학습시켜 둔다 (처음 질문할 때 기다리지 않도록)."""
    import threading
    t = threading.Thread(target=sales_model.get_model, daemon=True)
    t.start()
    return True


def main():
    st.set_page_config(page_title="서울시 상권 분석 AI ChatBot", page_icon="🏢")

    # 새 답변이 나올 때 화면이 순간이동하듯 뚝뚝 끊기지 않고
    # 부드럽게 스크롤되도록 하는 CSS
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            scroll-behavior: smooth !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        mode = st.radio(
            "누가 쓰나요?",
            ["🧑‍🍳 창업자 (가게를 열고 싶어요)", "🏢 임대인 (상가를 채우고 싶어요)"],
            key="user_mode",
        )
    is_founder = mode.startswith("🧑‍🍳")
    st.session_state.system_prompt = SYSTEM_PROMPT_FOUNDER if is_founder else SYSTEM_PROMPT_LANDLORD

    # 모드를 바꾸면 이전 대화가 다른 말투로 이어지지 않도록 대화를 새로 시작한다
    if st.session_state.get("last_mode") not in (None, mode):
        st.session_state.display_messages = []
        st.session_state.api_messages = []
    st.session_state.last_mode = mode

    if is_founder:
        st.title("🧑‍🍳 서울 창업 입지 AI")
        st.caption("어디에, 무엇을 열면 좋을지 공공데이터로 같이 따져봐요.")
    else:
        st.title("🏢 서울시 상권 분석 AI ChatBot")
        st.caption("궁금한 상권의 유동인구, 업종 분포, 추천 임대료를 물어보세요.")

    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit secrets를 확인하세요.")
        st.stop()

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    warm_up_sales_model()

    with st.sidebar:
        with st.expander("🧠 매출 예측 모델 검증 결과"):
            st.caption(
                "서울 상권×업종 사례로 '가게 하나가 얼마를 버는지'를 학습한 모델이 실제로 얼마나 맞히는지 보여줍니다. "
                "처음 켜면 학습에 2~4분 걸립니다."
            )
            if st.button("검증 결과 보기"):
                with st.spinner("모델 확인 중... (아직 학습 중이면 끝날 때까지 기다립니다)"):
                    rep = sales_model.model_report()
                if rep.get("error"):
                    st.error(rep["error"])
                else:
                    names = {
                        "시간검증_모델A": "🧠 모델A (지난 분기 매출 포함)",
                        "시간검증_모델B": "🧠 모델B (동네 특성만)",
                        "시간검증_기준선_지난분기값": "📏 기준선: 지난 분기 값 그대로",
                        "시간검증_기준선_업종중앙값": "📏 기준선: 서울 업종 중앙값",
                        "공간검증_모델B(처음보는상권)": "🗺️ 처음 보는 상권 (모델B)",
                    }
                    for key, label in names.items():
                        m = rep.get(key)
                        if not isinstance(m, dict):
                            continue
                        parts = ["오차 중앙값 " + str(m.get("오차_중앙값(%)")) + "%",
                                 "±30% 안 " + str(m.get("오차_30%이내_비율(%)")) + "%"]
                        if "80%구간_적중률(%)" in m:
                            parts.append("범위 적중 " + str(m["80%구간_적중률(%)"]) + "%")
                        if "80%구간_적중률_보정전(%)" in m:
                            parts.append("범위 적중(보정 전) " + str(m["80%구간_적중률_보정전(%)"]) + "%")
                        st.markdown("**" + label + "**  \n" + " · ".join(parts))
                    st.caption(rep.get("설명", ""))
                    st.write("기준 분기:", rep.get("기준_분기"), "· 학습 사례:", rep.get("학습_사례수"))
            if st.button("모델 다시 학습"):
                with st.spinner("최신 데이터로 다시 학습 중... (2~4분)"):
                    sales_model.get_model(force=True)
                st.success("다시 학습했습니다. '검증 결과 보기'를 눌러 확인하세요.")

        with st.expander("🔧 데이터 점검 (개발용)"):
            st.caption("새로 연결한 서울시 API들이 실제로 데이터를 주는지, 필드 이름이 무엇인지 확인합니다.")
            if st.button("점검 실행"):
                with st.spinner("API 확인 중..."):
                    st.json(inspect_new_apis())

        if is_founder:
            st.header("🏠 매물 수익 계산기")
            st.caption("보고 있는 매물 정보를 넣으면 한 달에 얼마 남을지 계산해 드려요.")
            with st.form("listing_form"):
                f_addr = st.text_input("매물 위치 (역 이름이나 주소)", placeholder="예: 망원역, 서울 마포구 망원동 123")
                f_biz = st.text_input("하려는 업종", placeholder="예: 카페, 치킨, 미용실")
                c1, c2 = st.columns(2)
                f_area = c1.number_input("전용면적(평)", min_value=1.0, value=15.0, step=1.0)
                f_floor = c2.selectbox("층", ["1층", "2층", "3층 이상", "지하"])
                f_deposit = c1.number_input("보증금(만원)", min_value=0, value=3000, step=500)
                f_rent = c2.number_input("월세(만원)", min_value=0, value=200, step=10)
                f_maint = c1.number_input("관리비(만원, 모르면 0)", min_value=0, value=0, step=5)
                f_premium = c2.number_input("권리금(만원)", min_value=0, value=0, step=500)
                submitted = st.form_submit_button("💵 수익 계산하기", use_container_width=True)
            if submitted:
                if not f_addr or not f_biz:
                    st.warning("매물 위치와 업종을 입력해 주세요.")
                else:
                    floor_txt = {"1층": "1층", "2층": "2층", "3층 이상": "3층", "지하": "지하1층"}[f_floor]
                    q = (
                        f_addr + "에 있는 매물에서 " + f_biz + "를 하면 한 달에 얼마나 남을까? "
                        + "전용 " + str(f_area) + "평, " + floor_txt + ", 보증금 " + str(f_deposit) + "만원, "
                        + "월세 " + str(f_rent) + "만원"
                        + (", 관리비 " + str(f_maint) + "만원" if f_maint else ", 관리비는 몰라")
                        + (", 권리금 " + str(f_premium) + "만원" if f_premium else ", 권리금 없음")
                    )
                    st.session_state.pending_question = q
                    st.rerun()

        if not is_founder:
            st.header("📐 도면 기반 호실별 분석")

            uploaded_file = st.file_uploader("상가 도면 이미지 업로드", type=["png", "jpg", "jpeg"])
            keyword = st.text_input("상권 키워드 (예: OO역)", key="floor_keyword")
            address = st.text_input("지오코딩 주소 (예: 서울특별시 OO구 OO동)", key="floor_address")
            lawd_cd = st.text_input("법정동코드 5자리", value="11230", key="floor_lawd")
            deal_ym = st.text_input("매매실거래가 계약년월 6자리", value="202412", key="floor_deal_ym")

            if st.button("호실 목록 초기화"):
                st.session_state.extracted_rooms = []
                st.session_state.room_price_results = []

            if uploaded_file and st.button("1단계: 호실 목록 추출 (현재 업로드된 층 추가)"):
                file_bytes = uploaded_file.getvalue()
                media_type = uploaded_file.type
                with st.spinner("도면에서 호실 읽는 중..."):
                    new_rooms, floor_summary = extract_rooms_from_image(client, file_bytes, media_type)

                if "extracted_rooms" not in st.session_state:
                    st.session_state.extracted_rooms = []

                existing_keys = {(r.get("room_name"), r.get("floor")) for r in st.session_state.extracted_rooms}
                for r in new_rooms:
                    key = (r.get("room_name"), r.get("floor"))
                    if key not in existing_keys:
                        st.session_state.extracted_rooms.append(r)
                        existing_keys.add(key)

                if floor_summary and floor_summary.get("avg_pyeong"):
                    st.info(
                        (floor_summary.get("floor_label") or "이 층") + ": "
                        + str(floor_summary.get("total_room_count")) + "개 호실, "
                        + "전체 " + str(floor_summary.get("total_area_pyeong")) + "평 "
                        + "→ 평균 " + str(floor_summary.get("avg_pyeong")) + "평/호실로 기본 반영됨"
                    )

            if st.session_state.get("extracted_rooms"):
                st.write("추출된 호실 목록 - 평수는 층 평균값이 기본으로 채워져 있어요. 필요시 직접 수정하세요.")
                rooms_df = pd.DataFrame(st.session_state.extracted_rooms)
                if "평수" not in rooms_df.columns:
                    rooms_df["평수"] = 10.0

                edited_df = st.data_editor(rooms_df, num_rows="dynamic", key="rooms_editor")

                if st.button("2단계: 업종 추천 + 임대료 계산"):
                    if not keyword or not address:
                        st.warning("상권 키워드와 주소를 입력해주세요.")
                    else:
                        with st.spinner("상권 데이터 조회 중..."):
                            area_result = analyze_single_area(keyword, address)

                        if area_result["error"]:
                            st.error(area_result["error"])
                        else:
                            rooms_list = edited_df.to_dict("records")
                            with st.spinner("호실별 업종 추천 중..."):
                                recommendations = recommend_categories_for_rooms(client, rooms_list, area_result)

                            sale_items = get_commercial_sale_prices(lawd_cd, deal_ym)
                            price_per_pyeong = estimate_price_per_pyeong(sale_items)

                            result_rows = []
                            for rec in recommendations:
                                room_name = rec.get("room_name")
                                floor = rec.get("floor", "")
                                category = rec.get("recommended_category", "")
                                hours = rec.get("business_hours", [])
                                reasoning = rec.get("reasoning", "")

                                area_pyeong = 10.0
                                if "room_name" in edited_df.columns:
                                    match_row = edited_df[edited_df["room_name"] == room_name]
                                    if not match_row.empty and "평수" in match_row.columns:
                                        area_pyeong = float(match_row.iloc[0]["평수"])

                                floor_weight = floor_weight_from_label(floor)

                                monthly_rent = None
                                if price_per_pyeong is not None:
                                    rent_result = estimate_rent(
                                        base_price_per_pyeong=price_per_pyeong,
                                        area_pyeong=area_pyeong,
                                        pop_summary=area_result["population"],
                                        store_breakdown=area_result["store_breakdown"],
                                        business_hours=hours,
                                        category_name=category,
                                        category_level="mid",
                                        cap_rate=0.05,
                                        floor_weight=floor_weight,
                                    )
                                    monthly_rent = rent_result["최종_추천_월임대료_만원"]

                                result_rows.append({
                                    "호실": room_name,
                                    "층": floor,
                                    "평수": area_pyeong,
                                    "추천업종": category,
                                    "추천월임대료(만원)": monthly_rent,
                                    "추천이유": reasoning,
                                })

                            st.session_state.room_price_results = result_rows

            if st.session_state.get("room_price_results"):
                st.subheader("호실별 추천 결과")
                result_df = pd.DataFrame(st.session_state.room_price_results)
                st.dataframe(result_df, use_container_width=True)

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "api_messages" not in st.session_state:
        st.session_state.api_messages = []

    for idx, msg in enumerate(st.session_state.display_messages):
        with st.chat_message(msg["role"]):
            st.markdown(fix_markdown_tables(msg["content"]))
            if msg.get("charts"):
                render_charts(msg["charts"])
            if msg.get("map_info"):
                render_naver_map(msg["map_info"]["lat"], msg["map_info"]["lng"], msg["map_info"]["title"])

    examples = []
    if is_founder and not st.session_state.display_messages and "pending_question" not in st.session_state:
        st.write("**이렇게 물어보세요** (눌러서 바로 질문)")
        examples = [
            "망원역 근처에서 무슨 가게를 하면 좋을까?",
            "카페를 하고 싶은데 서울에서 어디가 좋을까?",
            "마포구에서 20~30대 여성 대상 네일숍 하기 좋은 곳 추천해줘",
            "청량리역에 치킨집 열면 하루 손님 몇 명쯤 올까?",
            "망원역 1층 15평, 보증금 3천에 월세 250인 자리에서 카페 하면 얼마 남을까?",
            "성수역 근처 20평 2층 월세 300 매물, 어떤 업종이 제일 많이 남을까?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key="ex_" + str(i), use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    placeholder = "예: 카페 하려면 어디가 좋아?" if is_founder else "예: OO역 상권은 어때?"
    user_input = st.chat_input(placeholder) or st.session_state.pop("pending_question", None)

    if user_input:
        st.session_state.display_messages.append({"role": "user", "content": user_input, "charts": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.api_messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    final_text, updated_messages, tool_call_records = get_claude_response(
                        client, st.session_state.api_messages
                    )
                except AuthenticationError:
                    # 실패한 질문은 대화 기록에서 빼야 다음 질문이 정상 동작한다
                    st.session_state.api_messages.pop()
                    st.session_state.display_messages.pop()
                    st.error(
                        "Claude API 키가 맞지 않거나 만료되었습니다. "
                        "console.anthropic.com에서 새 키를 만든 뒤, Streamlit Cloud의 "
                        "Settings → Secrets에 있는 ANTHROPIC_API_KEY 값을 바꿔주세요."
                    )
                    st.stop()
            st.markdown(fix_markdown_tables(final_text))

            all_charts = []
            map_info = None
            for record in tool_call_records:
                all_charts.extend(build_charts_from_tool_call(record["name"], record["result"]))
                if record["name"] == "recommend_locations_for_business":
                    top = (record["result"].get("추천_상권") or [None])[0]
                    if top and top.get("lat") and top.get("lng") and not map_info:
                        map_info = {"lat": top["lat"], "lng": top["lng"], "title": "추천 1위: " + str(top["상권명"])}
                if record["name"] in ("analyze_trade_area", "analyze_foot_traffic", "estimate_target_customers", "predict_listing_profit"):
                    geo = record["result"].get("geo")
                    if geo and geo.get("lat") and geo.get("lng"):
                        map_info = {
                            "lat": geo["lat"],
                            "lng": geo["lng"],
                            "title": record["result"].get("trdar_cd_nm") or record["result"].get("keyword", ""),
                        }

            if all_charts:
                render_charts(all_charts)
            if map_info:
                st.write("**📍 위치 확인 (네이버 지도)**")
                render_naver_map(map_info["lat"], map_info["lng"], map_info["title"])

        st.session_state.api_messages = updated_messages
        st.session_state.display_messages.append({
            "role": "assistant",
            "content": final_text,
            "charts": all_charts,
            "map_info": map_info,
        })


if __name__ == "__main__":
    main()