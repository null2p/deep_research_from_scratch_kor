
"""연구 유틸리티 및 도구.

이 모듈은 연구 에이전트를 위한 검색 및 콘텐츠 처리 유틸리티를 제공합니다.
웹 검색 기능과 콘텐츠 요약 도구를 포함합니다.
"""

from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

from deep_research_from_scratch.state_research import Summary
from deep_research_from_scratch.prompts import summarize_webpage_prompt

# ===== 유틸리티 함수 =====

def get_today_str() -> str:
    """현재 날짜를 사람이 읽기 쉬운 형식으로 반환합니다."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """모듈의 현재 디렉토리를 반환합니다.

    이 함수는 Jupyter notebook과 일반 Python 스크립트 모두에서 호환됩니다.

    Returns:
        현재 디렉토리를 나타내는 Path 객체
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__이 정의되지 않은 경우
        return Path.cwd()

# ===== 설정 =====

summarization_model = init_chat_model(model="openai:gpt-4.1-mini")
tavily_client = TavilyClient()

# ===== 검색 함수 =====

def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    """Tavily API를 사용하여 여러 쿼리를 검색합니다.

    Args:
        search_queries: 실행할 검색 쿼리 리스트
        max_results: 쿼리당 최대 결과 개수
        topic: 검색 결과의 주제 필터
        include_raw_content: 원본 웹페이지 콘텐츠 포함 여부

    Returns:
        검색 결과 딕셔너리 리스트
    """

    # 순차적으로 검색 실행. 참고: AsyncTavilyClient를 사용하여 이 단계를 병렬화할 수 있습니다.
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """설정된 요약 모델을 사용하여 웹페이지 콘텐츠를 요약합니다.

    Args:
        webpage_content: 요약할 원본 웹페이지 콘텐츠

    Returns:
        핵심 발췌문이 포함된 형식화된 요약
    """
    try:
        # 요약을 위한 구조화된 출력 모델 설정
        structured_model = summarization_model.with_structured_output(Summary)

        # 요약 생성
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=get_today_str()
            ))
        ])

        # 명확한 구조로 요약 형식화
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"웹페이지 요약 실패: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """중복 콘텐츠 처리를 방지하기 위해 URL 기준으로 검색 결과를 중복 제거합니다.

    Args:
        search_results: 검색 결과 딕셔너리 리스트

    Returns:
        URL을 고유한 결과에 매핑하는 딕셔너리
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """가능한 경우 콘텐츠를 요약하여 검색 결과를 처리합니다.

    Args:
        unique_results: 고유한 검색 결과 딕셔너리

    Returns:
        요약이 포함된 처리된 결과 딕셔너리
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # 요약할 원본 콘텐츠가 없으면 기존 콘텐츠 사용
        if not result.get("raw_content"):
            content = result['content']
        else:
            # 더 나은 처리를 위해 원본 콘텐츠 요약
            content = summarize_webpage_content(result['raw_content'])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """검색 결과를 잘 구조화된 문자열 출력으로 형식화합니다.

    Args:
        summarized_results: 처리된 검색 결과 딕셔너리

    Returns:
        명확한 소스 구분이 있는 형식화된 검색 결과 문자열
    """
    if not summarized_results:
        return "유효한 검색 결과를 찾을 수 없습니다. 다른 검색 쿼리를 시도하거나 다른 검색 API를 사용해주세요."

    formatted_output = "검색 결과: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- 소스 {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"요약:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== 연구 도구 =====

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """콘텐츠 요약 기능이 포함된 Tavily 검색 API에서 결과를 가져옵니다.

    Args:
        query: 실행할 단일 검색 쿼리
        max_results: 반환할 최대 결과 개수
        topic: 결과를 필터링할 주제 ('general', 'news', 'finance')

    Returns:
        요약이 포함된 형식화된 검색 결과 문자열
    """
    # 단일 쿼리 검색 실행
    search_results = tavily_search_multiple(
        [query],  # 내부 함수를 위해 단일 쿼리를 리스트로 변환
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 중복 콘텐츠 처리를 방지하기 위해 URL 기준으로 결과 중복 제거
    unique_results = deduplicate_search_results(search_results)

    # 요약과 함께 결과 처리
    summarized_results = process_search_results(unique_results)

    # 사용을 위한 출력 형식화
    return format_search_output(summarized_results)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """연구 진행 상황과 의사 결정에 대한 전략적 성찰을 위한 도구입니다.

    각 검색 후 결과를 분석하고 다음 단계를 체계적으로 계획하기 위해 이 도구를 사용하세요.
    이것은 품질 있는 의사 결정을 위해 연구 워크플로우에서 의도적인 멈춤을 만듭니다.

    사용 시점:
    - 검색 결과를 받은 후: 어떤 핵심 정보를 찾았는가?
    - 다음 단계를 결정하기 전: 포괄적으로 답변하기에 충분한가?
    - 연구 격차를 평가할 때: 아직 놓치고 있는 구체적인 정보는 무엇인가?
    - 연구를 마무리하기 전: 이제 완전한 답변을 제공할 수 있는가?

    성찰에서 다루어야 할 내용:
    1. 현재 발견사항 분석 - 어떤 구체적인 정보를 수집했는가?
    2. 격차 평가 - 아직 놓치고 있는 중요한 정보는 무엇인가?
    3. 품질 평가 - 좋은 답변을 위한 충분한 증거/예시가 있는가?
    4. 전략적 결정 - 검색을 계속해야 하는가 아니면 답변을 제공해야 하는가?

    Args:
        reflection: 연구 진행 상황, 발견사항, 격차 및 다음 단계에 대한 자세한 성찰

    Returns:
        의사 결정을 위해 성찰이 기록되었다는 확인 메시지
    """
    return f"성찰이 기록되었습니다: {reflection}"
