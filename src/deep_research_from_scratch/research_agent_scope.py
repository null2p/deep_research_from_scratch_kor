
"""사용자 명확화 및 Research Brief 생성.

이 모듈은 research workflow의 scoping 단계를 구현합니다:
1. 사용자 요청에 명확화가 필요한지 평가
2. 대화 내용으로부터 상세한 research brief 생성

workflow는 structured output을 사용하여 연구를 진행하기에 충분한
컨텍스트가 있는지에 대해 결정론적 결정을 내립니다.
"""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

# ===== 유틸리티 함수 =====

def get_today_str() -> str:
    """현재 날짜를 사람이 읽기 쉬운 형식으로 반환."""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== 설정 =====

# 모델 초기화
model = init_chat_model(model="openai:gpt-4.1", temperature=0.0)

# ===== WORKFLOW 노드 =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    사용자 요청에 연구를 진행하기 위한 충분한 정보가 있는지 확인.

    structured output을 사용하여 결정론적 결정을 내리고 환각을 방지합니다.
    research brief 생성으로 라우팅하거나 clarification 질문으로 종료합니다.
    """
    # structured output model 설정
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # clarification 지시사항과 함께 모델 호출
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # clarification 필요 여부에 따라 라우팅
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    대화 이력을 포괄적인 research brief로 변환.

    structured output을 사용하여 brief가 필요한 형식을 따르고
    효과적인 연구를 위한 모든 필수 세부 정보를 포함하도록 보장합니다.
    """
    # structured output model 설정
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # 대화 이력으로부터 research brief 생성
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # 생성된 research brief로 state를 업데이트하고 supervisor로 전달
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ===== GRAPH 구성 =====

# scoping workflow 구축
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# workflow 노드 추가
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# workflow edge 추가
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# workflow 컴파일
scope_research = deep_researcher_builder.compile()
