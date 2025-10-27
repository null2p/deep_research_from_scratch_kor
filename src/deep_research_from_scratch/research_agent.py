
"""Research Agent 구현.

이 모듈은 반복적인 웹 검색과 종합을 수행하여 복잡한 연구 질문에 답변할 수 있는
research agent를 구현합니다.
"""

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import tavily_search, get_today_str, think_tool
from deep_research_from_scratch.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message

# ===== 설정 =====

# tool과 model 바인딩 설정
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# model 초기화
model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(model="openai:gpt-4.1-mini")
compress_model = init_chat_model(model="openai:gpt-4.1", max_tokens=32000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000

# ===== AGENT NODE =====

def llm_call(state: ResearcherState):
    """현재 state를 분석하고 다음 행동을 결정합니다.

    model은 현재 대화 state를 분석하고 다음 중 하나를 결정합니다:
    1. 더 많은 정보를 수집하기 위해 검색 tool 호출
    2. 수집된 정보를 바탕으로 최종 답변 제공

    model의 응답으로 업데이트된 state를 반환합니다.
    """
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    """이전 LLM 응답의 모든 tool 호출을 실행합니다.

    이전 LLM 응답의 모든 tool 호출을 실행합니다.
    tool 실행 결과로 업데이트된 state를 반환합니다.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # 모든 tool 호출 실행
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # tool message 출력 생성
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """연구 결과를 간결한 요약으로 압축합니다.

    모든 연구 message와 tool 출력을 가져와서
    supervisor의 의사결정에 적합한 압축된 요약을 생성합니다.
    """

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    # tool과 AI message에서 원본 노트 추출
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== 라우팅 로직 =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """연구를 계속할지 최종 답변을 제공할지 결정합니다.

    LLM이 tool 호출을 했는지 여부에 따라 agent가 연구 루프를 계속할지
    최종 답변을 제공할지 결정합니다.

    Returns:
        "tool_node": tool 실행 계속
        "compress_research": 중지하고 연구 압축
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # LLM이 tool 호출을 하면 tool 실행 계속
    if last_message.tool_calls:
        return "tool_node"
    # 그렇지 않으면 최종 답변 제공
    return "compress_research"

# ===== GRAPH 구성 =====

# agent 워크플로우 구축
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# graph에 node 추가
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# node를 연결하기 위한 edge 추가
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # 연구 루프 계속
        "compress_research": "compress_research", # 최종 답변 제공
    },
)
agent_builder.add_edge("tool_node", "llm_call") # 추가 연구를 위해 루프 복귀
agent_builder.add_edge("compress_research", END)

# agent 컴파일
researcher_agent = agent_builder.compile()
