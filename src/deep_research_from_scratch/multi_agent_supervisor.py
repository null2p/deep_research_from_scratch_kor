
"""여러 전문 agent를 조정하는 multi-agent supervisor.

이 모듈은 다음과 같은 supervisor 패턴을 구현합니다:
1. Supervisor agent가 research 활동을 조정하고 작업을 위임
2. 여러 researcher agent가 특정 하위 주제에 대해 독립적으로 작업
3. 결과를 집계하고 압축하여 최종 보고서 작성

Supervisor는 병렬 research 실행을 사용하여 효율성을 향상시키면서
각 research 주제에 대해 격리된 context window를 유지합니다.
"""

import asyncio

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import lead_researcher_prompt
from deep_research_from_scratch.research_agent import researcher_agent
from deep_research_from_scratch.state_multi_agent_supervisor import (
    SupervisorState, 
    ConductResearch, 
    ResearchComplete
)
from deep_research_from_scratch.utils import get_today_str, think_tool

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Supervisor message history의 ToolMessage 객체에서 research note를 추출합니다.

    이 함수는 sub-agent가 ToolMessage content로 반환하는 압축된 research 결과를 검색합니다.
    Supervisor가 ConductResearch tool call을 통해 sub-agent에게 research를 위임하면,
    각 sub-agent는 압축된 결과를 ToolMessage의 content로 반환합니다. 이 함수는
    모든 ToolMessage content를 추출하여 최종 research note를 컴파일합니다.

    Args:
        messages: Supervisor의 대화 기록에서 가져온 message 목록

    Returns:
        ToolMessage 객체에서 추출한 research note 문자열 목록
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

# Jupyter 환경에서 async 호환성 보장
try:
    import nest_asyncio
    # Jupyter/IPython 환경에서 실행 중인 경우에만 적용
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Jupyter가 아니면 nest_asyncio 불필요
except ImportError:
    pass  # nest_asyncio를 사용할 수 없으면 없이 진행


# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# System 상수
# 개별 researcher agent의 최대 tool call 반복 횟수
# 이는 무한 루프를 방지하고 주제당 research 깊이를 제어합니다
max_researcher_iterations = 6 # think_tool + ConductResearch 호출

# Supervisor가 실행할 수 있는 최대 동시 research agent 수
# 병렬 research 작업을 제한하기 위해 lead_researcher_prompt에 전달됩니다
max_concurrent_researchers = 3

# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Research 활동을 조정합니다.

    Research brief와 현재 진행 상황을 분석하여 다음을 결정합니다:
    - 어떤 research 주제를 조사해야 하는지
    - 병렬 research를 수행할지 여부
    - Research가 완료되었는지 여부

    Args:
        state: Message와 research 진행 상황이 포함된 현재 supervisor state

    Returns:
        업데이트된 state와 함께 supervisor_tools 노드로 진행하는 Command
    """
    supervisor_messages = state.get("supervisor_messages", [])

    # 현재 날짜와 제약 조건이 포함된 system message 준비
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 다음 research 단계에 대한 결정
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Supervisor의 결정을 실행합니다 - research를 수행하거나 프로세스를 종료합니다.

    다음을 처리합니다:
    - 전략적 성찰을 위한 think_tool 호출 실행
    - 다양한 주제에 대한 병렬 research agent 실행
    - Research 결과 집계
    - Research 완료 시점 결정

    Args:
        state: Message와 반복 횟수가 포함된 현재 supervisor state

    Returns:
        Supervision을 계속하거나, 프로세스를 종료하거나, 오류를 처리하는 Command
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 단일 return 패턴을 위한 변수 초기화
    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"  # 기본 다음 단계
    should_end = False

    # 먼저 종료 기준 확인
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END

    else:
        # 다음 단계를 결정하기 전에 모든 tool call 실행
        try:
            # think_tool 호출과 ConductResearch 호출 분리
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            # think_tool 호출 처리 (동기)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # ConductResearch 호출 처리 (비동기)
            if conduct_research_calls:
                # 병렬 research agent 실행
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                # 모든 research 완료 대기
                tool_results = await asyncio.gather(*coros)

                # Research 결과를 tool message로 포맷
                # 각 sub-agent는 result["compressed_research"]에 압축된 research 결과를 반환
                # 이 압축된 research를 ToolMessage의 content로 작성하여,
                # supervisor가 나중에 get_notes_from_tool_calls()를 통해 이러한 결과를 검색할 수 있도록 함
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # 모든 research의 raw note 집계
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]

        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    # 적절한 state 업데이트와 함께 단일 return 지점
    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )

# ===== GRAPH CONSTRUCTION =====

# Supervisor graph 구축
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()
