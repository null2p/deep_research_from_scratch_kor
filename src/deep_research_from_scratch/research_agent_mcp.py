
"""MCP 통합 Research Agent.

이 모듈은 Model Context Protocol (MCP) server와 통합하여 tool과 resource에 접근하는
research agent를 구현합니다. Agent는 로컬 문서 연구 및 분석을 위해 MCP filesystem
server를 사용하는 방법을 보여줍니다.

주요 기능:
- Tool 접근을 위한 MCP server 통합
- 동시 tool 실행을 위한 비동기 작업 (MCP protocol에서 필수)
- 로컬 문서 연구를 위한 Filesystem 작업
- 권한 검사를 통한 안전한 디렉터리 접근
- 효율적인 처리를 위한 연구 압축
- LangGraph Platform 호환성을 위한 지연 MCP client 초기화
"""

import os

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from deep_research_from_scratch.prompts import research_agent_prompt_with_mcp, compress_research_system_prompt, compress_research_human_message
from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import get_today_str, think_tool, get_current_dir

# ===== 설정 =====

# Filesystem 접근을 위한 MCP server 설정
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # 필요시 자동 설치
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files")  # 연구 문서 경로
        ],
        "transport": "stdio"  # stdin/stdout을 통한 통신
    }
}

# 전역 client 변수 - 지연 초기화됨
_client = None

def get_mcp_client():
    """LangGraph Platform 문제를 방지하기 위해 MCP client를 지연 초기화하여 반환."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client

# Model 초기화
compress_model = init_chat_model(model="openai:gpt-4.1", max_tokens=32000)
model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")

# ===== AGENT NODE =====

async def llm_call(state: ResearcherState):
    """MCP 통합을 통해 현재 state를 분석하고 tool 사용을 결정.

    이 node는:
    1. MCP server에서 사용 가능한 tool을 검색
    2. Tool을 language model에 바인딩
    3. 사용자 입력을 처리하고 tool 사용을 결정

    Model 응답으로 업데이트된 state를 반환.
    """
    # MCP server에서 사용 가능한 tool 가져오기
    client = get_mcp_client()
    mcp_tools = await client.get_tools()

    # 로컬 문서 접근을 위한 MCP tool 사용
    tools = mcp_tools + [think_tool]

    # Tool 바인딩으로 model 초기화
    model_with_tools = model.bind_tools(tools)

    # System prompt로 사용자 입력 처리
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt_with_mcp.format(date=get_today_str()))] + state["researcher_messages"]
            )
        ]
    }

async def tool_node(state: ResearcherState):
    """MCP tool을 사용하여 tool 호출 실행.

    이 node는:
    1. 마지막 메시지에서 현재 tool 호출을 검색
    2. 비동기 작업을 사용하여 모든 tool 호출 실행 (MCP에 필수)
    3. 형식화된 tool 결과 반환

    참고: MCP는 MCP server subprocess와의 프로세스 간 통신으로 인해
    비동기 작업이 필요합니다. 이는 불가피합니다.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        """모든 tool 호출을 실행. MCP tool은 비동기 실행이 필요."""
        # MCP server에서 새로운 tool 참조 가져오기
        client = get_mcp_client()
        mcp_tools = await client.get_tools()
        tools = mcp_tools + [think_tool]
        tools_by_name = {tool.name: tool for tool in tools}

        # Tool 호출 실행 (안정성을 위해 순차적으로)
        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "think_tool":
                # think_tool은 동기식, 일반 invoke 사용
                observation = tool.invoke(tool_call["args"])
            else:
                # MCP tool은 비동기식, ainvoke 사용
                observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # 결과를 tool 메시지로 형식화
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    messages = await execute_tools()

    return {"researcher_messages": messages}

def compress_research(state: ResearcherState) -> dict:
    """연구 결과를 간결한 요약으로 압축.

    모든 연구 메시지와 tool 출력을 가져와서 추가 처리나 보고에
    적합한 압축된 요약을 생성합니다.

    이 함수는 think_tool 호출을 필터링하고 MCP tool의 실질적인
    파일 기반 연구 콘텐츠에 중점을 둡니다.
    """

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]

    response = compress_model.invoke(messages)

    # Tool 및 AI 메시지에서 원시 노트 추출
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
    """Tool 실행을 계속할지 연구를 압축할지 결정.

    LLM이 tool을 호출했는지 여부에 따라 tool 실행을 계속할지
    연구를 압축할지 결정합니다.
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # Tool이 호출되었으면 tool 실행 계속
    if last_message.tool_calls:
        return "tool_node"
    # 그렇지 않으면 연구 결과 압축
    return "compress_research"

# ===== GRAPH 구성 =====

# Agent workflow 구축
agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Graph에 node 추가
agent_builder_mcp.add_node("llm_call", llm_call)
agent_builder_mcp.add_node("tool_node", tool_node)
agent_builder_mcp.add_node("compress_research", compress_research)

# Node를 연결하는 edge 추가
agent_builder_mcp.add_edge(START, "llm_call")
agent_builder_mcp.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",        # Tool 실행 계속
        "compress_research": "compress_research",  # 연구 결과 압축
    },
)
agent_builder_mcp.add_edge("tool_node", "llm_call")  # 추가 처리를 위해 다시 루프
agent_builder_mcp.add_edge("compress_research", END)

# Agent 컴파일
agent_mcp = agent_builder_mcp.compile()
