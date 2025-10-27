
"""
Multi-Agent Research Supervisor를 위한 State 정의

이 모듈은 multi-agent research supervisor workflow에서 사용되는 state 객체와 tool을 정의하며,
coordination state 및 research tool을 포함합니다.
"""

import operator
from typing_extensions import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    """
    Multi-agent research supervisor를 위한 State.

    supervisor와 research agent 간의 coordination을 관리하며, research 진행 상황을 추적하고
    여러 sub-agent로부터 수집된 findings를 축적합니다.
    """

    # coordination 및 의사결정을 위해 supervisor와 교환되는 message
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 전체 research 방향을 안내하는 상세한 research brief
    research_brief: str
    # 최종 report 생성을 위해 처리되고 구조화된 note
    notes: Annotated[list[str], operator.add] = []
    # 수행된 research iteration 횟수를 추적하는 카운터
    research_iterations: int = 0
    # sub-agent research에서 수집된 처리되지 않은 원시 research note
    raw_notes: Annotated[list[str], operator.add] = []

@tool
class ConductResearch(BaseModel):
    """특정 sub-agent에게 research task를 위임하기 위한 Tool."""
    research_topic: str = Field(
        description="research할 주제. 단일 주제여야 하며, 높은 수준의 세부 정보로 설명되어야 합니다 (최소 한 단락).",
    )

@tool
class ResearchComplete(BaseModel):
    """Research 프로세스가 완료되었음을 나타내기 위한 Tool."""
    pass
