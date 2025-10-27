
"""
Research Agent를 위한 State 정의 및 Pydantic Schema

이 모듈은 research agent workflow에서 사용되는 state 객체와 구조화된 schema를 정의하며,
researcher state 관리 및 output schema를 포함합니다.
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== STATE 정의 =====

class ResearcherState(TypedDict):
    """
    Message 기록 및 research metadata를 포함하는 research agent를 위한 State.

    이 state는 researcher의 대화, tool 호출 제한을 위한 iteration 횟수,
    조사 중인 research 주제, 압축된 findings,
    그리고 상세 분석을 위한 원시 research note를 추적합니다.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    최종 research 결과를 포함하는 research agent를 위한 Output state.

    이는 압축된 research findings와 research 프로세스의 모든 원시 note를 포함한
    research 프로세스의 최종 output을 나타냅니다.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== 구조화된 OUTPUT SCHEMA =====

class ClarifyWithUser(BaseModel):
    """Scoping 단계에서 사용자 명확화 결정을 위한 Schema."""
    need_clarification: bool = Field(
        description="사용자에게 명확화 질문을 해야 하는지 여부.",
    )
    question: str = Field(
        description="report 범위를 명확히 하기 위해 사용자에게 할 질문",
    )
    verification: str = Field(
        description="사용자가 필요한 정보를 제공한 후 research를 시작할 것임을 확인하는 메시지.",
    )

class ResearchQuestion(BaseModel):
    """Research brief 생성을 위한 Schema."""
    research_brief: str = Field(
        description="Research를 안내하는 데 사용될 research 질문.",
    )

class Summary(BaseModel):
    """웹페이지 콘텐츠 요약을 위한 Schema."""
    summary: str = Field(description="웹페이지 콘텐츠의 간결한 요약")
    key_excerpts: str = Field(description="콘텐츠에서 중요한 인용문 및 발췌")
