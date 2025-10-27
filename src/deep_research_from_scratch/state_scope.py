
"""Research Scoping을 위한 State 정의 및 Pydantic Schema.

이는 research agent scoping workflow에서 사용되는 state 객체와 구조화된 schema를 정의하며,
researcher state 관리 및 output schema를 포함합니다.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE 정의 =====

class AgentInputState(MessagesState):
    """전체 agent를 위한 Input state - 사용자 입력의 message만 포함합니다."""
    pass

class AgentState(MessagesState):
    """
    전체 multi-agent research 시스템을 위한 Main state.

    Research coordination을 위한 추가 field로 MessagesState를 확장합니다.
    참고: subgraph와 main workflow 간의 적절한 state 관리를 위해
    일부 field가 서로 다른 state class에 중복되어 있습니다.
    """

    # 사용자 대화 기록에서 생성된 Research brief
    research_brief: Optional[str]
    # Coordination을 위해 supervisor agent와 교환되는 message
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Research 단계에서 수집된 처리되지 않은 원시 research note
    raw_notes: Annotated[list[str], operator.add] = []
    # Report 생성을 위해 처리되고 구조화된 note
    notes: Annotated[list[str], operator.add] = []
    # 최종 형식화된 research report
    final_report: str

# ===== 구조화된 OUTPUT SCHEMA =====

class ClarifyWithUser(BaseModel):
    """사용자 명확화 결정 및 질문을 위한 Schema."""

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
    """구조화된 research brief 생성을 위한 Schema."""

    research_brief: str = Field(
        description="Research를 안내하는 데 사용될 research 질문.",
    )
