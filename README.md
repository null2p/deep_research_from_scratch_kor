# 🧱 Deep Research From Scratch

Deep research는 가장 인기 있는 agent 애플리케이션 중 하나로 자리잡았습니다. [OpenAI](https://openai.com/index/introducing-deep-research/), [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system), [Perplexity](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research), [Google](https://gemini.google/overview/deep-research/?hl=en) 모두 [다양한 소스](https://www.anthropic.com/news/research)의 컨텍스트를 활용하여 포괄적인 리포트를 생성하는 deep research 제품을 출시했습니다. 또한 많은 [오픈](https://huggingface.co/blog/open-deep-research) [소스](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 구현체들도 존재합니다. 우리는 간단하고 설정 가능한 [오픈 deep researcher](https://github.com/langchain-ai/open_deep_research)를 만들었으며, 사용자가 자신의 모델, 검색 도구, MCP server를 자유롭게 사용할 수 있습니다. 이 저장소에서는 deep researcher를 처음부터 직접 구축해볼 것입니다! 우리가 만들 주요 구성 요소의 지도는 다음과 같습니다:

![overview](https://github.com/user-attachments/assets/b71727bd-0094-40c4-af5e-87cdb02123b4)

## 🚀 Quickstart

### 사전 요구사항

- **Node.js와 npx** (notebook 3의 MCP server에 필요):
```bash
# Node.js 설치 (npx 포함)
# macOS에서 Homebrew 사용:
brew install node

# Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 설치 확인:
node --version
npx --version
```

- Python 3.11 이상을 사용하는지 확인하세요.
- 이 버전은 LangGraph와의 최적의 호환성을 위해 필요합니다.
```bash
python3 --version
```
- [uv](https://docs.astral.sh/uv/) package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 새 uv 버전을 사용하도록 PATH 업데이트
export PATH="/Users/$USER/.local/bin:$PATH"
```

### 설치

1. 저장소를 클론합니다:
```bash
git clone https://github.com/langchain-ai/deep_research_from_scratch
cd deep_research_from_scratch
```

2. 패키지와 의존성을 설치합니다 (가상 환경을 자동으로 생성하고 관리합니다):
```bash
uv sync
```

3. 프로젝트 루트에 API key가 포함된 `.env` 파일을 생성합니다:
```bash
# .env 파일 생성
touch .env
```

`.env` 파일에 API key를 추가합니다:
```env
# 외부 검색을 사용하는 research agent에 필요
TAVILY_API_KEY=your_tavily_api_key_here

# 모델 사용에 필요
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 선택사항: 평가 및 tracing을 위한 설정
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep_research_from_scratch
```

4. uv를 사용하여 notebook이나 코드를 실행합니다:
```bash
# Jupyter notebook을 직접 실행
uv run jupyter notebook

# 또는 원하는 경우 가상 환경을 활성화
source .venv/bin/activate  # Windows: .venv\Scripts\activate
jupyter notebook
```

## 배경

Research는 개방형 작업입니다. 사용자 요청에 답변하기 위한 최선의 전략을 미리 쉽게 알 수 없습니다. 요청마다 서로 다른 research 전략과 다양한 수준의 검색 깊이가 필요할 수 있습니다.

[Agent](https://langchain-ai.github.io/langgraph/tutorials/workflows/#agent)는 research에 매우 적합합니다. 중간 결과를 활용하여 탐색을 안내하면서 다양한 전략을 유연하게 적용할 수 있기 때문입니다. Open deep research는 다음 세 단계 프로세스의 일부로 agent를 사용하여 research를 수행합니다:

1. **Scope** – research 범위 명확화
2. **Research** – research 수행
3. **Write** – 최종 리포트 작성

## 📝 구성

이 저장소는 처음부터 deep research 시스템을 구축하는 5개의 튜토리얼 notebook을 포함합니다:

### 📚 튜토리얼 Notebook

#### 1. 사용자 명확화 및 Brief 생성 (`notebooks/1_scoping.ipynb`)
**목적**: Research 범위를 명확히 하고 사용자 입력을 구조화된 research brief로 변환

**핵심 개념**:
- **사용자 명확화**: Structured output을 사용하여 사용자에게 추가 컨텍스트가 필요한지 판단
- **Brief 생성**: 대화를 상세한 research 질문으로 변환
- **LangGraph Command**: 흐름 제어 및 상태 업데이트를 위한 Command 시스템 사용
- **Structured Output**: 신뢰할 수 있는 의사결정을 위한 Pydantic schema

**구현 하이라이트**:
- 2단계 워크플로우: 명확화 → brief 생성
- Hallucination을 방지하는 structured output 모델 (`ClarifyWithUser`, `ResearchQuestion`)
- 명확화 필요성에 기반한 조건부 라우팅
- 컨텍스트에 민감한 research를 위한 날짜 인식 프롬프트

**학습 내용**: 상태 관리, structured output 패턴, 조건부 라우팅

---

#### 2. Custom Tool을 사용한 Research Agent (`notebooks/2_research_agent.ipynb`)
**목적**: 외부 검색 도구를 사용하는 반복적인 research agent 구축

**핵심 개념**:
- **Agent 아키텍처**: LLM 의사결정 노드 + 도구 실행 노드 패턴
- **순차적 도구 실행**: 신뢰할 수 있는 동기식 도구 실행
- **검색 통합**: 콘텐츠 요약 기능이 있는 Tavily 검색
- **도구 실행**: Tool calling을 사용한 ReAct 스타일 agent 루프

**구현 하이라이트**:
- 신뢰성과 단순성을 위한 동기식 도구 실행
- 검색 결과를 압축하기 위한 콘텐츠 요약
- 조건부 라우팅이 있는 반복적 research 루프
- 포괄적인 research를 위한 풍부한 프롬프트 엔지니어링

**학습 내용**: Agent 패턴, 도구 통합, 검색 최적화, research 워크플로우 설계

---

#### 3. MCP를 사용한 Research Agent (`notebooks/3_research_agent_mcp.ipynb`)
**목적**: Model Context Protocol (MCP) server를 research 도구로 통합

**핵심 개념**:
- **Model Context Protocol**: AI 도구 액세스를 위한 표준화된 프로토콜
- **MCP 아키텍처**: stdio/HTTP를 통한 client-server 통신
- **LangChain MCP Adapter**: MCP server를 LangChain 도구로 원활하게 통합
- **Local vs Remote MCP**: Transport 메커니즘 이해

**구현 하이라이트**:
- MCP server 관리를 위한 `MultiServerMCPClient`
- 구성 기반 server 설정 (파일시스템 예제)
- 도구 출력 표시를 위한 풍부한 포맷팅
- MCP 프로토콜에서 요구하는 비동기 도구 실행 (중첩된 event loop 불필요)

**학습 내용**: MCP 통합, client-server 아키텍처, 프로토콜 기반 도구 액세스

---

#### 4. Research Supervisor (`notebooks/4_research_supervisor.ipynb`)
**목적**: 복잡한 research 작업을 위한 multi-agent 조정

**핵심 개념**:
- **Supervisor 패턴**: 조정 agent + 작업 agent
- **병렬 Research**: 병렬 tool call을 사용한 독립적인 주제에 대한 동시 research agent
- **Research 위임**: 작업 할당을 위한 구조화된 도구
- **컨텍스트 격리**: 서로 다른 research 주제에 대한 별도의 컨텍스트 윈도우

**구현 하이라이트**:
- 2개 노드 supervisor 패턴 (`supervisor` + `supervisor_tools`)
- 진정한 동시성을 위한 `asyncio.gather()`를 사용한 병렬 research 실행
- 위임을 위한 구조화된 도구 (`ConductResearch`, `ResearchComplete`)
- 병렬 research 지침이 포함된 향상된 프롬프트
- Research 집계 패턴에 대한 포괄적인 문서화

**학습 내용**: Multi-agent 패턴, 병렬 처리, research 조정, 비동기 오케스트레이션

---

#### 5. 완전한 Multi-Agent Research 시스템 (`notebooks/5_full_agent.ipynb`)
**목적**: 모든 구성 요소를 통합한 완전한 end-to-end research 시스템

**핵심 개념**:
- **3단계 아키텍처**: Scope → Research → Write
- **시스템 통합**: Scoping, multi-agent research, 리포트 생성 결합
- **상태 관리**: Subgraph 간 복잡한 상태 흐름
- **End-to-End 워크플로우**: 사용자 입력부터 최종 research 리포트까지

**구현 하이라이트**:
- 적절한 상태 전환을 갖춘 완전한 워크플로우 통합
- Output schema가 있는 supervisor 및 researcher subgraph
- Research 종합을 통한 최종 리포트 생성
- 명확화를 위한 thread 기반 대화 관리

**학습 내용**: 시스템 아키텍처, subgraph 구성, end-to-end 워크플로우

---

### 🎯 핵심 학습 성과

- **Structured Output**: 신뢰할 수 있는 AI 의사결정을 위한 Pydantic schema 사용
- **비동기 오케스트레이션**: 병렬 조정을 위한 비동기 패턴 vs 동기식 단순성의 전략적 사용
- **Agent 패턴**: ReAct 루프, supervisor 패턴, multi-agent 조정
- **검색 통합**: 외부 API, MCP server, 콘텐츠 처리
- **워크플로우 설계**: 복잡한 다단계 프로세스를 위한 LangGraph 패턴
- **상태 관리**: Subgraph와 노드 간 복잡한 상태 흐름
- **프로토콜 통합**: MCP server 및 도구 생태계

각 notebook은 이전 개념을 기반으로 구축되며, 지능적인 scoping과 조정된 실행으로 복잡하고 다면적인 research 쿼리를 처리할 수 있는 프로덕션 준비가 완료된 deep research 시스템으로 완성됩니다. 
