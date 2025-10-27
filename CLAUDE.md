# Deep Research From Scratch - 저장소 가이드

## 저장소 구조

이 저장소는 LangGraph를 사용하여 처음부터 포괄적인 deep research 시스템을 구축합니다. 다양한 구성 요소와 패턴을 보여주는 5개의 튜토리얼 notebook으로 진행됩니다.

```
deep_research_from_scratch/
├── notebooks/              # 대화형 튜토리얼 notebook (이 파일들을 수정하세요)
│   ├── 1_scoping.ipynb     # 사용자 명확화 및 brief 생성
│   ├── 2_research_agent.ipynb       # Custom tool을 사용한 research agent
│   ├── 3_research_agent_mcp.ipynb   # MCP server를 사용한 research agent
│   ├── 4_research_supervisor.ipynb  # Multi-agent supervisor 조정
│   ├── 5_full_agent.ipynb  # 완전한 end-to-end 시스템
│   └── utils.py            # Notebook용 공유 유틸리티
├── src/deep_research_from_scratch/  # 생성된 소스 코드 (수정하지 마세요)
│   ├── multi_agent_supervisor.py
│   ├── prompts.py
│   ├── research_agent.py
│   ├── research_agent_mcp.py
│   ├── state_*.py
│   └── utils.py
└── README.md              # 포괄적인 문서
```

## 🚨 중요한 개발 워크플로우

**`notebooks/` 디렉토리의 notebook이 유일한 진실의 원천이며 수정해야 할 유일한 파일입니다.**

`src/deep_research_from_scratch/`의 소스 코드는 `%%writefile` magic command를 사용하여 notebook에서 자동으로 생성됩니다. 작동 방식은 다음과 같습니다:

### 코드 생성 작동 방식

1. **Notebook에 `%%writefile` cell 포함**: 각 notebook은 Jupyter의 `%%writefile` magic을 사용하여 `src/`의 파일에 코드를 직접 작성
2. **Notebook은 실행 가능한 튜토리얼**: 프로덕션 코드를 생성하면서 대화형으로 개념을 시연
3. **소스 파일은 생성된 산출물**: `src/`의 `.py` 파일은 출력이지 입력이 아님

### Notebook 예제:
```python
%%writefile ../src/deep_research_from_scratch/research_agent.py

"""
Research Agent Implementation
"""
# ... 실제 구현 코드가 이어짐
```

### 개발 가이드라인

- ✅ **해야 할 것**: `notebooks/` 디렉토리의 notebook 편집
- ✅ **해야 할 것**: Notebook cell을 실행하여 소스 코드 재생성
- ✅ **해야 할 것**: Notebook을 실행하여 변경 사항 테스트
- ❌ **하지 말아야 할 것**: `src/deep_research_from_scratch/`의 파일을 직접 편집
- ❌ **하지 말아야 할 것**: `src/` 파일의 수동 변경 사항이 유지될 것으로 기대

## 시스템 아키텍처

이 시스템은 3단계 deep research 워크플로우를 구현합니다:

1. **Scope** (Notebook 1): Research 범위를 명확히 하고 구조화된 brief 생성
2. **Research** (Notebook 2-4): 다양한 agent 패턴을 사용하여 research 수행
3. **Write** (Notebook 5): 연구 결과를 포괄적인 리포트로 종합

### 주요 구성 요소

- **Scoping Agent**: 사용자 의도를 명확히 하고 research brief 생성
- **Research Agent**: Custom tool 또는 MCP server를 사용한 반복적 research
- **Supervisor Agent**: 복잡한 주제에 대해 여러 research agent를 조정
- **Full System**: 모든 구성 요소를 end-to-end 워크플로우로 통합

## 개발 빠른 시작

1. `notebooks/`의 적절한 notebook을 수정
2. 수정된 cell을 실행하여 소스 코드 재생성
3. 후속 notebook cell을 실행하여 변경 사항 테스트
4. `src/`의 생성된 코드가 자동으로 변경 사항을 반영

이 접근 방식은 대화형 튜토리얼이 권위 있는 소스로 유지되면서 자동으로 해당 Python 패키지 구조를 유지하도록 보장합니다.

## 코드 품질 및 포맷팅

### Ruff 포맷팅 검사

생성된 소스 파일 전체에서 일관된 코드 포맷팅을 유지하려면 주기적으로 ruff를 실행하세요:

```bash
# 포맷팅 문제 확인
ruff check src/

# 가능한 경우 포맷팅 문제 자동 수정
ruff check src/ --fix

# 특정 파일 확인
ruff check src/deep_research_from_scratch/research_agent.py
```

**중요**: `src/`의 소스 파일은 notebook에서 생성되므로, 모든 포맷팅 문제는 소스 파일에서 직접 수정하는 것이 아니라 notebook의 `%%writefile` cell에서 수정해야 합니다. Notebook에서 포맷팅을 수정한 후 notebook cell을 실행하여 소스 파일을 재생성하세요.

**일반적으로 필요한 포맷팅 수정:**
- **D212**: Docstring 요약이 삼중 따옴표와 같은 줄에서 시작하도록 보장
- **I001**: Import를 올바르게 정리 (표준 라이브러리 → 서드파티 → 로컬 import)
- **F401**: 사용하지 않는 import 제거
- **D415**: Docstring 요약에 마침표 추가