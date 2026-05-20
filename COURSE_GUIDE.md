# MIDAS × Scientific-Agent-Skills
## 공학도를 위한 AI/ML 기반 QSPR 논문 작성 실습 가이드

> **대상**: 재료/화학/디스플레이 공학 전공자  
> **목표**: Claude Code + Scientific-Agent-Skills 140개 스킬을 활용해  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SMILES → 기술자 → ML → 스크리닝 → 논문 초안까지 완성  
> **기반 사례**: OLED 박막봉지용 저유전율 유기막 소재 연구 (MIDAS_v01)

---

## 전체 흐름 (4 파이프라인)

```
Pipeline 1              Pipeline 2              Pipeline 3              Pipeline 4
────────────────        ────────────────        ────────────────        ────────────────
문헌·데이터 수집        피처 엔지니어링          ML 모델링·스크리닝       논문 초안 생성
paper-lookup            rdkit                   scikit-learn            scientific-writing
bgpt-paper-search       datamol                 shap                    scientific-visualization
literature-review       molfeat                 statistical-analysis    scientific-schematics
database-lookup         exploratory-data-       pymoo (다목적)           markdown-mermaid-
                        analysis                                        writing
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
  dataset.csv          feature_engine.py        ml_engine.py            paper_engine.py
  (SMILES + 물성값)    (X matrix, 233 feat)     (모델 + CV + AD)        (IMRAD 초안)
                              └────── MIDAS_v02 자동화 영역 ──────┘
```

---

## Pipeline 1 — 문헌·데이터 수집

### 왜 이 단계가 중요한가

ML 모델의 품질은 데이터 품질에 종속됩니다.  
MIDAS_v01은 3개 소스에서 263개 분자를 수집했습니다:
- **CRC Handbook**: 193개 — 소분자 유전율 실험값 수동 정리
- **PubChem REST API**: ~50개 — 분자명 → Canonical SMILES 자동 취득
- **논문 직접 수집**: 30개 — BCB, Parylene, POSS, 6FDA-PI 등 Low-k 소재

### 사용 스킬

#### `paper-lookup` — 학술 DB 10개 동시 검색
```
사용 시점: "이 물성에 대한 실험 데이터가 있는 논문 찾기"
커버 DB: PubMed, arXiv, Semantic Scholar, OpenAlex, Crossref,
         bioRxiv, medRxiv, CORE, PMC, Unpaywall
```
**실습 명령 예시 (유전율)**:
```
"dielectric constant organic thin film OLED 관련 논문에서 
실험 측정값이 포함된 논문을 Semantic Scholar와 arXiv에서 찾아줘"
```

**실습 명령 예시 (다른 물성 적용 시)**:
```
"water vapor transmission rate OLED encapsulation 
논문에서 측정값 포함된 것 PubMed + OpenAlex로 검색해줘"
```

---

#### `bgpt-paper-search` — 논문 전문에서 실험값 직접 추출
```
사용 시점: 제목/초록만으로는 부족할 때 — 표, 본문의 수치 필요
반환값: 25개 이상 필드 (방법, 결과, 샘플수, 오차, 결론 등)
```
**실습 명령 예시**:
```
"parylene low dielectric constant measurement 논문에서
k값, 측정 주파수, 필름 두께, 측정 방법을 추출해줘"
```

---

#### `literature-review` — 체계적 문헌 검토 + 인용 정리
```
사용 시점: Introduction 섹션 작성 전 배경 지식 정리
출력: 인증된 인용이 포함된 마크다운 문서
```
**실습 명령 예시**:
```
"저유전율 유기 박막 재료 연구 동향에 대해
최근 5년 논문 기반으로 체계적 문헌 검토 수행해줘.
주요 물성값 범위와 연구 갭도 정리해줘"
```

---

#### 데이터 수집 후 체크리스트
- [ ] SMILES 컬럼 존재 여부 확인
- [ ] 타겟 물성 컬럼 (예: `k_exp`) 확인
- [ ] 측정 조건 통일 여부 (온도, 주파수, 상태)
- [ ] 이상값(outlier) 제거 기준 설정
- [ ] 최소 30개 이상 (ML 의미 있는 최소 기준)

---

## Pipeline 2 — 피처 엔지니어링 (SMILES → 분자 기술자)

### 왜 이 단계가 중요한가

ML은 "숫자"만 입력받습니다. SMILES 문자열을 수치 벡터로 변환해야 합니다.  
MIDAS는 RDKit의 **233개 기술자**를 계산하고, 그 중 상위 N개를 선택합니다.

### 사용 스킬

#### `rdkit` — 분자 기술자 계산의 핵심
```
사용 시점: SMILES → 수치 기술자 변환 전 과정
주요 기능: MolWt, LogP, TPSA, HBD/HBA, 링 개수, 방향족성, 
           Morgan 핑거프린트, 3D 좌표
```
**MIDAS feature_engine.py에서 실제 사용된 방식**:
```python
from rdkit.Chem import Descriptors
# 233개 기술자 일괄 계산
desc_dict = Descriptors.CalcMolDescriptors(mol)
```

**실습 명령 예시**:
```
"SMILES 'C[Si](C)(C)O[Si](C)(C)C' 로 RDKit 분자 기술자를 
모두 계산하고, 유전율과 상관관계가 높을 것 같은 
기술자 10개를 골라서 이유를 설명해줘"
```

---

#### `datamol` — RDKit 래퍼, 더 쉬운 인터페이스
```
사용 시점: 대량 SMILES 전처리 (표준화, 중복 제거, 유효성 검사)
```
**실습 명령 예시**:
```
"dataset.csv의 smiles 컬럼을 datamol로 전처리해줘.
무효 SMILES 제거, 정규화, 중복 제거까지"
```

---

#### `exploratory-data-analysis` — 기술자 분포 시각화
```
사용 시점: 피처 선택 전 데이터 파악
출력: 분포, 상관관계 히트맵, 이상값 탐지
```
**실습 명령 예시**:
```
"계산된 233개 기술자에 대해 EDA를 수행해줘.
k_exp와 상관관계 Top-20 기술자를 막대그래프로 보여주고,
다중공선성 문제가 있는 기술자 쌍도 찾아줘"
```

---

#### 피처 엔지니어링 체크리스트
- [ ] 무효 SMILES 제거 후 남은 분자 수 확인
- [ ] NaN/Inf 기술자 처리
- [ ] 분산 0인 기술자 제거
- [ ] 선택된 피처 이름 저장 (스크리닝 단계에서 동일 피처 사용 필수)

---

## Pipeline 3 — ML 모델링 · 스크리닝

### 왜 이 단계가 중요한가

모델 하나가 아닌 **4종 앙상블**을 학습하고, 교차검증으로 과적합을 방지합니다.  
Leverage 기반 **적용가능 도메인(AD)** 평가로 예측 신뢰도를 정량화합니다.

### MIDAS_v01/v02 자동화 범위

| 단계 | MIDAS 함수 | 수동 작업 |
|------|-----------|---------|
| 데이터 분할 | `split_data()` | 없음 (자동) |
| 모델 학습 | `train_model()` × 4 | 없음 |
| 교차검증 | `cross_validate()` | 없음 |
| 성능 평가 | `evaluate()` | 결과 해석 |
| 스크리닝 | `screen()` | 후보 분자 제공 |
| AD 판정 | Leverage h-통계 | 없음 |

### 보완 스킬

#### `scikit-learn` — ML 모델 커스터마이징
```
사용 시점: MIDAS 기본 4종(GBR/RF/Ridge/GPR) 외 모델 추가 시
예: XGBoost, SVM, Neural Network 비교
```
**실습 명령 예시**:
```
"MIDAS ml_engine.py에 XGBoost와 SVR 모델을 추가하고,
5-fold CV R² 기준으로 5개 모델을 비교하는 표를 만들어줘"
```

---

#### `shap` — 모델 예측 근거 설명
```
사용 시점: "왜 이 분자의 유전율이 낮게 예측되었나?" 를 설명해야 할 때
출력: SHAP 값 기반 피처 기여도 시각화
```
**실습 명령 예시**:
```
"best_model(GBR)에 shap을 적용해서 
예측에 가장 중요한 분자 기술자 Top-10의 
waterfall plot을 그려줘"
```

---

#### `pymoo` — 다목적 최적화 (Goal 1 확장용)
```
사용 시점: 단일 물성이 아닌 복수 KPI 동시 최적화
예: k < 2.4 이면서 모듈러스 > 2 GPa 인 Pareto 최적 분자 탐색
알고리즘: NSGA-II, NSGA-III
```
**실습 명령 예시**:
```
"pymoo NSGA-II를 사용해서 
목적함수1: k 최소화 (유전율), 목적함수2: 모듈러스 최대화
Pareto front를 시각화해줘"
```

---

#### `statistical-analysis` — 모델 성능 통계 검증
```
사용 시점: R², RMSE 외에 신뢰구간, 통계적 유의성 필요 시
```

---

## Pipeline 4 — 논문 초안 생성

### 왜 이 단계가 중요한가

MIDAS의 `paper_engine.py`가 기초 초안(Abstract~Conclusion)을 생성하지만,  
**출판 수준의 논문**이 되려면 Scientific-Agent-Skills로 보완이 필요합니다.

### MIDAS paper_engine.py 생성 범위

```
자동 생성:
  - Abstract (모델 성능, 스크리닝 결과 포함)
  - Introduction (물성 배경, 연구 목적)
  - Methods (기술자 계산, ML 알고리즘)
  - Results (CV 결과, 피처 중요도, 스크리닝 순위)
  - Conclusion (요약, 의의, 한계)

미생성 → 스킬로 보완:
  - 그래픽 초록 (graphical abstract)
  - 피어리뷰 수준의 문장 다듬기
  - 저널별 포맷 적용
  - 인용 검증
```

### 사용 스킬

#### `scientific-writing` — IMRAD 논문 작성 원칙
```
사용 시점: MIDAS 초안을 출판용으로 업그레이드
핵심 원칙: 
  - 불릿 포인트 → 완전한 문장으로 변환 (필수)
  - 2단계 프로세스: 아웃라인 작성 → 완전한 단락으로 확장
  - 능동태/과거형 혼용 규칙
```
**실습 명령 예시**:
```
"MIDAS가 생성한 Abstract를 scientific-writing 원칙에 따라
완전한 단락으로 다듬어줘. 
연구 목적 → 방법 → 결과 → 의의 흐름을 유지하면서
250단어 이내로 써줘"
```

---

#### `scientific-visualization` — 논문용 그래프 생성
```
사용 시점: ML 결과를 논문 품질의 그림으로 변환
주요 그림: 
  - Parity plot (예측값 vs 실험값)
  - 피처 중요도 막대 그래프
  - Pareto front (다목적 최적화)
  - Applicability domain 산점도
```
**실습 명령 예시**:
```
"GBR 모델의 test set 예측값과 실험값으로 parity plot을 그려줘.
R²와 RMSE를 그래프 안에 표시하고, 
AD 외부 샘플은 다른 색으로 표시해줘.
Nature Materials 스타일로"
```

---

#### `scientific-schematics` — 연구 플로우 다이어그램
```
사용 시점: 연구 방법론을 시각화 (필수 - 모든 논문에 권장)
```
**실습 명령 예시**:
```
"QSPR 연구 워크플로우 다이어그램을 만들어줘:
데이터 수집 → SMILES 입력 → RDKit 기술자 → 피처 선택 
→ 앙상블 ML → 교차검증 → AD 평가 → 스크리닝 → 논문 생성"
```

---

#### `peer-review` — 논문 셀프 리뷰
```
사용 시점: 제출 전 최종 검토
```
**실습 명령 예시**:
```
"작성된 논문 초안에 대해 Materials Science 저널 심사자 관점에서
주요 약점과 보완 사항을 리뷰해줘"
```

---

## 실습 시나리오: 처음부터 끝까지

### 사례: 유전율 저유전율 유기막 QSPR 연구 (MIDAS_v01 재현)

```
Step 1  [paper-lookup + bgpt-paper-search]
        "low dielectric constant organic polymer 실험값 포함 논문 검색
         k값, SMILES, 측정 조건 추출"
        → dataset.csv 구축 목표: 30개 이상 분자

Step 2  [rdkit + exploratory-data-analysis]  
        "dataset.csv의 SMILES로 233개 기술자 계산
         k_exp와 상관관계 분석, 분포 시각화"
        → feature matrix X, target y

Step 3  [MIDAS_v02 실행]
        python -m streamlit run app.py
        → S2 피처 선택 → S3 ML 학습 → S4 스크리닝
        → S5 논문 초안 자동 생성

Step 4  [shap]
        "best model 예측의 분자 기술자 기여도 분석
         어떤 구조적 특성이 저유전율을 결정하는가?"

Step 5  [scientific-writing + scientific-visualization]
        "MIDAS 초안을 IMRAD 완전 문장으로 업그레이드
         parity plot, 피처 중요도 그래프 생성"

Step 6  [peer-review]
        "최종 논문 검토 및 보완"
```

---

## 다른 물성 적용 가이드

### 새 물성 추가 순서

```
1. properties/<물성명>/config.yaml 작성
   ─ target_column, direction, screening_threshold, log_transform

2. properties/<물성명>/dataset.csv 배치
   ─ 최소 30행, smiles + target 컬럼 필수

3. app.py 재시작 → 사이드바에 자동 등록

4. 논문 초안: properties/<물성명>/paper_template.py 작성 (선택)
   ─ 없으면 _paper_generic.py가 자동 사용됨
```

### 물성별 추천 스킬 조합

| 물성 | 데이터 소스 스킬 | 피처 특이사항 | 모델링 특이사항 |
|------|----------------|--------------|----------------|
| 유전율 (완성) | paper-lookup, CRC DB | 불소·실리콘 기술자 강조 | log 변환, lower_better |
| WVTR | bgpt-paper-search | 극성·공극 관련 기술자 | log 변환 필수 (범위 넓음) |
| 모듈러스 | paper-lookup (재료 DB) | 분자량, 가교 밀도 관련 | log 변환 고려 |
| 밴드갭 | paper-lookup (arXiv) | 방향족, 공역 길이 기술자 | 직접 회귀 |
| 용해도 | paper-lookup (AqSolDB) | TPSA, HBD/HBA | log 변환 없음, higher_better |
| 열전도율 | literature-review | 원소 조성, 밀도 기술자 | 데이터 희소 → GPR 유리 |

---

## 수강생 체크리스트

### Pipeline 1 완료 기준
- [ ] 30개 이상 분자 수집
- [ ] 모든 SMILES RDKit 유효성 통과
- [ ] 출처 (source, doi) 컬럼 포함

### Pipeline 2 완료 기준
- [ ] X matrix shape 확인 (n_molecules × n_features)
- [ ] NaN 없음 확인
- [ ] 선택된 피처 이름 목록 저장

### Pipeline 3 완료 기준
- [ ] 최소 2개 이상 모델 학습
- [ ] Test R² > 0.6 (최소 기준)
- [ ] CV R² ≈ Test R² (과적합 없음)
- [ ] 스크리닝 결과 AD 판정 포함

### Pipeline 4 완료 기준
- [ ] Abstract: 완전한 단락, 250단어 이내
- [ ] Parity plot 포함
- [ ] 피처 중요도 그래프 포함
- [ ] 연구 한계 명시

---

## 참고 자료

- **MIDAS_v01**: `C:\Users\nackm\MIDAS_v01\` — 유전율 완성 사례
- **MIDAS_v02**: `C:\Users\nackm\MIDAS_v02\` — 다물성 확장 플랫폼
- **Skills 위치**: `C:\Users\nackm\.claude\skills\`
- **Delaney ESOL**: solubility 검증용 51개 분자 데이터셋

---

*최종 업데이트: 2026-05-14*  
*작성: NEXTIO × Claude Code*
