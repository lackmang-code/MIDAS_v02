# Scientific-Agent-Skills 카탈로그
## 136개 스킬 도메인별 분류

> 설치 위치: `C:\Users\nackm\.claude\skills\`  
> 최종 업데이트: 2026-05-14

---

## 빠른 탐색

| 번호 | 도메인 | 스킬 수 | MIDAS 관련도 |
|-----|--------|--------|-------------|
| A | 재료과학·화학정보학 | 11 | ★★★★★ |
| B | 논문작성·연구지원 | 19 | ★★★★★ |
| C | ML·통계·최적화 | 14 | ★★★★★ |
| D | 데이터처리·인프라 | 8 | ★★★☆☆ |
| E | 생명과학·유전체·단백질 | 22 | ★★☆☆☆ |
| F | 의학·임상·헬스케어 | 11 | ★★☆☆☆ |
| G | 양자컴퓨팅·물리 | 6 | ★★★☆☆ |
| H | 지구과학·지리정보 | 3 | ★☆☆☆☆ |
| I | 문서·미디어·시각화 | 8 | ★★★☆☆ |
| J | 실험자동화·클라우드 | 11 | ★★☆☆☆ |
| K | AI 에이전트·연구지원 특수 | 10 | ★★★★☆ |
| L | 경제·금융·법규 | 3 | ★☆☆☆☆ |

---

## A. 재료과학 · 화학정보학 ★★★★★

> **MIDAS 핵심 도메인** — Pipeline 2(피처) · Pipeline 3(모델링)에 직결

| 스킬 | 한줄 설명 |
|-----|---------|
| **rdkit** | SMILES/SDF 파싱, 분자 기술자 233개, 핑거프린트, 유사도, 반응 — MIDAS feature_engine 기반 |
| **datamol** | RDKit 래퍼, 더 쉬운 인터페이스 — SMILES 표준화·중복제거·전처리 |
| **molfeat** | 100+ 분자 피처라이저 (ECFP, MACCS, ChemBERTa 사전학습 모델) |
| **deepchem** | GNN 포함 분자 ML — ADMET, 독성, MoleculeNet 벤치마크 |
| **medchem** | 약물유사성 필터 (Lipinski, PAINS, 구조적 경고) — 후보 분자 우선순위화 |
| **pymatgen** | 결정 구조 (CIF/POSCAR), 상다이어그램, 밴드구조, Materials Project 연동 |
| **molecular-dynamics** | OpenMM + MDAnalysis — 분자동역학 시뮬레이션, RMSD/RMSF 분석 |
| **matchms** | 질량스펙트럼 유사도 — 대사체 동정, 스펙트럼 라이브러리 검색 |
| **diffdock** | 확산 모델 기반 단백질-리간드 도킹 (구조 기반 약물 설계) |
| **rowan** | 클라우드 분자 모델링 — pKa, 컨포머, 도킹, 단백질-리간드 코폴딩 |
| **torchdrug** | PyTorch GNN 기반 분자·단백질 모델링, 역합성 |

**MIDAS 적용 포인트**:
- `rdkit` → feature_engine.py 의 핵심
- `pymatgen` → 결정 구조 물성 (밴드갭, 격자 파라미터) 추가 시
- `molfeat` → Morgan fingerprint 기반 피처 추가 시

---

## B. 논문작성 · 연구지원 ★★★★★

> **Pipeline 1(문헌수집) · Pipeline 4(논문작성)**에 직결

### 문헌 검색·수집

| 스킬 | 한줄 설명 |
|-----|---------|
| **paper-lookup** | PubMed·arXiv·Semantic Scholar 등 **10개 학술DB** API 동시 검색 |
| **bgpt-paper-search** | 논문 전문에서 실험값 25개 이상 필드 추출 (방법·결과·샘플수·오차) |
| **literature-review** | 체계적 문헌검토 — 다중 DB 검색 + 인증 인용 포함 MD/PDF 출력 |
| **citation-management** | Google Scholar/PubMed 검색 → BibTeX 자동 생성·검증 |
| **paperzilla** | 프로젝트 추천 논문 + 정규 논문 요약 채팅 인터페이스 |

### 논문 작성·편집

| 스킬 | 한줄 설명 |
|-----|---------|
| **scientific-writing** | **IMRAD 구조 논문 작성** — 두 단계(아웃라인→완전 문장) 방법론 |
| **scientific-visualization** | 저널용 출판 품질 그래프 (Nature/Science/Cell 스타일, 다중패널) |
| **scientific-schematics** | AI 기반 기술 다이어그램 — 플로우차트, 신경망, 생물학적 경로 |
| **markdown-mermaid-writing** | 마크다운 + Mermaid 다이어그램 24종 — 기본 문서화 표준 |
| **venue-templates** | Nature·Science·NeurIPS·IEEE 등 저널별 LaTeX 템플릿·제출 가이드 |
| **latex-posters** | LaTeX beamerposter/tikzposter 학술 포스터 |
| **pptx-posters** | HTML/CSS → PDF/PPTX 연구 포스터 |

### 연구 프로세스

| 스킬 | 한줄 설명 |
|-----|---------|
| **scientific-brainstorming** | 창의적 연구 아이디어 발굴 — 학제간 연결, 가정 도전 |
| **hypothesis-generation** | 데이터·관측에서 검증 가능한 가설 정형화 → 실험 설계 제안 |
| **scientific-critical-thinking** | 실험 설계 타당성·편향·교란변수 평가, 근거 수준 등급 |
| **peer-review** | 체크리스트 기반 공식 심사 의견서 작성 |
| **scholar-evaluation** | 연구 품질 정량 점수화 (ScholarEval 프레임워크) |
| **research-grants** | NSF·NIH·DOE·DARPA·NSTC 연구제안서 작성 |

**MIDAS 적용 포인트**:
- `paper-lookup` + `bgpt-paper-search` → Pipeline 1 데이터셋 수집
- `scientific-writing` → Pipeline 4 논문 초안 업그레이드
- `scientific-visualization` → parity plot, 피처 중요도 그래프
- `hypothesis-generation` → 새 물성 연구 시작 시

---

## C. ML · 통계 · 최적화 ★★★★★

> **Pipeline 3(모델링)**에 직결 + Goal 1(다목적 최적화)

| 스킬 | 한줄 설명 |
|-----|---------|
| **scikit-learn** | GBR/RF/Ridge/GPR 등 클래식 ML 전반 — MIDAS ml_engine 기반 |
| **shap** | SHAP 기반 ML 모델 예측 근거 설명 — 피처 기여도 waterfall/beeswarm |
| **statistical-analysis** | 통계 검정 선택 가이드 + APA 형식 결과 보고 |
| **statsmodels** | OLS·GLM·혼합모델·ARIMA — 계수 테이블·잔차·추론 |
| **pymoo** | **NSGA-II/III Pareto 최적화** — 다목적 공학 설계 (Goal 1 핵심!) |
| **pymc** | 베이지안 모델링 + MCMC — 불확실성 정량화, 계층 모델 |
| **pytorch-lightning** | 딥러닝 대규모 학습 — 멀티GPU, 콜백, W&B 로깅 |
| **transformers** | HuggingFace 사전학습 모델 — NLP·비전·오디오·멀티모달 |
| **torch-geometric** | PyG 기반 그래프 신경망 — 분자 GNN, 노드/링크 예측 |
| **umap-learn** | UMAP 차원 축소 — 화학 공간 시각화, 클러스터링 전처리 |
| **aeon** | 시계열 ML — 분류·회귀·예측·이상탐지 (공정 데이터에 유용) |
| **timesfm-forecasting** | Google TimesFM 제로샷 시계열 예측 |
| **stable-baselines3** | 강화학습 (PPO·SAC·DQN) — 표준 알고리즘, 빠른 프로토타이핑 |
| **hypogenic** | LLM 기반 자동 가설 생성·검증 (표 형식 데이터) |

**MIDAS 적용 포인트**:
- `shap` → 어떤 구조적 특성이 저유전율을 만드는지 설명
- `pymoo` → OLED 5개 KPI 동시 최적화 (Goal 1)
- `umap-learn` → 263개 분자의 화학 공간 2D 시각화

---

## D. 데이터처리 · 인프라 ★★★☆☆

| 스킬 | 한줄 설명 |
|-----|---------|
| **polars** | 초고속 DataFrame (pandas 대체) — 1~100GB 데이터 |
| **dask** | RAM 초과 분산 병렬 pandas/NumPy |
| **vaex** | 수십억 행 대용량 CSV/HDF5 처리 |
| **zarr-python** | 청크 N차원 배열 — 클라우드 스토리지, 병렬 I/O |
| **lamindb** | 생물학 데이터 레이크하우스 — 계보·재현성 추적 |
| **exploratory-data-analysis** | 200+ 파일 형식 자동 EDA 리포트 |
| **get-available-resources** | 연산 집약 작업 전 CPU/GPU/메모리/디스크 자원 탐지 |
| **optimize-for-gpu** | CuPy·Numba·cuML로 NumPy/pandas/scikit-learn GPU 가속 |

---

## E. 생명과학 · 유전체 · 단백질 ★★☆☆☆

> 바이오인포매틱스 특화 — 재료공학 적용은 제한적

| 스킬 | 한줄 설명 |
|-----|---------|
| **biopython** | 시퀀스 분석·FASTA/GenBank/PDB 파싱·NCBI 접근 |
| **bioservices** | 40+ 바이오DB 통합 API (UniProt·KEGG·ChEMBL) |
| **gget** | 20+ 바이오DB 빠른 조회 (AlphaFold 구조 포함) |
| **pysam** | SAM/BAM/VCF 게놈 파일 처리 |
| **deeptools** | NGS 분석 (BAM→BigWig, ChIP-seq 히트맵) |
| **pydeseq2** | RNA-seq 차등 유전자 발현 분석 |
| **anndata** | 단일세포 분석용 주석 행렬 (h5ad 포맷) |
| **scanpy** | 단일세포 RNA-seq 표준 파이프라인 |
| **scvi-tools** | 딥 생성 모델 기반 단일세포 분석 |
| **scvelo** | RNA velocity — 세포 상태 전환 분석 |
| **arboreto** | 유전자 조절 네트워크 추론 (GRNBoost2) |
| **cellxgene-census** | 6100만 세포 단일세포 아틀라스 쿼리 |
| **polars-bio** | BED/VCF/BAM 게놈 구간 고성능 연산 |
| **geniml** | 게놈 구간 ML (Region2Vec, scATAC-seq) |
| **gtars** | Rust 기반 게놈 구간 분석 |
| **phylogenetics** | MAFFT·IQ-TREE2로 계통수 구축·분석 |
| **scikit-bio** | 시퀀스 분석·다양성 지수·UniFrac |
| **esm** | ESM3 단백질 언어 모델 — 서열·구조·기능 설계 |
| **adaptyv** | Adaptyv Bio Foundry API — 단백질 실험 설계·제출 |
| **tiledbvcf** | 대규모 VCF 게놈 변이 데이터 저장·쿼리 |
| **glycoengineering** | 단백질 당화 분석·공학 |
| **primekg** | Precision Medicine 지식그래프 (유전자·약물·질병) |

---

## F. 의학 · 임상 · 헬스케어 ★★☆☆☆

| 스킬 | 한줄 설명 |
|-----|---------|
| **clinical-decision-support** | 바이오마커 계층화 환자 코호트 분석·치료 권고 문서 |
| **clinical-reports** | 증례 보고·진단 보고서·임상시험 보고 (CARE/ICH-E3) |
| **pyhealth** | EHR/신호/영상 기반 임상 딥러닝 파이프라인 |
| **treatment-plans** | 의학적 치료 계획서 LaTeX/PDF 생성 |
| **pydicom** | DICOM 의료영상 파일 읽기·쓰기·익명화 |
| **imaging-data-commons** | NCI 암 공개 영상 데이터 쿼리·다운로드 |
| **neurokit2** | ECG·EEG·EDA·PPG 생체신호 분석 |
| **neuropixels-analysis** | Neuropixels 신경 기록 + Kilosort4 스파이크 정렬 |
| **histolab** | WSI 병리 슬라이드 타일 추출·전처리 |
| **pathml** | 다중 면역형광 포함 전산 병리 딥러닝 |
| **flowio** | FCS 유세포 분석 파일 파싱 |

---

## G. 양자컴퓨팅 · 물리 ★★★☆☆

> 디스플레이 광학·소자 물리 연구로 확장 가능

| 스킬 | 한줄 설명 |
|-----|---------|
| **qiskit** | IBM 양자컴퓨터 — 회로 설계·런타임·오류 완화 |
| **cirq** | Google 양자컴퓨터 — 노이즈 인식 회로 설계 |
| **pennylane** | 하드웨어 독립 양자ML — 자동미분, IBM/Google/IonQ 지원 |
| **qutip** | 개방 양자계 시뮬레이션 — Lindblad 동역학·양자광학 |
| **fluidsim** | 전산 유체역학 (Navier-Stokes·난류·소용돌이) |
| **astropy** | 천문학·천체물리학 — 좌표·단위·FITS·우주론 |
| **sympy** | 기호 수학 — 방정식·미적분·행렬 해석 풀이 |

**디스플레이 공학 적용 가능성**:
- `qutip` → OLED 여기 상태 에너지 전달 모델링
- `sympy` → 광학 필름 전달행렬 해석적 계산

---

## H. 지구과학 · 지리정보 ★☆☆☆☆

| 스킬 | 한줄 설명 |
|-----|---------|
| **geopandas** | 지리 벡터 데이터 (shapefile/GeoJSON) 공간 분석 |
| **geomaster** | 위성 영상·GIS·원격탐사 ML (Sentinel·Landsat) |
| **fluidsim** | (물리 섹션과 중복 — CFD 포함) |

---

## I. 문서 · 미디어 · 시각화 ★★★☆☆

| 스킬 | 한줄 설명 |
|-----|---------|
| **docx** | Word 문서 (.docx) 생성·편집·서식 |
| **pdf** | PDF 읽기·병합·분할·OCR·폼 작성 |
| **pptx** | PowerPoint 슬라이드 생성·편집 |
| **xlsx** | Excel 스프레드시트 생성·편집·정제 |
| **markitdown** | PDF·DOCX·PPTX·이미지 → 마크다운 변환 |
| **generate-image** | AI 이미지 생성 (사진·일러스트·개념 시각화) |
| **infographics** | 전문 인포그래픽 생성 (10가지 유형) |
| **matplotlib** | 저수준 완전 커스터마이징 플로팅 |
| **seaborn** | pandas 통합 통계 시각화 — 빠른 탐색용 |

---

## J. 실험자동화 · 클라우드 ★★☆☆☆

| 스킬 | 한줄 설명 |
|-----|---------|
| **pylabrobot** | 벤더 무관 실험 자동화 (Hamilton·Tecan·Opentrons) |
| **opentrons-integration** | Opentrons OT-2/Flex 공식 프로토콜 API |
| **modal** | 클라우드 GPU 서버리스 — ML 모델 배포·H100 가속 |
| **dnanexus-integration** | DNAnexus 클라우드 게놈 파이프라인 |
| **latchbio-integration** | Latch 바이오인포 서버리스 워크플로우 |
| **benchling-integration** | Benchling R&D 플랫폼 (레지스트리·인벤토리·ELN) |
| **labarchive-integration** | 전자실험노트 (ELN) API 연동 |
| **omero-integration** | OMERO 현미경 이미지 관리 플랫폼 |
| **protocolsio-integration** | protocols.io 실험 프로토콜 관리 |
| **ginkgo-cloud-lab** | Ginkgo Cloud Lab — 자율 실험 실행 |
| **adaptyv** | Adaptyv Bio Foundry — 단백질 실험 설계 |

---

## K. AI 에이전트 · 연구지원 특수 ★★★★☆

> 연구 전 과정을 AI로 지원하는 메타 스킬

| 스킬 | 한줄 설명 |
|-----|---------|
| **database-lookup** | **78개 공개 과학DB** 검색 (물리·화학·재료·생물·임상·경제) |
| **parallel-web** | 웹 검색·URL 추출·딥리서치 — 학술 소스 우선 |
| **hugging-science** | HuggingFace 과학 ML 데이터셋·모델 발굴·활용 |
| **open-notebook** | 셀프호스팅 AI 연구 노트북 (NotebookLM 대안) |
| **pyzotero** | Zotero 참고문헌 라이브러리 프로그래밍 접근 |
| **autoskill** | 사용자 반복 워크플로우 감지 → 새 스킬 자동 제안 |
| **what-if-oracle** | 가상 시나리오 분석 — 다분기 가능성 탐색 |
| **consciousness-council** | 멀티 관점 심의 — 어려운 결정·딜레마 분석 |
| **scientific-brainstorming** | 창의적 연구 아이디어 발산 (초기 연구 기획) |
| **dhdna-profiler** | 텍스트에서 인지 패턴·사고 방식 추출 분석 |

**특히 주목할 스킬**:
- `database-lookup`: 재료 DB(Materials Project·COD), 화학 DB(PubChem·ChEMBL·ZINC), 물리 DB(NIST·NASA) 포함 — 데이터셋 수집에 강력
- `hugging-science`: 재료과학 사전학습 모델 발굴 (MatBERT, ChemBERTa 등)

---

## L. 경제 · 금융 · 법규 ★☆☆☆☆

| 스킬 | 한줄 설명 |
|-----|---------|
| **usfiscaldata** | 미국 재무부 재정 데이터 API |
| **market-research-reports** | 컨설팅 스타일 시장조사 보고서 (50페이지+) |
| **iso-13485-certification** | ISO 13485 의료기기 QMS 인증 문서 작성 |

---

## MIDAS × Skills 최적 조합 요약

```
연구 단계          추천 스킬 조합
─────────────────────────────────────────────────────────────────
데이터 수집         database-lookup → paper-lookup → bgpt-paper-search
피처 엔지니어링     rdkit → datamol → exploratory-data-analysis
ML 모델링           scikit-learn → shap → pymoo (다목적)
논문 작성           scientific-writing → scientific-visualization → venue-templates
발표 자료           scientific-slides → pptx → infographics
연구 기획           scientific-brainstorming → hypothesis-generation
```

---

*총 136개 스킬 / Scientific-Agent-Skills Package*  
*작성: NEXTIO × Claude Code*
