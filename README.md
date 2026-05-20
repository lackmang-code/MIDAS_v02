# MIDAS v02 — Multi-Property QSPR Platform
### Materials Informatics for Design and Automated Screening (Extended)

> **다물성(Multi-property) QSPR 기반 후보 분자 탐색 + 논문 초안 자동 생성 플랫폼**
> v01의 단일 물성(유전율) 제약을 풀어, 임의 물성에 대해 동일한 워크플로우를 적용한다.

---

## v01 ↔ v02 차이

| 항목 | v01 | v02 |
|------|-----|-----|
| 타겟 물성 | 유전율(k) 단일 | 임의 물성 (properties/ 폴더 등록) |
| 데이터셋 | `crc_dielectric_combined.csv` 하드코딩 | `properties/<name>/dataset.csv` 자동 로드 |
| 임계값/방향 | `k < 2.4` 박힘 | `direction` + `screening_threshold` 메타로 분리 |
| log 변환 | 항상 적용 | `log_transform: false` 도 가능 (예: logS) |
| 논문 템플릿 | 단일 dielectric 본문 | 라우터 → dielectric/generic 별도 모듈 |
| Streamlit | 5탭 (S1-S5) | 동일 + 사이드바 물성 선택 |

---

## 새 물성 추가법
```
properties/<my_property>/
├── config.yaml          # 메타데이터 (target, threshold, direction 등)
├── dataset.csv          # SMILES + target 컬럼 포함
├── candidates.csv       # (선택) 내장 후보 분자 — name, smiles
└── paper_template.py    # (선택) 도메인 전용 논문 템플릿
```
앱을 재시작하면 사이드바에 자동 등록됨.

---

## 디렉토리 구조
```
MIDAS_v02/
├── app.py                          # Streamlit 진입점
├── run.bat                         # Windows 실행 스크립트 (port 8503)
├── requirements.txt
├── properties/                     # 물성 레지스트리
│   ├── base.py                     # PropertyConfig 데이터클래스
│   ├── registry.py                 # load_property / list_properties
│   ├── dielectric/                 # v01에서 이식
│   │   ├── config.yaml
│   │   └── dataset.csv             # 263 molecules
│   └── solubility/                 # 다물성 검증용 샘플
│       ├── config.yaml
│       └── dataset.csv             # ESOL/Delaney 50 mols
├── engines/                        # 물성-무관 엔진
│   ├── feature_engine.py           # RDKit 233 descriptors
│   ├── ml_engine.py                # GBR/RF/Ridge/GPR + CV + AD
│   ├── screening_engine.py         # 앙상블 예측 + leverage AD
│   ├── paper_engine.py             # 라우터
│   ├── _paper_dielectric.py        # v01 논문 템플릿
│   └── _paper_generic.py           # 범용 논문 템플릿
└── output/                         # 자동 저장 결과물
```

---

## 실행
```bash
pip install -r requirements.txt
run.bat          # 또는: python -m streamlit run app.py --server.port 8503
```
브라우저: http://localhost:8503

---

## 워크플로우 (v01과 동일)
1. **사이드바** — 타겟 물성 선택 (예: 유전율 / 수용해도)
2. **S1 Dataset** — EDA, 우수 후보 테이블
3. **S2 Features** — RDKit 기술자 계산 + Top-N 선택 (~5초)
4. **S3 ML Model** — 4종 학습 + CV + AD (~20-40초)
5. **S4 Screening** — 후보 분자 예측 + AD 판정
6. **S5 Paper** — 도메인-라우팅 논문 초안 생성 (.md/.docx)

---

## v01과의 호환성
- v01 디렉토리는 **수정하지 않음** — `read-only`
- v02의 `dielectric` 물성은 v01 데이터셋·논문 템플릿을 그대로 사용
- `random_state=42` 고정 — 동일 입력에 대해 동일 결과 보장

---

## License
MIT
