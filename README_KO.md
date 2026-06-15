# 한국어 난독화 악성 댓글 탐지 Robustness 실험

이 프로젝트는 동일한 한국어 댓글을 로마자와 rune/glyph 문자로 점점 강하게
난독화했을 때 악성 댓글 탐지 모델의 성능이 어떻게 변하는지 분석합니다.

핵심은 단순 정확도가 아니라 다음 두 오류를 동시에 줄이는 것입니다.

- **Under-detection, FNR**: 실제 악성 댓글을 정상으로 놓치는 비율
- **Over-blocking, FPR**: 실제 정상 댓글을 악성으로 차단하는 비율

## 최종 결론

500개의 동일한 K-MHaS test 댓글을 5개 variant로 만든 2,500행 공통
benchmark에서 `normalization + char_tfidf_mlp`가 가장 높은 worst-case
balanced accuracy `0.772`를 기록했습니다.

Raw Qwen은 강한 `roman_glyph_70`에서 정상 댓글까지 악성으로 판단하는
FPR이 `0.852`였습니다. Qwen에 normalization, validation threshold
calibration, LoRA 학습을 적용한 뒤 FPR은 `0.388`로 감소했고 balanced
accuracy는 `0.522`에서 `0.672`로 향상됐습니다.

즉, 큰 언어 모델이라고 해서 난독화 입력을 자동으로 균형 있게 판단하는 것은
아니며, 입력 복원과 task-specific adaptation이 필요하다는 결과입니다.

## 제출 요구사항 대응

- **All code materials**: `scripts/`, `configs/`
- **Used models**: `models/` 및 별도 model release asset
- **Used datasets**: `outputs/kmhas_free_llm_benchmark/data/`
- **Training recipes and results**:
  `docs/TRAINING_RECIPES_AND_RESULTS.md`
- **전체 결과표와 그래프**:
  `outputs/kmhas_free_llm_benchmark/reports/`,
  `outputs/kmhas_free_llm_benchmark/figures/`

## 먼저 확인할 파일

1. `README.md`: GitHub 메인 설명
2. `docs/DATASET.md`: 데이터 출처와 생성 과정
3. `docs/MODELS.md`: 사용 모델과 모델 파일
4. `docs/TRAINING_RECIPES_AND_RESULTS.md`: 학습 설정과 결과
5. `docs/GITHUB_UPLOAD_GUIDE_KO.md`: GitHub 업로드 순서

주의: 데이터에는 실제 혐오 및 공격적 표현이 포함됩니다.

