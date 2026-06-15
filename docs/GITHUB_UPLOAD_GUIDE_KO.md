# GitHub 업로드 가이드

## 1. 저장소 본체

`korean_obfuscation_robustness_github_repo_20260615.zip`을 압축 해제한 뒤,
압축 안의 폴더 내용을 GitHub repository에 올립니다.

권장 repository 이름:

```text
korean-obfuscation-hate-speech-robustness
```

## 2. 대용량 모델 파일

다음 두 파일은 별도 release asset으로 준비됩니다.

```text
char_tfidf_mlp_clean_only_model.zip
normalization_char_tfidf_mlp_model.zip
```

GitHub repository의 **Releases**에서 새 release를 만들고 두 ZIP을
첨부하면 됩니다. README의 `docs/MODELS.md`에 압축 해제 위치가 적혀
있습니다.

Qwen base weight는 포함하지 않고 Hugging Face 링크만 제공합니다.

## 3. 공개 전 확인

1. `README.md`의 프로젝트 설명과 결과를 확인합니다.
2. `KMHAS_BENCHMARK_SHARING_NOTE.md`를 삭제하지 않습니다.
3. K-MHaS 원본 repository의 최신 배포 조건을 다시 확인합니다.
4. 데이터에 실제 혐오 및 공격적 표현이 포함된다는 경고를 유지합니다.
5. 무료 UI 비교는 정확한 backend model이 확인되지 않은 보조 실험임을
   유지합니다.

## 4. 업로드 후 권장 Release 설명

```text
This release contains the two scikit-learn model artifacts used in the
K-MHaS Korean obfuscation robustness benchmark. See docs/MODELS.md for
expected paths, loading instructions, and model recipes.
```

