# ChatGPT Free / Claude Free supplementary evaluation

이 폴더의 각 JSONL batch를 새 대화에 붙여 넣고 다음 prompt를 사용한다.

```text
아래 각 한국어 댓글을 hate 또는 non-hate로 분류하세요.
부정적인 의견이더라도 개인/집단에 대한 공격·비하·적대가 없으면 non-hate입니다.
설명은 하지 말고 입력과 같은 순서의 JSONL만 출력하세요.
각 줄 형식: {"id":"...","variant":"입력의 variant","label":"hate 또는 non-hate"}
```

실행 시 다음 metadata를 기록한다.

- service: ChatGPT Free / ChatGPT Plus / Claude Free
- displayed model or mode
- run date
- fallback or rate-limit message
- batch filename

결과는 `scripts/import_ui_llm_predictions.py`로 통합한다.
