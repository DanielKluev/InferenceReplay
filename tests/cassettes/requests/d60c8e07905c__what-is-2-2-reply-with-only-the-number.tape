---
tape_version: 2
content_hash: d60c8e07905c
prompt_model_hash: d60c8e07905c
prompt_hash: 8c81e05fef75a909
model: openai/gpt-oss-120b
endpoint: /v1/chat/completions
max_tokens: 200
stop_sequences: []
tools: []
logprobs: false
recorded: '2026-05-17T08:57:36.115573Z'
replies: 1
max_replies: 1
status_code: 200
boundary: cf544f
sampling:
  is_greedy: false
---
--cf544f user

What is 2+2? Reply with only the number.
--cf544f reply 1
Response: 6b58441440ae.json
Stream: 6b58441440ae.chunks.ndjson
Status: 200
Stop-Reason: stop
Input-Tokens: 67
Output-Tokens: 42

4
--cf544f reply 1 reasoning

The user asks: "What is 2+2? Reply with only the number." So answer: 4. Only number, no extra text.
--cf544f--
