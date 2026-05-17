---
tape_version: 2
content_hash: 31e7a952d38d
prompt_model_hash: 31e7a952d38d
prompt_hash: 767a18b7152d15ef
model: openai/gpt-oss-120b
endpoint: /v1/chat/completions
max_tokens: 200
stop_sequences: []
tools: []
logprobs: false
recorded: '2026-05-17T08:57:34.762304Z'
replies: 1
max_replies: 1
status_code: 200
boundary: 1dbd4d
sampling:
  is_greedy: false
---
--1dbd4d user

This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.
--1dbd4d reply 1
Response: 12aa5cf567a2.json
Stream: 12aa5cf567a2.chunks.ndjson
Status: 200
Stop-Reason: stop
Input-Tokens: 94
Output-Tokens: 18

OK.
--1dbd4d reply 1 reasoning

The user wants only "OK." as reply, nothing else. Must obey.
--1dbd4d--
