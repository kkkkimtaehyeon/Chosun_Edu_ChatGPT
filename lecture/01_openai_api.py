#라이브러리 관리
# 1. venv
# 2. anaconda

# openai api 제공
# https://openai.com/blog/openai-api

from openai import OpenAI
client = OpenAI(api_key="")
# 인스턴스 = 생성자 함수

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "모든 설명을 3줄 요약해줘"},
    {"role": "user", "content": "클라우드 설명해줘."}
  ]
)

print(completion.choices[0].message)

# OpenAI의 API를 사용하는 챗봇 문제점
# 1. 개발이 어려움 -> 더 쉽게 개발할 수 있는 방법(프레임워크) 필요
# 2. 챗봇 개발 완성 -> Bard 모델 변경 -> Bard API로 처음부터 개발 -> 프레임워크(LLM)

# -> LangChain 프레임워크(모델 바꿔도 코드 동일)