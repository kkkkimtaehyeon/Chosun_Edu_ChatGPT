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
    {"role": "system", "content": ""},
    {"role": "user", "content": "클라우드 설명해줘."}
  ]
)

print(completion.choices[0].message)