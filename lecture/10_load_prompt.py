from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import load_prompt

_ = load_dotenv(find_dotenv())

prompt = load_prompt("./json/data.json")
#prompt = load_prompt("data.yaml")

# 1. Chat 모델 생성
chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    # 답변 생성하는 과정을 시각화 가능
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

print(prompt.format(country="Japan"))