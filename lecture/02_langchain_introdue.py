from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    # temperature: 0.1~1.0      : 0에 가까울수록 사실 기반, 
    #                            1에 가까울수록 창의력 기반
    temperature=0.1
)

template = PromptTemplate.from_template(
    "{country_a}과 {country_b}의 거리는 얼마나 먼가요?"
)

prompt = template.format(country_a="한국", country_b="일본")

print(chat.predict(prompt))