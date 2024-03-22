from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1
)

messages = [
    SystemMessage(content="너는 나의 탐험가야, 모든 답변을 영어로 해야해!"),
    SystemMessage(content="Hello, I'm sara"),
    SystemMessage(content="한국과 일본의 거리는 얼마인가요? 그리고 너의 이름은 무엇이니?"),
]

print(chat.predict_messages(messages))