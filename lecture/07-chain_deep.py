from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import StreamingStdOutCallbackHandler

_ = load_dotenv(find_dotenv())

# 1. Chat 모델 생성
chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    # 답변 생성하는 과정을 시각화 가능
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

chef_prompt = ChatPromptTemplate.from_messages([
    ("system","당신은 전세계에서 유명한 요리사입니다. 찾기 쉬운 재료를 사용해서 모든 종류의 요리에 대해 쉽게 따라할 수 있는 레시피를 만드세요"),
    ("human","나는{cook}요리를 만들고 싶어요")    
])

#chain 1 생성(=> 음식 종류)
chef_chain = chef_prompt | chat

veg_chef_prompt = ChatPromptTemplate.from_messages([
    ("system","당신은 전통적인 요리법을 채식으로 만드는 채식주의 요리사입니다. 대체 재료를 찾고 그 준비과정을설명하세요. 근본적으로 레시피를 수정하지 말고, 대체 재료가 없는 경우 없다고 하세요."),
    ("human","{recipe}")   

])

#chain2 생성(=> recipe)
veg_chain = veg_chef_prompt | chat


#chain3 생성(연결)
final_chain = chef_chain | veg_chain

#chef_chain을 json으로 veg_chain에 input
final_chain = {"recipe" : chef_chain} | veg_chain

#chain 실행
final_chain.invoke({
    "cook" : "india"
})



# 2. Parser 생성
from langchain.schema import BaseOutputParser
class CommaOutputParser(BaseOutputParser):
    
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))

# 3. Template 생성
template = ChatPromptTemplate.from_messages([
    ("system", "너는 리스트 생성 기계다. 모든 답변을 콤마로 구분해서 대답하라."),
    ("human", "{question}")
])


# 4. Chain 생성
#   - 모든 요소를 합쳐주는 역할
#   - 합쳐진 요소들은 하나의 chain으로 실행(하나하나 순서대로 result를 반환할 때 까지)
#   - 2개 이상의 Chain을 연결 가능
chain = template | chat | CommaOutputParser()

# 5. Chain 실행 (입력 매개변수: dict type 전달)
# result = chain.invoke({
#     "max_items": 5,
#     "question": "포켓몬은 무엇인가?"
# })

#print(result)

