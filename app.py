import chainlit as cl
from dozer import getCredit,getCustomerData
import os
from langchain.prompts import  PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
## For Vector DB
import pinecone
import uuid
from langchain.vectorstores import Pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
#from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from chainlit import user_session
import os


#user_env = user_session.get(".env")
OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINE_ENV")
 # platform.openai.com
model_name = 'text-embedding-ada-002'


embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPEN_API_KEY
)

# find ENV (cloud region) next to API key in console
YOUR_ENV = "us-west1-gcp-free"
index_name = 'langchain-retrieval-agent'
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)


#embeddings = OpenAIEmbeddings()


text_field = "Answer"
index = pinecone.Index('langchain-retrieval-agent')
# switch back to normal index for langchain
#index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(
    openai_api_key = OPEN_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def setName(userName):
    global name
    name = userName

def customerProfile(input = ""):
    data = getCustomerData(input=name)
    name1 = data[0]
    income = data[1]
    age= data[2]
    dependents= data[3]
    credit_amt= data[4]
    repay_status= data[5]
    util_ratio= data[6]
    address= data[7]

    return ("Age = {age}, income = {income}, dependents = {dependents},repay_status={repay_status}, credit utilisation ratio ={util_ratio} ,address={address}",age,income,dependents,repay_status,util_ratio,address)

    
tools = [
    Tool(
        name='Knowledge Base',
        func=chain.run,
        description=(
            "Useful when you need general information about bank policies and bank offerings. "
        )
    )
]


agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


global res
@cl.on_chat_start

async def start():
    intro = "Hi there, I am an assistant for Bank A. I am here to assist you with all your banking needs! Please enter your name:  "
    
    res = await cl.AskUserMessage(content=intro,timeout=45,raise_on_timeout=True).send()
    

    greeting = f"Hi {res['content']}. What brings you here?"
    await cl.Message(content=greeting).send()
    setName(res['content'])
    # global credit 
    # credit = getCredit(int(res['content']))
    cl.user_session.set("chain", chain)


'''@cl.langchain_factory(use_async = False)
def factory():
    cl.user_session.set("chain", qa)
    
    str1 = f" The total credit for the user is {credit}. "
    

    template = "You are an intelligent chatbot. You are a banking assistant. You will help the user with finance related questions and along with schemes credit card schemes" + str1 + "{question}"
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    return llm_chain'''


@cl.on_message
async def main(message: str):
    response = await cl.make_async(agent.run)(message.content)

    await cl.Message(
        content=response,
    ).send()