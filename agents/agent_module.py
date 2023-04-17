import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent

# Constants
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "chatgpt-retrieval-plugin"
index = pinecone.Index(index_name)

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Pinecone(index=index, embedding_function=embeddings.embed_query, text_key="text")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name='gpt-3.5-turbo')
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())
search = GoogleSearchAPIWrapper()

# Initialize Tools
db_desc = "Use this tool to answer user questions about Volksbank internals (e.g. internal guidelines, sales data, work instructions, bankinternal tools) and related topics lika BaFin publications."
search_desc = "Use this tool to answer questions about current events or the current state of the world. the input to this should be a single search term."
tools = [Tool(func=retriever.run, description=db_desc, name='Bank internal Databank'), 
         Tool(func=search.run, description=search_desc, name='Search Internet')]

# Initialize conversational_agent
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
    memory=memory,
)

sys_msg = "You are a helpful assistant that answers the user's questions. Use the 'Bank internal Databank' first to retrieve information about unknown terms. It's very important that everytime you generate a 'response' it is in german language!"
prompt = conversational_agent.agent.create_prompt(system_message=sys_msg, tools=tools)
conversational_agent.agent.llm_chain.prompt = prompt

def get_agent_response(query_text: str):
    response = conversational_agent(query_text)
    return response

def deinitialize_pinecone():
    pinecone.deinit()

