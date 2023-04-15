import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent

# Constants
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "chatgpt-retrieval-plugin"
index = pinecone.Index(index_name)

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Pinecone(index=index, embedding_function=embeddings.embed_query, text_key="text")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name='gpt-3.5-turbo')
retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())
tool_desc = "Use this tool to answer user questions about LangChain..."
tools = [Tool(func=retriever.run, description=tool_desc, name='LangChain DB')]
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

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

sys_msg = "You are a helpful chatbot that answers the user's questions."
prompt = conversational_agent.agent.create_prompt(system_message=sys_msg, tools=tools)
conversational_agent.agent.llm_chain.prompt = prompt

def get_agent_response(query_text: str):
    response = conversational_agent(query_text)
    return response

def deinitialize_pinecone():
    pinecone.deinit()
