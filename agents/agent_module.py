# -*- coding: utf-8 -*-
"""Agent Module.ipynb
"""
import pinecone
import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent

#Constants
OPENAI_API_KEY = "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"  # platform.openai.com
PINECONE_API_KEY = "3690f713-ac87-4332-be84-1d8bec84b146"  # app.pinecone.io
PINECONE_ENV = "asia-southeast1-gcp"

def get_agent_response(query_text: str):

    pinecone.init(
    api_key=PINECONE_API_KEY,  # app.pinecone.io
    environment=PINECONE_ENV  # next to API key in console
    )

    index_name = "chatgpt-retrieval-plugin"

    if index_name not in pinecone.list_indexes():
    raise ValueError(
    f"No '{index_name}' index exists. You must create the index before "
    "running this notebook. Please refer to the walkthrough at "
    "'github.com/pinecone-io/examples'."  # TODO add full link
    )

    index = pinecone.Index(index_name)



    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectordb = Pinecone(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key="text"
    )



    llm=ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name='gpt-3.5-turbo'
    )



    retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
    )

    tool_desc = """Use this tool to answer user questions about LangChain. If the user needs informations about LangChain use this tool to get
    the answer. This tool can also be used for follow up questions from
    the user. Also this toole gives you the infos about the latest publication from the Bafin about risks in the finance sector."""



    tools = [Tool(
    func=retriever.run,
    description=tool_desc,
    name='LangChain DB'
    )]



    memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # important to align with agent prompt (below)
    k=5,
    return_messages=True
    )



    conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
    memory=memory,
    )

    sys_msg = """You are a helpful chatbot that answers the user's questions.
    """

    prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
    )
    conversational_agent.agent.llm_chain.prompt = prompt


    response = conversational_agent(query_text)
    return response
