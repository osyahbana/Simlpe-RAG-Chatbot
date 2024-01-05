import os 
import openai
import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

store = LocalFileStore("./cache/")

load_dotenv()

# PINECONE
index_name = 'berkshire-letters'
core_embeddings_model = OpenAIEmbeddings()
embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model, store, namespace=core_embeddings_model.model
)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

text_field = "text"

index = pinecone.Index(index_name)


vectorstore = Pinecone(
    index,
    embedder.embed_query,
    text_field
)



# chainlit app development STARTS here



@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo-1106",
        "temperature": 0,
        "max_tokens": 1000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):

    settings = cl.user_session.get("settings")

    llm = ChatOpenAI(
        model = settings["model"],
        temperature=settings["temperature"]
    )
   
    
    retriever = vectorstore.as_retriever()
    

    # Craft a prompt with context and question
    template = """
        You are Morgan Freeman. You answer questions based on the provided context in a way of speech that Morgan Freeman would answer. If you don't know how to answer based on the context provided below, just say you don't know the answer.
        Provided Context
        {context}
        User Question
        {question}
        At the end of your answer, add a quote from Morgan Freeman
        """ 
    prompt = ChatPromptTemplate.from_template(template)
    
    retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | llm,
         "context": itemgetter("context"),
      }
    )
    
    res = retrieval_augmented_qa_chain.invoke({"question" : message.content})
    direct_response = res['response'].content
    all_context = res['context']

    # Generate the refs list
    refs = []
    if all_context:
        for context_doc_object in all_context:
            ref_link = f"Link - {context_doc_object.metadata['source_document']}, at page {context_doc_object.metadata['page_number']}"
            refs.append(ref_link)

    text_elements = []
    if refs:
        for source_idx, link in enumerate(refs):
            source_name = f"source {source_idx}"
            text_elements.append(
                cl.Text(content=link, name=source_name)
            )
      
    await cl.Message(
        content=direct_response, elements=text_elements        
    ).send()