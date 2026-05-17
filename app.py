# import os
# from dotenv import load_dotenv
# from langchain_nvidia_ai_endpoints import ChatNVIDIA

# load_dotenv()
# nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# if not nvidia_api_key:
#     raise ValueError("Error: NVIDIA_API_KEY Missing!")

# print("Connecting to official NVIDIA AI Enterprise pipeline...")

# try:
    
#     llm = ChatNVIDIA(
#         model="meta/llama-3.1-8b-instruct",
#         api_key=nvidia_api_key.strip(),
#         temperature=0.7,   
#         max_tokens=250     
#     )

#     response = llm.invoke([
#         {
#             "role": "user",
#             "content": "You are a brilliant AI mentor. Tell me a short, powerful motivational advice in Marathi language for a software engineer!"
#         }
#     ])

#     print("\n--- AI Mentor Marathi Advice ---")
#     print(response.content)

# except Exception as e:
#     print("\nError:", e)





## Add Bot memory 

# # 1. Load Environment Variables 
# from dotenv import load_dotenv
# load_dotenv()

# import os
# nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# if not nvidia_api_key:
#     raise ValueError("Error: NVIDIA_API_KEY Missing!")

# # 2.Set Llama 3.1 8B model via Nvidia NIM
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# llm = ChatNVIDIA(
#     model="meta/llama-3.1-8b-instruct",
#     api_key=nvidia_api_key.strip(),
#     temperature=0.5,
#     max_completion_tokens=500
# )

# # 3.Create System Prompt (English Only - Clinical Grade)
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# template = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are an expert Medical AI Assistant. Provide accurate, helpful, and empathetic "
#         "medical information based on the symptoms provided by the user.\n\n"
#         "RULES:\n"
#         "1. Strictly reply in English only.\n"
#         "2. Keep the advice highly structured using bold headings and bullet points for readability.\n"
#         "3. Provide practical, safe home remedies and general guidance, but never prescribe specific prescription-only medications.\n"
#         "4. Always conclude with a strong medical disclaimer emphasizing that this information is for educational purposes and the user must consult a certified doctor for definitive diagnosis."
#     ),
#     MessagesPlaceholder(variable_name = "chat_history"),
#     ("human", "{user_question}")
# ])

# # 4. LangChain Expression Language (LCEL) चेन
# chain = template | llm

# store = {}

# from langchain_core.chat_history import InMemoryChatMessageHistory
# def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]


# from langchain_core.runnables.history import RunnableWithMessageHistory
# chain_with_memory = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key = "user_question",
#     history_messages_key = "chat_history",
# )

# print("🩺 Medical AI Chatbot Kernel Initialized...")
# print("--------------------------------------------------")
# print("Type your medical symptoms or questions below.")
# print("Type 'exit' or 'quit' to close the chatbot.\n")


# config = {"configurable":{"session_id":"medical_chat_session_1"}}

# # 5. Infinite Loop to continuously receive input from the user
# while True:
#     try:
#         # Taking input from the user into the terminal
#         user_query = input("\n You: ")
        
#         # Stop the loop if the user types exit 
#         if user_query.strip().lower() == 'exit':
#             print("🩺 Medical Assistant closing. Take care!")
#             break
            
#         # If the user presses Enter without typing anything, ask again.
#         if not user_query.strip():
#             print("Please enter a valid question.")
#             continue
            
#         print("🤖 Thinking...")
        
#         # Sending a question to the model
#         response = chain_with_memory.invoke({"user_question": user_query},config=config)
        
#         # Print Output
#         print("\n🤖 Assistant Advice:")
#         print(response.content)
#         print("--------------------------------------------------")
        
#     except KeyboardInterrupt:
#         # To exit safely when pressing Ctrl+C
#         print("\n🩺 Medical Assistant closing. Take care!")
#         break
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")



# Add Bot memory + vectore Databse (pinecone)

# 1. Load environment variable
from dotenv import load_dotenv
load_dotenv()

# 2. import API Key
import os
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 3. reise Error if api key is missing or wrong
if not nvidia_api_key or not pinecone_api_key:
    raise ValueError("Error: NVIDIA_API_KEY or PINECONE_API_KEY is missing or wrong")

# 4. To set Large language model
from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(
    model = "meta/llama-3.1-8b-instruct",
    api_key=nvidia_api_key.strip(),
    temperature = 0.2,
    max_completion_tokens= 500
    )

# 5. Embedding model
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# embeddings = NVIDIAEmbeddings(
#    model="snowflake/arctic-embed-l",
#     api_key=nvidia_api_key.strip()
# )
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 6. Setup for pinecone to cloud Database
from pinecone import Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "medical-chat-bot-hf-index"

# 7. If an index with this name does not exist on the cloud, create it (Dimension 1024 - for Nvidia Model)
from pinecone import ServerlessSpec
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}.....")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud ="aws",region="us-east-1")
    )

# 8. Knowledge Base - Uploading data to Pinecone
medical_knowledge = [
    "For high fever in children oe elderly patients, always use a lukewarm water sponge bath to lower the temperature safely. Never use ice-cold or very cold water,as it causes severe shivering and can increase the core body temperature dangerously.",
    "A throbbing headache accompanied by sensitivity to light and sound is a classic sign of a Migraine. Relief can be archieved by resting in a dark, quite room and staying hydrated.",
    "A productive cough with yellowish mucus and high fever in older adults suggests a potential huge infection or pneumonia. Immediate medical consultation is reuired."
]

print("Uploading medical knowledge base to pinecone cloud....")
from langchain_pinecone import PineconeVectorStore
vectore_store = PineconeVectorStore.from_texts(
    texts = medical_knowledge,
    embedding = embeddings,
    index_name = index_name
)
print("Knowledge base successfully synchronized with pinecone cloud...")


# 9. Create RAG Retriever (This will retrieve the top 2 matching references from Pinecone)
retriver = vectore_store.as_retriever(search_kwargs={"k":2})

# 10. Advsnce system prompt (RAG Content + Memory)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
template = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are an expert Medical AI Assistant. Answer the user's question using ONLY the strictly varified medical context provided below."
        "If the answer cannot be derived from the context, use your general clinical knowledge but state it clearly.\n\n"
        "OFFICIAL MEDICAL CONTEXT:\n{rag_context}\n\n"
        "RULES:\n"
        "1. Strictly reply in English only.\n"
        "2. Keep the advice highly structured using bold headings and bullet points.\n"
        "3. Always conclude with a strong medical disclaimer."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{user_question}")
    ]
)


# 11. LangChain Expression Language (LCEL) chain
chain = template | llm


# 12. Chat Memory Management
store ={}

from langchain_core.chat_history import InMemoryChatMessageHistory
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


from langchain_core.runnables.history import RunnableWithMessageHistory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_question",
    history_messages_key = "chat_history",
)


# 13 Chat interface and runtime logic
print("\n Production_Grade Medical Chatbot (Llama 3.1 + Pinecone Cloud + Memory) Ready......")
print("----------------------------------------------------------------------------------------")
config = {"configurable":{"session_id":"production_session_1"}}


while True:
    try:
        user_input = input("You: ")

        if user_input.strip().lower()=="exit":
            print("Medical Assistance Closing. Take care!.....")
            break
        
        if not user_input.strip():
            continue

        print("Searching Pinecone Cloud & Thinking.........")


        # A. Finding relevant documents from Pinecone
        relevent_docs = retriver.invoke(user_input)
        # Combining the received reference
        context_str = "\n".join([doc.page_content for doc in relevent_docs])

        # B. Running the memory chain (in this we are giving both 'user_question' and 'rag_context' to the prompt)
        response = chain_with_memory.invoke(
            {
                "user_question":user_input,
                "rag_context": context_str
            },
            config=config
        )

        print("\n  Assistant Advice ( Varified by pinecone): ")
        print(response.content)
        print("-----------------------------------------------------")
    
    except KeyboardInterrupt:
        print("\n  Closing down....")
        break

    except Exception as e:
        print(f"\nAn Error occurred: {e}")
