import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Advanced Hybrid Medical AI", page_icon="🩺", layout="centered")

# st.title("🩺 Advanced Hybrid Medical AI")
# st.caption("Llama 3.1 8B (Nvidia NIM) + GPT-4o-mini (OpenAI) + Pinecone Cloud DB")

try:
    banner_image = Image.open(r"medicalbanner.png")
    # It is best to use columns to fit images in layout=centered
    col1, col2 = st.columns([1, 10]) # 1 part blank, 10 parts image
    with col1:
        st.write("") # empty space
    with col2:
        st.image(banner_image, use_container_width=True, caption=None) # The caption was removed because it is in the image.
except Exception as banner_err:
    st.error(f"Error loading banner image: {banner_err}")
    st.title("🩺 Advanced Hybrid Medical AI") # Fallback text if the image does not load
st.write("---")

# 2. Load Environment Variables
load_dotenv()

# १. आधी Streamlit Secrets (Cloud) चेक करा, जर तिथे नसतील तर लोकल .env मधून घ्या
nvidia_api_key = st.secrets.get("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not all([nvidia_api_key, pinecone_api_key, openai_api_key]):
    st.error("Error: One or more API Keys (NVIDIA, PINECONE, OPENAI) are missing! Please check Streamlit Secrets or .env file.")
    st.stop()

if not all([nvidia_api_key, pinecone_api_key, openai_api_key]):
    st.error("Error: One or more API Keys (NVIDIA, PINECONE, OPENAI) are missing in .env!")
    st.stop()

# 3. Streamlit Session State (Memory Management)
if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# 4. Sidebar Configuration (Language Toggle & File Upload)
st.sidebar.header("🛠️ Control Panel")

# A. Language Selection
selected_lang = st.sidebar.selectbox("Choose Language / भाषा निवडा", ["English", "Marathi"])

# B. Prescription Upload Box
uploaded_file = st.sidebar.file_uploader("Upload Doctor's Prescription (Optional)", type=["png", "jpg", "jpeg"])

prescription_text = ""
if uploaded_file:
    st.sidebar.success("Prescription uploaded successfully! ✅")

# 5. Initialize Models (Cached for Speed - Now it has added auto-index creation)
# ====================================================================
# 🎯 5. Initialize Models (Cached for Speed - Core System Patch)
# ====================================================================
@st.cache_resource
def init_models():
    # 🌟 ब्रह्मास्त्र जुगाड: 'langchain_pinecone' लोड होण्यापूर्वीच जुन्या प्लगइनला सिस्टीममधून पूर्णपणे हाकलून देणे
    import sys
    sys.modules["pinecone-plugin-inference"] = None  # सक्तीने डिसेबल केले
    
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    
    # Llama 3.1 Model (Nvidia)
    llama_llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=nvidia_api_key.strip(),
        temperature=0.2,
        max_completion_tokens=500
    )
    
    # GPT-4o-mini Model (OpenAI)
    openai_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key.strip(),
        temperature=0.3
    )
    
    # Embeddings Setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    index_name = "medical-chatbot-hf-index"
    
    try:
        vector_store = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            pinecone_api_key=pinecone_api_key
        )
    except Exception as e:
        st.error(f"Pinecone Connection Error: {e}")
        st.stop()
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    return llama_llm, openai_llm, retriever

# To initialize the model outside of the function:
llama_llm, openai_llm, retriver = init_models()

# 6. Prescription Vision Processing (GPT-4o-mini Vision - Pillow Full-Proof Fix)
if uploaded_file:
    with st.sidebar.spinner("Analyzing Doctor's Handwriting... 🔍"):
        import base64
        import io
        from PIL import Image
        from openai import OpenAI
        
        try:
            # 1. Reading the uploaded file into memory via PIL Image
            raw_image = Image.open(uploaded_file)
            
            # 2. Forcibly converting the image to RGB mode (to remove transparent/PNG background)
            rgb_image = raw_image.convert("RGB")
            
            # ३. Saving this image to a pure JPEG binary buffer
            buffer = io.BytesIO()
            rgb_image.save(buffer, format="JPEG")
            jpeg_bytes = buffer.getvalue()
            
            # 4. Now encode this pure JPEG data into Base64
            base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            # 5. OpenAI API call
            client = OpenAI(api_key=openai_api_key)
            vision_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ("You are an expert medical pharmacist specialized in reading extremely poor doctor handwriting.\n"
                            "Analyze the attached medical prescription image carefully.\n\n"
                            "CRITICAL MEDICAL CONTEXT FOR YOU:\n"
                            "- Doctors often use short abbreviations like 'G.' or 'Gtt.' which stands for 'Guttae' (Eye Drops).\n"
                            "- Look closely at the shapes of the letters and cross-reference them with standard, real-world FDA-approved brand names or generics (e.g., Azopt, Combigan, Xalatan, etc.). Do not invent fake drug names like 'xolten'.\n\n"
                            "OUTPUT FORMAT:\n"
                            "Provide a clean, highly structured list of:\n"
                            "1. Extracted/Corrected Drug Name (with potential real-world match)\n"
                            "2. Form (e.g., Eye Drops, Tablet)\n"
                            "3. Dosage / Frequency instructions if visible.")},
                            # Now we can write 'image/jpeg' with our eyes closed, because the above code has converted the file to a JPEG!
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
                max_tokens=300
            )
            prescription_text = vision_response.choices[0].message.content
            st.sidebar.markdown("### 📝 Extracted Info:")
            st.sidebar.info(prescription_text)
            
        except Exception as vision_err:
            st.sidebar.error(f"Vision Error: {vision_err}")

# 7. LangChain Expression Language (LCEL) Chains Setup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# prompt 1: English Prompt for Llama 3.1
english_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Medical AI Assistant. Answer the user's question using ONLY the strictly verified medical context and prescription data provided.\n\nOFFICIAL CONTEXT:\n{rag_context}\n\nPRESCRIPTION INFO:\n{prescription_info}\n\nRULES:\n1. Reply in English only.\n2. Keep advice structured in bullet points.\n3. Always include a medical disclaimer."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_question}")
])

# prompt 2: Marathi Prompt for GPT-4o-mini
marathi_template = ChatPromptTemplate.from_messages([
    ("system", "तुम्ही एक तज्ज्ञ मेडिकल एआय असिस्टंट आहात. दिलेल्या पाइनकोन संदर्भ आणि प्रिस्क्रिप्शनच्या आधारे युझरच्या प्रश्नाचे अत्यंत शुद्ध, नैसर्गिक आणि व्याकरणदृष्ट्या परिपूर्ण मराठीत उत्तर द्या.\n\nOFFICIAL CONTEXT:\n{rag_context}\n\nPRESCRIPTION INFO:\n{prescription_info}\n\nनियम:\n1. फक्त आणि फक्त मराठीतच उत्तर द्या (English वैद्याकीय शब्द कंसात लिहू शकता).\n2. उत्तर स्वच्छ बुलेट पॉईंट्समध्ये द्या.\n3. शेवटी डॉक्टरांचा सल्ला घेण्याचा कडक मराठी डिस्क्लेमर नक्की जोडा."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_question}")
])

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_store[session_id]

# Creating memory chains for both models
llama_chain = RunnableWithMessageHistory(english_template | llama_llm, get_session_history, input_messages_key="user_question", history_messages_key="chat_history")
openai_chain = RunnableWithMessageHistory(marathi_template | openai_llm, get_session_history, input_messages_key="user_question", history_messages_key="chat_history")

# 8. Render Chat History on UI
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 9. Runtime User Input & Routing Logic
if user_input := st.chat_input("Type your question here..."):
    
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.ui_messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Processing request..."):
        try:
            # Searching for documents from Pinecone
            relevent_docs = retriver.invoke(user_input)
            context_str = "\n".join([doc.page_content for doc in relevent_docs])
            
            p_info = prescription_text if prescription_text else "No prescription uploaded."
            
            # 🎯 HYBRID ROUTING LOGIC: Selecting a model by language
            if selected_lang == "Marathi":
                config = {"configurable": {"session_id": "marathi_session_pro"}}
                response = openai_chain.invoke(
                    {"user_question": user_input, "rag_context": context_str, "prescription_info": p_info},
                    config=config
                )
            else:
                config = {"configurable": {"session_id": "english_session_pro"}}
                response = llama_chain.invoke(
                    {"user_question": user_input, "rag_context": context_str, "prescription_info": p_info},
                    config=config
                )
            
            with st.chat_message("assistant"):
                st.write(response.content)
            st.session_state.ui_messages.append({"role": "assistant", "content": response.content})
            
        except Exception as e:
            st.error(f"An Error occurred: {e}")
