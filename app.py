import streamlit as st
import os
from PyPDF2 import PdfReader
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import time

# ----------------------------
# 1. पेज सेटिंग्ज
# ----------------------------
st.set_page_config(
    page_title="मराठी AI एजंट बिल्डर",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Yatra+One&display=swap');
html, body, [class*="css"] {
    font-family: 'Yatra One', cursive;
    direction: ltr;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 मराठी AI एजंट")
st.write("मराठी PDF अपलोड करा आणि प्रश्न विचारा!")

# ----------------------------
# 2. Bhashini API Key (तू खाली तुझी KEY टाकशील)
# ----------------------------
BHASHINI_API_KEY = "तुझी_BHASHINI_API_KEY_इथे_टाका"  # ← ही ओळ बदल!
USER_ID = "default_user"

# ----------------------------
# 3. PDF अपलोड
# ----------------------------
uploaded_file = st.file_uploader("मराठी PDF अपलोड करा", type="pdf")

if uploaded_file:
    st.success("PDF अपलोड झाले आहे! 📄")
    
    # PDF वाचा
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    
    # जर PDF मध्ये टेक्स्ट नसेल (स्कॅन केलेला असेल), तर Bhashini OCR वापरा
    if not text.strip():
        st.info("PDF मध्ये टेक्स्ट नाही — OCR वापरत आहे...")
        # Bhashini OCR API ला कॉल करा (सोपे उदाहरण)
        # (हे भविष्यात वाढवता येईल; सध्या फक्त टेक्स्ट PDF सपोर्ट)
        text = "हा PDF स्कॅन केलेला आहे. कृपया टेक्स्ट असलेला PDF अपलोड करा."
    
    if text.strip():
        # टेक्स्ट चिंक्यांमध्ये विभागा
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        
        # एम्बेडिंग्ज + व्हेक्टर डीबी
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = Chroma.from_texts(chunks, embeddings)
        
        # LLM (Llama 3 via Hugging Face)
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.3,
            huggingfacehub_api_token="तुझी_HUGGINGFACE_TOKEN_इथे_टाका"  # ← ही ओळ बदल!
        )
        
        # RAG चेन
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # प्रश्न विचारा
        user_question = st.text_input("तुमचा प्रश्न मराठीत विचारा:")
        
        if user_question:
            with st.spinner("मराठीत उत्तर शोधत आहे... 🤖"):
                try:
                    response = qa_chain({"query": user_question})
                    st.subheader("उत्तर:")
                    st.write(response["result"])
                    
                    # स्रोत दाखवा (ऐच्छिक)
                    with st.expander("स्रोत (PDF मधील भाग)"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.write(f"भाग {i+1}: {doc.page_content[:200]}...")
                except Exception as e:
                    st.error(f"त्रुटी: {str(e)}")
                    st.write("कृपया Hugging Face आणि Bhashini API keys तपासा.")
    else:
        st.warning("PDF मध्ये काहीही टेक्स्ट नाही. कृपया दुसरा PDF अपलोड करा.")

# ----------------------------
# 4. सूचना
# ----------------------------
st.markdown("---")
st.caption("हा प्लॅटफॉर्म मराठी भाषिकांसाठी AI ची शक्ती देतो. 🙏")
