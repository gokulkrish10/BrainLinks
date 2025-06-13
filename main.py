import streamlit as st
import pickle as pt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from gtts import gTTS
import io

load_dotenv()

# Enhanced Custom CSS with better color schemee
import streamlit as st

st.set_page_config(
    page_title= "BrainLinks",
    page_icon = "brainlink.jpg",
    layout = "wide",
)
#api_keys = os.getenv("GOOGLE_API_KEY")

api_keys = st.secrets['GOOGLE_API_KEY']
# Inject CSS to hide Streamlit branding
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');
    
    /* Main background and theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        background-attachment: fixed;
        color: #e2e8f0;
    }
    
    /* Custom container */
    .main-container {
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        # # background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
        # -webkit-background-clip: text;
        # -webkit-text-fill-color: transparent;
        # background-clip: text;
        # margin-bottom: 0.5rem;
        # text-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .main-subtitle {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        color: #94a3b8;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Capabilities section */
    .capabilities-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .capabilities-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #06b6d4;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .capability-card {
        background: rgba(51, 65, 85, 0.6);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
    }
    
    .capability-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .capability-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }
    
    .capability-desc {
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .how-to-use {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .how-to-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #22c55e;
        margin-bottom: 1rem;
    }
    
    .step {
        font-family: 'Poppins', sans-serif;
        color: #e2e8f0;
        margin: 0.8rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .step::before {
        content: "▶";
        position: absolute;
        left: 0;
        color: #06b6d4;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(51, 65, 85, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .sidebar-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #06b6d4;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem;
        background: rgba(6, 182, 212, 0.1);
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(51, 65, 85, 0.8);
        border: 2px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 0.8rem;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        color: #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        outline: none;
        background: rgba(51, 65, 85, 0.9);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.5);
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Poppins', sans-serif;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Poppins', sans-serif;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Poppins', sans-serif;
        border: none;
    }
    
    /* Answer section styling */
    .answer-container {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .answer-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #06b6d4;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .answer-text {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #e2e8f0;
        background: rgba(30, 41, 59, 0.6);
        padding: 2rem;
        border-radius: 15px;
        border-left: 4px solid #06b6d4;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .sources-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #8b5cf6;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .source-link {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        font-family: 'Fira Code', monospace;
        font-size: 0.9rem;
        color: #c4b5fd;
        transition: all 0.3s ease;
        word-break: break-all;
    }
    
    .source-link:hover {
        background: rgba(139, 92, 246, 0.2);
        transform: translateX(8px);
        border-color: rgba(139, 92, 246, 0.5);
    }
    
    /* Loading animation */
    .loading-text {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        color: #06b6d4;
        text-align: center;
        padding: 1.5rem;
        background: rgba(6, 182, 212, 0.1);
        border-radius: 12px;
        animation: pulse 2s infinite;
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.02); }
        100% { opacity: 0.7; transform: scale(1); }
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
    }
    
    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .feature-text {
        font-family: 'Poppins', sans-serif;
        color: #06b6d4;
        font-weight: 500;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


st.write("⤴️Use the **Sidebar** to paste the Links")

# Custom Vector Store Class to replace FAISS
class SimpleVectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.array(embeddings)
        self.doc_embeddings = {}
        
    def similarity_search(self, query_embedding, k=4):
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k similar documents
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.documents[i] for i in top_indices]
    
    def as_retriever(self):
        return SimpleRetriever(self)

class SimpleRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_keys
        )
    
    def get_relevant_documents(self, query):
        query_embedding = self.embedding_function.embed_query(query)
        return self.vectorstore.similarity_search(query_embedding)

def load_url_content(url):
    """Custom function to load content from URL using requests and BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return Document(page_content=text, metadata={"source": url})
        
    except Exception as e:
        st.error(f"Error loading URL {url}: {str(e)}")
        return None

# Main header with custom styling


st.markdown(' <h1 class="main-header"> 🧠BrainLinks </h1> ', unsafe_allow_html=True)





st.markdown('<p class="main-subtitle">Advanced AI-powered analysis of news articles, research papers, and web content</p>', unsafe_allow_html=True)
st.markdown("""
<div class="how-to-use">
    <div class="how-to-title">📋 How to Use This Tool</div>
    <div class="step">Enter the number of link you want to analyze (e.g., 2, 3, or more) in the sidebar section</div>
    <div class="step">Paste the URLs of news articles, research papers, or web pages</div>
    <div class="step">Click "Process URLs" to let the AI analyze and understand the content</div>
    <div class="step">Wait for processing to complete (this may take a few moments)</div>
    <div class="step">Ask questions about the content in natural language</div>
    <div class="step">Review the AI-generated answers with source citations</div>
</div>
""", unsafe_allow_html=True)
# Capabilities Section
st.markdown("""
<div class="capabilities-section">
   <div class="capability-card">
        <div class="capability-title">🌐 Web Content Extraction</div>
        <div class="capability-desc">Automatically extract and clean content from web pages, removing ads, navigation, and other noise to focus on the main content.</div>
    </div>
<div class="capability-card">
        <div class="capability-title">💡 Comparative Analysis</div>
        <div class="capability-desc">Compare information across multiple sources, identify contradictions, verify facts, and provide balanced perspectives on topics.</div>
    </div>
<div class="capability-card">
        <div class="capability-title">📚 Source Attribution</div>
        <div class="capability-desc">Every answer includes proper source citations, allowing you to verify information and explore the original content.</div>
    </div> 
     <div class= "capability-card">
           <div class= "capability-title"> 🔉Text to Speeach </div> 
            <div class = "capability-desc"> Hear Your Answers, Not Just Read Them  </div>
             </div>   
</div>
         
""", unsafe_allow_html=True)

# How to Use Section




# Sidebar with enhanced styling
st.sidebar.markdown('<div class="sidebar-title">🔗 URL Configuration</div>', unsafe_allow_html=True)

urls_no = st.sidebar.text_input("📝 Number of Links", placeholder="e.g., 3", help="Enter how many link you want to analyze")
process_no = st.sidebar.button("Submit")
urls = []
if urls_no and urls_no.isdigit():
    urls_no = int(urls_no)
    
    for i in range(urls_no):
        url = st.sidebar.text_input(
            f"🌐 URL {i+1}", 
            key=f"url_{i}", 
            placeholder="https://example.com/article",
            help=f"Paste the URL of article/paper #{i+1}"
        )
        if url:  # Only append non-empty URLs
            urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process Link", help="Click to analyze and process all links")

if process_url_clicked:
    if urls:
        st.sidebar.success("Go to Ask questions section 😊")

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
**💡 Tips:**
- Use high-quality news sources
- Ensure link are accessible
- Processing time varies by content length
""")

file_path = "vector_store_data.pkl"
main_placeholder = st.empty()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',
    google_api_key=api_keys,
    temperature=0
)

if process_url_clicked:
    if urls:  # Check if URLs exist
        try:
            main_placeholder.markdown('<div class="loading-text">🔄 Starting URL processing...</div>', unsafe_allow_html=True)
            
            # Load documents using custom function
            data = []
            for i, url in enumerate(urls):
                if url.strip():  # Only process non-empty URLs
                    main_placeholder.markdown(f'<div class="loading-text">📖 Loading content from link {i+1} of {len(urls)}...</div>', unsafe_allow_html=True)
                    doc = load_url_content(url.strip())
                    if doc:
                        data.append(doc)
            
            if not data:
                st.error("❌ No valid content could be loaded from the provided links.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ",", " "],
                    chunk_size=1000,
                    chunk_overlap=500
                )

                main_placeholder.markdown('<div class="loading-text">✂️ Splitting text into analyzable chunks...</div>', unsafe_allow_html=True)
                docs = text_splitter.split_documents(data)
                
                main_placeholder.markdown('<div class="loading-text">🧠 Creating AI embeddings for semantic search...</div>', unsafe_allow_html=True)
                embedding_model = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=api_keys
                )
                
                # Create embeddings for all documents
                doc_texts = [doc.page_content for doc in docs]
                embeddings = embedding_model.embed_documents(doc_texts)
                if not embeddings:
                    main_placeholder.markdown('<div class="loading-text"> 😔 Nothing is extracted from the link/s </div>' , unsafe_allow_html=True )
                    st.error("⚠️ Hmm... we couldn’t analyze the content from this link. Please try with different or more informative pages.")
                else:

                

                
                # converting 1d into 2d 

            
                    vectorstore = SimpleVectorStore(docs, embeddings)
                    
                    main_placeholder.markdown('<div class="loading-text">💾 Saving processed knowledge base...</div>', unsafe_allow_html=True)
                    
                    # Save the vectorstore data
                    store_data = {
                        'documents': docs,
                        'embeddings': embeddings
                    }
                    
                    with open(file_path, 'wb') as f:
                        pt.dump(store_data, f)
                    
                    main_placeholder.success("✅ Processing completed successfully! You can now ask questions about the analyzed content.")
            
        except Exception as e:
            st.error(f"❌ Error processing URLs: {str(e)}")
            st.info("💡 Try installing required packages: pip install scikit-learn numpy requests beautifulsoup4")
    else:
        st.warning("⚠️ Please enter valid links before processing.")

# Query input with enhanced styling
st.markdown('<div id="section2"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown("### 🤔 Ask Your Question")
st.markdown("*Ask anything about the processed content - comparisons, summaries, specific facts, analysis, etc.*")

with st.form("Question"):

    query = st.text_input(
    "", 
    placeholder="e.g., 'What are the main arguments presented in these articles?' or 'Compare the different viewpoints on this topic'", 
    key="query_input",
    help="Ask questions in natural language about the content you've processed"
    )

    submitted = st.form_submit_button("Submit")

if submitted:

    if query:
        if os.path.exists(file_path):
            try:
                with st.spinner("🔍 Searching through processed content and generating response..."):
                    with open(file_path, 'rb') as f:
                        store_data = pt.load(f)
                        
                    # Recreate vector store
                    vectorstore = SimpleVectorStore(store_data['documents'], store_data['embeddings'])
                    
                    # Create a simple QA chain
                    retriever = vectorstore.as_retriever()
                    relevant_docs = retriever.get_relevant_documents(query)
                    
                    # Create context from relevant documents
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    sources = list(set([doc.metadata.get("source", "") for doc in relevant_docs]))
                    
                    # Create prompt for the LLM
                    prompt = f"""Based on the following context from news articles and documents, please provide a comprehensive, accurate, and well-structured answer to the question. 

    Context:
    {context}

    Question: {query}

    Please provide a detailed answer that:
    1. Directly addresses the question
    2. Uses specific information from the context
    3. Is well-organized and easy to understand
    4. Mentions different perspectives if they exist
    5. Is factual and based only on the provided content

    Answer:"""
                    
                    # Get response from LLM
                    response = llm.invoke(prompt)
                
                    text_input = response.content.replace("*" , " ")
                
                    try:
                        tts = gTTS(text=text_input , lang='en' , slow = False)
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)

                    # Display the audio player
                        st.audio(audio_fp, format="audio/mpeg")

                    except Exception as e:
                                st.error(f"An error occurred: {e}")
                        
                    # Display answer with custom styling
                    st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                    st.markdown('<div class="answer-header">💡 AI Analysis & Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-text">{response.content}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources if available
                    if sources and any(sources):
                        st.markdown('<div class="sources-header">📚 Source References</div>', unsafe_allow_html=True)
                        for i, source in enumerate(sources):
                            if source:
                                st.markdown(f'<div class="source-link">🔗 Source {i+1}: {source}</div>', unsafe_allow_html=True)
                                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("💡 Please try rephrasing your question or check if the links were processed correctly.")
        else:
            st.warning("⚠️ Please process links first before asking questions.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-family: Poppins, sans-serif; padding: 2rem;">
        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">🚀 Fueled by Semantic Search & Advanced NLP</div>
        <div style="font-size: 0.9rem;">Built with LangChain, Streamlit & Custom Vector Search</div>
    </div>
    """, 
    unsafe_allow_html=True
)
