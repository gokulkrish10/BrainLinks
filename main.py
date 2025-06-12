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

# Enhanced Custom CSS with better color scheme
import streamlit as st

st.set_page_config(
    page_title= "BrainLinks",
    page_icon = "brainlink.jpg",
    layout = "wide",
)

api_keys = st.secrets['GOOGLE_API_KEY']
# Inject CSS to hide Streamlit branding
st.markdown(
    """
    <style>
    
    [data-testid="stSidebar"] {
         background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
         color: #e2e8f0;
    }
     button[kind="primary"] {
        background-color: #1f77b4;  /* Blue */
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: 0.3s;
    }

    button[kind="primary"]:hover {
        background-color: #135d96;
    }
    
    /* Enhanced Sidebar Toggle Button - Visible in all themes */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 48px !important;
        height: 48px !important;
        padding: 8px !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 999999 !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background: linear-gradient(135deg, #5b21b6 0%, #6d28d9 100%) !important;
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    [data-testid="collapsedControl"] svg {
        color: white !important;
        width: 24px !important;
        height: 24px !important;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) !important;
    }
    
    /* Force visibility in all color schemes */
    @media (prefers-color-scheme: light) {
        [data-testid="collapsedControl"] {
            background: linear-gradient(135deg, #4338ca 0%, #7c2d12 100%) !important;
            border: 2px solid rgba(0, 0, 0, 0.1) !important;
            box-shadow: 0 4px 15px rgba(67, 56, 202, 0.3), 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        }
        
        [data-testid="collapsedControl"]:hover {
            background: linear-gradient(135deg, #3730a3 0%, #991b1b 100%) !important;
            box-shadow: 0 8px 25px rgba(67, 56, 202, 0.5), 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="collapsedControl"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4), 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        }
        
        [data-testid="collapsedControl"]:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6), 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        }
    }
    
    /* Sidebar Toggle Animation */
    [data-testid="collapsedControl"] {
        animation: pulseGlow 3s ease-in-out infinite !important;
    }
    
    @keyframes pulseGlow {
        0%, 100% { 
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4), 0 0 0 0 rgba(79, 70, 229, 0.7) !important;
        }
        50% { 
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4), 0 0 0 10px rgba(79, 70, 229, 0) !important;
        }
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Enhanced Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-gradient: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-heavy: 0 15px 35px rgba(0, 0, 0, 0.3);
        --transition-smooth: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        --border-radius: 20px;
        --text-primary: #ffffff;
        --text-secondary: #b8c5d1;
        --text-accent: #4facfe;
    }
    
    /* Enhanced Main Background */
    .stApp {
        background: var(--dark-gradient);
        background-attachment: fixed;
        color: var(--text-primary);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Glassmorphism Main Container */
    .main-container {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 3rem;
        margin: 1.5rem;
        box-shadow: var(--shadow-heavy);
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    }
    
    /* Enhanced Header Styling */
    .main-header {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
        position: relative;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        color: var(--text-secondary);
        margin-bottom: 3rem;
        font-weight: 400;
        opacity: 0.9;
        line-height: 1.6;
    }
    
    /* Enhanced Capabilities Section */
    .capabilities-section {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 2.5rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .capabilities-section::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(135deg, #667eea, #764ba2, #4facfe);
        border-radius: var(--border-radius);
        z-index: -1;
        opacity: 0.3;
    }
    
    .capabilities-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-align: center;
        letter-spacing: -0.01em;
    }
    
    /* Premium Capability Cards */
    .capability-card {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.08) 0%, 
            rgba(255, 255, 255, 0.04) 100%);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }
    
    .capability-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .capability-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(79, 172, 254, 0.2);
        border-color: rgba(79, 172, 254, 0.4);
    }
    
    .capability-card:hover::before {
        left: 100%;
    }
    
    .capability-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }
    
    .capability-desc {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: var(--text-secondary);
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Enhanced How-to-use Section */
    .how-to-use {
        background: linear-gradient(135deg, 
            rgba(34, 197, 94, 0.1) 0%, 
            rgba(16, 185, 129, 0.1) 100%);
        border-left: 4px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        margin: 2.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.1);
    }
    
    .how-to-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #10b981;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    .step {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        margin: 1rem 0;
        padding: 1rem 0 1rem 2rem;
        position: relative;
        transition: var(--transition-smooth);
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .step::before {
        content: "▶";
        position: absolute;
        left: 0;
        color: #4facfe;
        font-weight: bold;
        font-size: 1.1rem;
        transition: var(--transition-smooth);
    }
    
    .step:hover {
        color: #4facfe;
        transform: translateX(8px);
    }
    
    .step:hover::before {
        color: #00f2fe;
        transform: scale(1.2);
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.08) 0%, 
            rgba(255, 255, 255, 0.04) 100%);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-light);
    }
    
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1.5rem;
        background-color: rgba(79, 172, 254, 0.1);
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(79, 172, 254, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Premium Input Styling */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.1) 0%, 
            rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: var(--text-primary);
        transition: var(--transition-smooth);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4facfe;
        box-shadow: 0 0 0 4px rgba(79, 172, 254, 0.2),
                    0 8px 25px rgba(79, 172, 254, 0.15);
        outline: none;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.15) 0%, 
            rgba(255, 255, 255, 0.08) 100%);
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        transition: var(--transition-smooth);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        width: 100%;
        position: relative;
        overflow: hidden;
        letter-spacing: 0.02em;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Enhanced Alert Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        font-family: 'Inter', sans-serif;
        border: none;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        font-family: 'Inter', sans-serif;
        border: none;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        font-family: 'Inter', sans-serif;
        border: none;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Premium Answer Container */
    .answer-container {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.08) 0%, 
            rgba(255, 255, 255, 0.04) 100%);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 3rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(79, 172, 254, 0.2);
        box-shadow: var(--shadow-heavy);
        position: relative;
        overflow: hidden;
    }
    
    .answer-container::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        border-radius: var(--border-radius);
        z-index: -1;
        opacity: 0.1;
    }
    
    .answer-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.01em;
    }
    
    .answer-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        line-height: 1.8;
        color: var(--text-primary);
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.06) 0%, 
            rgba(255, 255, 255, 0.03) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 16px;
        border-left: 4px solid #4facfe;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        font-weight: 400;
    }
    
    .sources-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #a855f7 0%, #e879f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 3rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.01em;
    }
    
    .source-link {
        background: linear-gradient(135deg, 
            rgba(168, 85, 247, 0.1) 0%, 
            rgba(232, 121, 249, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #d8b4fe;
        transition: var(--transition-smooth);
        word-break: break-all;
        position: relative;
        overflow: hidden;
    }
    
    .source-link::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .source-link:hover {
        background: linear-gradient(135deg, 
            rgba(168, 85, 247, 0.2) 0%, 
            rgba(232, 121, 249, 0.2) 100%);
        transform: translateX(12px) translateY(-2px);
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.2);
    }
    
    .source-link:hover::before {
        left: 100%;
    }
    
    /* Enhanced Loading Animation */
    .loading-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 500;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 2rem;
        background-color: rgba(79, 172, 254, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        animation: enhancedPulse 2s ease-in-out infinite;
        border: 1px solid rgba(79, 172, 254, 0.2);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.1);
    }
    
    @keyframes enhancedPulse {
        0%, 100% { 
            opacity: 0.8; 
            transform: scale(1); 
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.1);
        }
        50% { 
            opacity: 1; 
            transform: scale(1.02); 
            box-shadow: 0 12px 35px rgba(79, 172, 254, 0.2);
        }
    }
    
    /* Enhanced Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4facfe, #00f2fe, transparent);
        margin: 3rem 0;
        border-radius: 1px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Premium Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Feature Highlight */
    .feature-highlight {
        background: linear-gradient(135deg, 
            rgba(79, 172, 254, 0.1) 0%, 
            rgba(102, 126, 234, 0.1) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.1);
        transition: var(--transition-smooth);
    }
    
    .feature-highlight:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.2);
    }
    
    .feature-text {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        font-size: 1.15rem;
        letter-spacing: -0.01em;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-container {
            padding: 2rem;
            margin: 1rem;
        }
        
        .main-header {
            font-size: 2.5rem;
        }
        
        .capabilities-section {
            padding: 2rem;
        }
        
        .answer-container {
            padding: 2rem;
        }
        
        [data-testid="collapsedControl"] {
            width: 44px !important;
            height: 44px !important;
        }
    }
    
    /* Accessibility Enhancements */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* Focus indicators */
    button:focus-visible,
    input:focus-visible {
        outline: 2px solid #4facfe;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.write("Use the **Sidebar** to paste the Links")

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
                
                # Create custom vector store
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
