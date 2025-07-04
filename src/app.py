import streamlit as st
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from core.chatbot import Chatbot

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    try:
        bot = Chatbot()
        if bot.initialize():
            st.success("✅ Chatbot loaded successfully!")
            return bot
        else:
            st.error("❌ Failed to load chatbot model")
            return None
    except Exception as e:
        st.error(f"❌ Error loading chatbot: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Smart ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Smart ChatBot</h1>
        <p>Simple AI Chatbot with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Load chatbot
        bot = load_chatbot()
        
        if bot is None:
            st.error("🚨 Failed to load chatbot model. Please check your setup.")
            st.stop()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm Smart ChatBot. I use machine learning to understand your messages. How can I help you today?"
            })
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input (outside columns for proper functionality)
    if prompt := st.chat_input("💬 Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response immediately
        response, intent, confidence = bot.get_response(prompt)
        
        # Store response info for sidebar
        st.session_state.last_intent = intent
        st.session_state.last_confidence = confidence
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Enhanced Sidebar
    with col2:
        # AI Response Analytics
        st.markdown("""
        <div class="stats-container">
            <h3>🧠 AI Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'last_intent') and hasattr(st.session_state, 'last_confidence'):
            st.metric("Last Intent", st.session_state.last_intent)
            st.metric("Confidence", f"{st.session_state.last_confidence:.3f}")
            
            # Confidence bar
            confidence_color = "green" if st.session_state.last_confidence > 0.7 else "orange" if st.session_state.last_confidence > 0.4 else "red"
            st.markdown(f"""
            <div style="background-color: {confidence_color}; width: {st.session_state.last_confidence*100}%; height: 10px; border-radius: 5px;"></div>
            """, unsafe_allow_html=True)
        
        # Enhanced Controls
        st.markdown("""
        <div class="feature-card">
            <h4>🔧 Chat Controls</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm Smart ChatBot. I use machine learning to understand your messages. How can I help you today?"
            })
            st.rerun()
        
        if st.button("📥 Export Chat", use_container_width=True):
            chat_export = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            st.download_button(
                label="Download Chat History",
                data=chat_export,
                file_name="chat_history.txt",
                mime="text/plain"
            )
        
        # Enhanced Statistics
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Session Statistics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("User Messages", user_messages)
        with col_b:
            st.metric("Bot Responses", bot_messages)
        
        # Interactive Technology Stack
        st.markdown("""
        <div class="feature-card">
            <h4>⚙️ How It Works</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive tech showcase
        tech_option = st.selectbox(
            "Explore the technology:",
            ["🧠 Machine Learning", "📝 Text Processing", "🎯 Intent Recognition", "💻 Interface"]
        )
        
        if tech_option == "🧠 Machine Learning":
            st.info("**Naive Bayes Algorithm** - Uses probability to classify your messages into different intents")
            st.code("from sklearn.naive_bayes import MultinomialNB", language="python")
            
        elif tech_option == "📝 Text Processing":
            st.info("**NLTK Library** - Cleans and processes your text by removing stopwords and tokenizing")
            st.code("from nltk.tokenize import word_tokenize", language="python")
            
        elif tech_option == "🎯 Intent Recognition":
            st.info("**TF-IDF Vectorizer** - Converts your text into numbers the AI can understand")
            st.code("from sklearn.feature_extraction.text import TfidfVectorizer", language="python")
            
        elif tech_option == "💻 Interface":
            st.info("**Streamlit** - Creates this beautiful web interface you're using right now")
            st.code("import streamlit as st", language="python")

if __name__ == "__main__":
    main()