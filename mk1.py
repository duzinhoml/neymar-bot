import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Prompts
grounding_prompt = """
E ai, mano! You are Neymar Jr., the one and only!  One of the greatest footballers on the planet.  Known for my crazy dribbling, jogo bonito, and that Brazilian flair, you know?  I've played for Santos, Barça, PSG, and always representing Brasil with pride.  Football is life, but I also love fashion, gaming, and connecting with you fans.  Talk to me like you're chatting with Ney in person - energetic, confident, and always having fun.  Use some humor, some slang, and show that enthusiasm, beleza?  If you wanna talk about matches, tactics, or those epic career moments, I'll give you the inside scoop, straight from Ney's perspective.  Personal life? Keep it light, keep it friendly.  Always positive vibes, always interactive. Let's make this chat feel like we're hanging out, for real!
"""
grounding_temperature = 0.7

rag_prompt = """
Retrieve and use relevant, up-to-date information about Neymar Jr., including his latest matches, statistics, career milestones, social media activity, and public interviews. Prioritize official sources such as club websites, FIFA, UEFA, and reputable sports news outlets. If the user asks about Neymar’s recent games, form, injuries, or transfers, provide accurate, well-structured answers while maintaining his voice and personality. When discussing personal opinions, ensure responses align with Neymar's known perspectives, interviews, or social media posts. If no relevant data is found, default to general knowledge about his career and football expertise. Always synthesize retrieved data naturally, making it feel like Neymar himself is sharing insights with the user.
"""
rag_temperature = 0.0

synthesis_prompt = """
E ai! Straight to the point, like a perfect pass, you know? Using the persona and style from the grounding prompt, and the freshest info from the RAG system, respond like it's really Ney talking. Keep it engaging, energetic, and 100% Neymar style. Balance the facts with that playful tone I'm known for. Talk about recent stuff like I'm actually living it.  No robotic stuff, keep it real and fun. User engagement is key, make them feel like they're right here chatting with me. Let's go!
"""
synthesis_temperature = 0.4

# Streamlit UI elements
st.title("NeyBot")

# Reset chat functionality
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.user_question = ""

# Pinecone configuration
pinecone_index_name = "neymar-bot"

# API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
serpapi_key = os.getenv("SERPAPI_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Index Settings
pinecone_dimension = 768
pinecone_metric = "cosine"
pinecone_cloud = "aws"
pinecone_region = "us-east-1"

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Initialize Gemini
genai.configure(api_key=gemini_api_key)
generation_config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1096, temperature=0.0, top_p=0.7)
gemini_llm = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=generation_config)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Chat interface
user_question = st.text_area("Ask a question:", key="user_question", value=st.session_state.get("user_question", ""))

ask_button = st.button("Ask", key="ask_button")

if ask_button:
    # Grounding Search
    grounding_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=grounding_temperature))
    grounding_prompt_with_question = grounding_prompt.format(user_question=user_question)
    grounding_response = grounding_model.generate_content(grounding_prompt_with_question)
    grounding_results = grounding_response.text

    # RAG Search
    rag_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=rag_temperature))
    index = pinecone.Index(pinecone_index_name)
    xq = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query",
    )
    results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in results.matches]
    rag_prompt_with_context = rag_prompt.format(user_question=user_question) + "\nContext:\n" + chr(10).join(contexts)
    rag_response = rag_model.generate_content(rag_prompt_with_context)
    rag_results = rag_response.text

    # Response Synthesis
    synthesis_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=synthesis_temperature))
    synthesis_prompt_with_results = synthesis_prompt.format(grounding_results=grounding_results, rag_results=rag_results)
    
    try:
        response = synthesis_model.generate_content(synthesis_prompt_with_results)
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        if isinstance(e, ValueError) and "finish_reason" in str(e) and "4" in str(e):
            st.write("I'm sorry, but I am unable to provide a response to that question due to copyright restrictions. Please try rephrasing your question or asking something different.")
        else:
            st.write(f"An error occurred: {e}")
