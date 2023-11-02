from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os


from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def init():
    # Load the Hugging Face API key from the environment variable
    load_dotenv()
    # test that the API key exists
    if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None or os.getenv("HUGGINGFACE_API_KEY") == "":
        print("HUGGINGFACE_API_KEY is not set")
        exit(1)
    else:
        print("HUGGINGFACE_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="It's Me 🤖"
    )

def main():
    init()
    repo_id = 'google/flan-t5-base'
    llm = HuggingFaceHub(repo_id=repo_id,
                        model_kwargs = {'temperature': 1e-10, "max_length":200})

    conversation_buf = ConversationChain(
    llm=llm)


    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Chatbot🤖")

# sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")


        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Loading..."):
                response = conversation_buf.predict(input = user_input)
            st.session_state.messages.append(
                AIMessage(content=response))
            

    messages = st.session_state.get('messages', [])
    # display message history
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')
# st.text_area("Output", response)
if __name__ == '__main__':
    main()