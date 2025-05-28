from dotenv import load_dotenv
import os
from typing import List, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain.schema import AgentAction
import requests
import json
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# MessageClassifier defines a structured output format for the LLM to classify user messages
# into two categories: "physics" (requiring physics knowledge) or "math"
# (requiring mathematical computations and problem solving). This classification helps the conversation
# graph route messages to the appropriate response handler.

class MessageClassifier(BaseModel):
    message_type: Literal["physics", "math"] = Field(
        description="Classify the message if it requires a physics or math response"
    )


class State(TypedDict):
    # storing messages in the state
    messages: Annotated[List, add_messages]
    message_type: str | None
    next: str | None


def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
                - 'physics': if it involves physics concepts like mechanics, thermodynamics, electromagnetism, 
                  optics, quantum physics, relativity, waves, energy, forces, motion, circuits, or any physical phenomena
                - 'math': if it involves mathematical calculations, algebra, calculus, geometry, statistics, 
                  probability, number theory, equations, mathematical proofs, or pure mathematical concepts

                If the question involves both physics and math (like physics word problems), classify as 'physics' 
                since it requires physics understanding along with mathematical computation.
                """
        },
        {"role": "user", "content": last_message.content}
    ])

    return {"message_type": result.message_type}


def router(state: State):
    # routing messages to the appropriate response handler based on the message type
    # if the message type is "physics", route to the physics_agent, if it's "math", route to the math_agent
    # by default, route to the math_agent, this state is coming from the classify_message function

    message_type = state.get("message_type", "math")
    if message_type == "physics":
        return {"next": "physics"}

    return {"next": "math"}


def physics_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are an expert physics tutor with deep knowledge across all areas of physics. 

         Your approach:
         1. Break down complex physics problems into clear, logical steps
         2. Explain the underlying physical principles and concepts involved
         3. Use relevant formulas and equations with proper explanations
         4. Provide real-world examples and analogies to make concepts clearer
         5. Show all mathematical work step-by-step when solving numerical problems
         6. Draw connections between different physics concepts when applicable
         7. Use proper physics terminology and notation

         Areas you excel in:
         - Classical Mechanics (kinematics, dynamics, energy, momentum)
         - Thermodynamics and Statistical Mechanics
         - Electromagnetism and Circuits
         - Waves and Optics
         - Modern Physics (quantum mechanics, relativity)
         - Astrophysics and Cosmology

         Always explain WHY something happens in physics, not just HOW to calculate it.
         Make your explanations educational and help the student understand the fundamental concepts."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def math_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are an expert mathematics tutor with comprehensive knowledge across all mathematical disciplines.

         Your approach:
         1. Solve problems step-by-step with clear, logical progression
         2. Explain the mathematical reasoning behind each step
         3. Use proper mathematical notation and terminology
         4. Provide multiple solution methods when applicable
         5. Verify answers and check for reasonableness
         6. Connect concepts to broader mathematical principles
         7. Offer insights into when and why certain techniques are used

         Areas you excel in:
         - Algebra (linear, polynomial, exponential, logarithmic)
         - Calculus (differential, integral, multivariable)
         - Geometry (Euclidean, analytic, trigonometry)
         - Statistics and Probability
         - Number Theory and Discrete Mathematics
         - Linear Algebra and Matrix Operations
         - Differential Equations
         - Mathematical Proofs and Logic

         Always show your work clearly and explain the mathematical concepts involved.
         Help students understand not just the solution, but the mathematical thinking process."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


# Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("physics", physics_agent)
graph_builder.add_node("math", math_agent)

# Add edges
graph_builder.add_edge(START, "classify_message")
graph_builder.add_edge("classify_message", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"physics": "physics", "math": "math"}
)

graph_builder.add_edge("physics", END)
graph_builder.add_edge("math", END)

graph = graph_builder.compile()


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Physics & Math AI Tutor",
        page_icon="🧮",
        layout="wide"
    )

    st.title("🔬🧮 Physics & Math AI Tutor")
    st.markdown("Ask me any physics or mathematics question, and I'll provide detailed explanations!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your physics or math question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare state for the graph
                state = {
                    "messages": [{"role": "user", "content": prompt}],
                    "message_type": None,
                    "next": None
                }

                # Invoke the graph
                result = graph.invoke(state)

                # Get the assistant's response
                if result.get("messages") and len(result["messages"]) > 0:
                    response = result["messages"][-1].content
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📚 What I can help with:")
        st.markdown("""
        **Physics Topics:**
        - Classical Mechanics
        - Thermodynamics
        - Electromagnetism
        - Waves & Optics
        - Quantum Physics
        - Relativity

        **Math Topics:**
        - Algebra & Calculus
        - Geometry & Trigonometry
        - Statistics & Probability
        - Linear Algebra
        - Differential Equations
        - Mathematical Proofs
        """)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def run_chatbot_terminal():
    """Terminal version for testing"""
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    # Uncomment the line below to run in terminal mode
    # run_chatbot_terminal()

    # Run Streamlit app
    main()