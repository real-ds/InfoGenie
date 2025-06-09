import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, json_save_tool, shorten_tool, markdown_tool
import os
import datetime

# -------- Load environment -------- #
load_dotenv()

# -------- Define Pydantic Response -------- #
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# -------- Initialize LLM and Parser -------- #
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# -------- Define Prompt -------- #
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are 'SageBot', a research-focused AI agent. Use tools like Wikipedia, DuckDuckGo, and SaveTool to research and store information.
        You must summarize the topic clearly using reliable sources. Wrap the result in the required format with NO extra commentary.
        If user includes phrases like 'save this' or 'save to file', invoke the save_text_to_file tool.
        {format_instructions}
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())

# -------- Register Tools -------- #
tools = [
    search_tool,
    wiki_tool,
    save_tool,
    json_save_tool,
    shorten_tool,
    markdown_tool
]

# -------- Build Agent Executor -------- #
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# -------- Streamlit App -------- #
st.set_page_config(page_title="🧠 SageBot Research Assistant", page_icon="🧠")
st.title("🧠 SageBot: Research with AI Tools")
st.write("Type a topic below.")

query = st.text_input("🔍 Enter your research query")

if query:
    with st.spinner("🔎 Researching..."):
        try:
            raw_response = agent_executor.invoke({
                "query": query,
                "chat_history": [],
                "agent_scratchpad": [],
            })

            structured = parser.parse(raw_response.get("output"))

            st.success("✅ Research complete!")

            st.subheader("📝 Summary")
            st.markdown(structured.summary)

            st.subheader("📌 Topic")
            st.write(structured.topic)

            st.subheader("🔗 Sources")
            for src in structured.sources:
                st.write(f"- {src}")

            st.subheader("🛠️ Tools Used")
            st.write(", ".join(structured.tools_used))

            # Prepare download text
            download_text = f"""
--- RESEARCH OUTPUT ---
Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Topic: {structured.topic}

Summary:
{structured.summary}

Sources: {structured.sources}
Tools Used: {structured.tools_used}
"""

            filename = f"research_summary_{structured.topic.replace(' ', '_')}.txt"

            st.download_button(
                label="💾 Download Summary as .txt",
                data=download_text,
                file_name=filename,
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
