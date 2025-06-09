from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json


# ---------------------- SAVE TOOL ---------------------- #
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- RESEARCH OUTPUT ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"✅ Data successfully saved to '{filename}'"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Use this tool to save structured research summaries to a local .txt file."
)


# ---------------------- SEARCH TOOL ---------------------- #
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Use DuckDuckGo to search the web for real-time information and recent updates."
)


# ---------------------- WIKIPEDIA TOOL ---------------------- #
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wiki_tool = Tool(
    name="wikipedia_query",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description="Use Wikipedia to find factual, encyclopedic summaries of people, events, or topics."
)


# ---------------------- JSON EXPORT TOOL ---------------------- #
def save_to_json(data: str, filename: str = "research_output.json") -> str:
    try:
        parsed = json.loads(data.replace("'", '"')) if isinstance(data, str) else data
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(parsed, f, indent=4)
            f.write(",\n")
        return f"✅ Research saved in JSON format to '{filename}'"
    except Exception as e:
        return f"❌ Failed to save JSON: {e}"

json_save_tool = Tool(
    name="save_as_json",
    func=save_to_json,
    description="Use this tool to save the research result as a JSON object (for programmatic use)."
)


# ---------------------- SUMMARY LENGTH TOOL ---------------------- #
def shorten_summary(data: str) -> str:
    sentences = data.split(". ")
    short_summary = ". ".join(sentences[:2]) + ("..." if len(sentences) > 2 else "")
    return short_summary

shorten_tool = Tool(
    name="shorten_summary",
    func=shorten_summary,
    description="Use this tool to shorten a long summary into 2-3 lines."
)


# ---------------------- FORMAT TOOL (Markdown) ---------------------- #
def format_markdown(data: str) -> str:
    return f"```\n{data}\n```"

markdown_tool = Tool(
    name="format_markdown",
    func=format_markdown,
    description="Wraps the research output in markdown-style code block formatting."
)
