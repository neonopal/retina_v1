from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.agents import initialize_agent
import os

def google_serper_agent():
    google_search = GoogleSerperAPIWrapper()
    serper_api_key = os.environ["SERPER_API_KEY"]

    tools = [
        Tool(
            name = "Google Search",
            func= google_search.run,
            description= "Useful to search in Google Search. User by default"
        )
    ]

    # google_search.run("tahun berapa sinetron si doel ditayangkan")
    return tools