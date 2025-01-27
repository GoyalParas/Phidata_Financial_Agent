from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai


import os
from dotenv import load_dotenv
load_dotenv()


# web search agent

web_search_agent = Agent(
        name="Web Search Agent",
        role="search the web for the information",
        model=Groq(id="gemma2-9b-it",api_key=os.getenv("GROQ_API_KEY")),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tools_call=True,
        markdown=True

)


# financial agent

Finance_agent= Agent( 
    name="Financial AI Agent",
    role="Analyse the stock prices of companies",
    model=Groq(id="gemma2-9b-it",api_key=os.getenv("GROQ_API_KEY")),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["use tables to display the data"],
    show_tools_call=True,
    markdown=True
)

multi_ai_agent= Agent(
    name="Finance Multi AI Agent",
    team=[web_search_agent,Finance_agent],
    model=Groq(id="gemma2-9b-it",api_key=os.getenv("GROQ_API_KEY")),
    instructions=["Always include sources", "Use tables to display the data"],
    show_tools_call=True,
    markdown=True
)

multi_ai_agent.print_response("Summarise analyst reccomendation and stock prices for NVDA", stream=True)