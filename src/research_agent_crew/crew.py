from crewai import Agent,Task,Process,Crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import agent,task,crew,CrewBase
from crewai import LLM
from typing import List
import os
from dotenv import load_dotenv
from .tools.research_tool import ResearchTool
load_dotenv()

llm = LLM(
    model="openrouter/arcee-ai/trinity-mini:free",
    api_base="https://openrouter.ai/api/v1",
    api_key= os.getenv("OPENAI_API_KEY")
)

tools = [ResearchTool()]

@CrewBase
class ResearchAgentCrew():
    agents: List[Agent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_summarization_agent(self) -> Agent:
        return Agent(
            config = self.agents_config["research_summarization_agent"],
            llm = llm,
            tools = tools,
            verbose = True
        )
    
    @task
    def summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarization_task"]
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents = self.agents,
            tasks=self.tasks,
            llm=llm,
            process = Process.sequential,
            verbose=True
        )