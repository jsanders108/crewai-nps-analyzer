import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, MDXSearchTool, CSVSearchTool
from langchain_openai import ChatOpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct full file paths
survey_data_path = os.path.join(script_dir, 'mock-nps-survey-data.csv')
objectives_path = os.path.join(script_dir, 'nps_analysis_objectives.md')


# FileReadTool allows the agent to read a specific file
read_data = FileReadTool(file_path=survey_data_path)
read_objectives = FileReadTool(file_path=objectives_path)

# MDXSearchTool allows us to perform RAG over our objectives
search_objectives = MDXSearchTool(mdx=objectives_path)

# CSVSearchTool allows us to perform RAG over our survey data
search_data = CSVSearchTool(csv=survey_data_path)


@CrewBase
class NpsAnalyzerCrew():
	"""NpsAnalyzer crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self) -> None:
		self.OpenAIGPT35 = ChatOpenAI(
			model_name="gpt-3.5-turbo", 
			temperature=0)
		
		self.OpenAIGPT4o = ChatOpenAI(
			model_name="gpt-4o", 
			temperature=0)

	@agent
	def analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['analyst'],
			tools=[read_objectives, search_objectives, read_data, search_data],
			allow_delegation=False,
			llm=self.OpenAIGPT4o,
			verbose=True
		)
	
	@agent
	def report_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['report_writer'],
			tools=[read_objectives, search_objectives],
			allow_delegation=False,
			llm=self.OpenAIGPT4o,
			verbose=True
		)

	@task
	def analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['analysis_task'],
			agent=self.analyst()
		)

	@task
	def report_writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['report_writing_task'],
			agent=self.report_writer(),
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the NpsAnalyzer crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			memory=True
		)