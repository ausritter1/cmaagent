__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Streamlit UI
st.title('Competitor Mapping Agent')

# Request for Serper and OpenAI API keys from the user
st.header("Enter Your API Keys")

serper_api_key = st.text_input("Enter your Serper API Key", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Input field for the startup/industry to analyze
topic = st.text_input('Enter Startup or Industry for Competitor Mapping')

# Button to start the process
if st.button('Run Competitor Mapping Agent') and topic and serper_api_key and openai_api_key:

    # Set environment variables with user-provided API keys
    os.environ["SERPER_API_KEY"] = serper_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize the search tool with user-provided Serper API key
    search_tool = SerperDevTool()

    # Competitor Mapping Researcher Agent
    researcher = Agent(
        role='Senior Researcher',
        goal='Identify and map out the competitive landscape for a given startup or industry. Do NOT run queries related entirely to finding information about one startup, that is a waste of time. Instead look for industry level research and aggregate the findings.',
        verbose=True,
        memory=True,
        max_iter=10,
        backstory=(
            "You look for market maps and industry reports related to the specific tech vertical {topic}"
            "You focus more on the industry dynamics than the particular startups for the report"
        ),
        tools=[search_tool],
        allow_delegation=True
    )

    # Writer Agent
    writer = Agent(
        role='Writer',
        goal='Deliver a 3 paragraph summary of the competitive landscape based on the researcher agent’s findings, including insights into key competitors.',
        verbose=True,
        memory=True,
        max_iter=3,
        backstory=(
            "Objective and precise, you focus on summarizing the competitive landscape efficiently."
            "You are comfortable highlighting key players and identifying competitive advantages and risks."
            "You do not get held up on going down a rabbit hole of finding info on one company. Focus insights at the industry level and discuss trends and shifts within the user specific industry."
        ),
        tools=[search_tool],
        allow_delegation=False
    )

    # Define the research task
    research_task = Task(
        description=(
            "Research the {topic} vertical within the tech industry."
            "Visit websites, news sources, and startup databases to gather insights on competitors’ business models, strengths, and weaknesses."
            "Provide detailed findings about competitors, including any unique advantages or challenges they present."
            "Mention at minimum 10 related innovative startups (NOT big companies), but do not research or write deep details about them"
        ),
        expected_output='Competitor map and key details.',
        tools=[search_tool],
        agent=researcher,
    )

    # Define the writing task
    write_task = Task(
        description=(
            "Summarize the findings of the research agent, providing a concise overview of the competitive landscape."
            "Highlight the most important competitors and their market positions."
            "Do not run queries related to only specific startups. Research at the industry level but take not of individual startups."
        ),
        expected_output='A 3 paragraph final competitive landscape summary for the target company.',
        tools=[search_tool],
        agent=writer,
        async_execution=False,
        output_file='competitive-analysis-summary.md'  # Output file to save the summary (optional)
    )

    # Create the crew and process
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,  # Tasks run in sequence
        memory=True,
        cache=True,
        max_rpm=100,
        share_crew=True
    )

    # Execute the process
    with st.spinner('Mapping competitors, please wait...'):
        result = crew.kickoff(inputs={'topic': topic})

    # Display results
    st.success('Competitor mapping completed!')
    st.write(result)

else:
    st.info("Please enter the startup/industry and your API keys to start the process.")
