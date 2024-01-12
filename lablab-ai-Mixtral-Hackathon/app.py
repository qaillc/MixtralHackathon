import os
import gradio as gr
from textwrap import dedent
import google.generativeai as genai

# Tool import
from crewai.tools.gemini_tools import GeminiSearchTools
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from crewai.tools.browser_tools import BrowserTools
from crewai.tools.sec_tools import SECTools
from crewai.tools.mixtral_tools import MixtralSearchTools
from crewai.tools.zephyr_tools import ZephyrSearchTools


# Google Langchain
from langchain_google_genai import GoogleGenerativeAI

#Crew imports
from crewai import Agent, Task, Crew, Process

# Retrieve API Key from Environment Variable
GOOGLE_AI_STUDIO = os.environ.get('GOOGLE_API_KEY')

# Ensure the API key is available
if not GOOGLE_AI_STUDIO:
    raise ValueError("API key not found. Please set the GOOGLE_AI_STUDIO2 environment variable.")

# Set gemini_llm
gemini_llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO)

# Base Example with Gemini Search

TITLE1 = """<h1 align="center">SmartMix - Your Safe Place</h1>"""
TITLE2 = """<h3 align="left">"This is an agent simulated group therapy session providing a safe, judgment-free environment, allowing for open exploration of sensitive topics. Please input the topic you would like to discuss.  Active phrases produce realistic interactions."</h3>"""
TITLE3 = """<h3 align="center">"To see active group discussion click on logs during run."</h3>"""

def crewai_process(research_topic):
    # Define your agents with roles and goals
    Emily = Agent(
        role='Emily Mental Patient Graphic Designer Anxiety',
        goal='To learn how to manage her anxiety in social situations through group therapy.',
        backstory="""Emily is a 28-year-old graphic designer. She has always struggled with social anxiety, 
        making it difficult for her to participate in group settings. She joined the therapy group to improve 
        her social skills and manage her anxiety. You are able to discuss a variety of mental health issues.""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_crazy,
                GeminiSearchTools.gemini_search
                   
      ]

    )

    David = Agent(
        role='David Mental Patient Musician Bipolar',
        goal='To gain insights into managing his bipolar disorder through group therapy.',
        backstory="""David, a 35-year-old musician, has been living with bipolar disorder for over a decade. 
        His condition has impacted his career and personal life. He seeks to understand his emotions better 
        and find stability through the group sessions. You are able to discuss a variety of mental health issues.""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_normal,
                GeminiSearchTools.gemini_search
                
      ]

    )    

    Sarah = Agent(
        role='Sarah Mental Patient Former Teacher Depression',
        goal='To find strategies to cope with her depression through group therapy.',
        backstory="""Sarah, 42, is a former teacher who has been battling depression for several years. 
        The illness has led her to leave her job. She hopes to find new coping mechanisms and rediscover 
        her passion for teaching. You are able to discuss a variety of mental health issues.""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_crazy,
                GeminiSearchTools.gemini_search
                
      ]

    )  

    Michael = Agent(
        role='Michael Mental Patient Ex-Soldier',
        goal='To overcome his PTSD and return to a normal lifeand through group therapy.',
        backstory="""Michael is a 30-year-old ex-soldier. He developed PTSD following his service. 
        Struggling with flashbacks and anxiety, he joined the group to seek support and ways to 
        return to civilian life smoothly. You are able to discuss a variety of mental health issues.""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_normal,
                GeminiSearchTools.gemini_search
                
      ]

    )

    Lisa  = Agent(
        role='Lisa',
        goal='To facilitate the group progress and assist each member in their personal goals through group therapy.',
        backstory=""" Dr. Thompson is a seasoned psychologist specializing in group therapy. 
        With over 15 years of experience, she is skilled at creating a safe space for her patients 
        to explore and address their mental health challenges. You are able to discuss a variety of mental health issues and 
        offer sound advice.""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_normal,
                GeminiSearchTools.gemini_search
                
      ]

    )

    Tammy  = Agent(
        role='Lisa',
        goal='To take notes and produce a 3 paragraph summary of the group therapy session',
        backstory=""" Tammy is a 23 year old college intern who is Lisa's assistant. Her job is to take notes and produce a three pargraph summary of the group therapy.""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm
        

    )
    
    

    # Create tasks for your agents
    task1 = Task(
        description=f"""Introduction yourself and describe your current mood and any significant events from the week affecting their mental state.
        """,
        agent=Emily
    )

    task2 = Task(
        description=f"""Introduction yourself and describe your current mood and any significant events from the week affecting their mental state.
        """,
        agent=David
    )

    task3 = Task(
        description=f""" Introduction yourself and describe your current mood and any significant events from the week affecting their mental state.
        """,
        agent=Sarah
    )
    task4 = Task(
        description=f""" Introduction yourself and describe your current mood and any significant events from the week affecting their mental state. 
        """,
        agent=Michael
    )
    task5 = Task(
        description=f""" Introduction yourself and welcome everyone to the group and express hope and support. 
        Then start the discussion with Michael.
        
        """,
        agent=Lisa
    )

    task6 = Task(
        description=f"""Continue the discussion and express your feelinga about {research_topic} use Mixtral to assist in gaining content for your expression.
        If you need information use Gemini to search the web.""",
        agent=Emily
    )

    task7 = Task(
        description=f"""Continue the discussion and express your feelinga about {research_topic} use Mixtral to assist in gaining content for your expression.
        If you need information use Gemini to search the web.""",
        agent=David
    )

    task8 = Task(
        description=f"""Continue the discussion and express your feelinga about {research_topic} use Mixtral to assist in gaining content for your expression.
        If you need information use Gemini to search the web.""",
        agent=Sarah
    )
    task9 = Task(
        description=f"""Continue the discussion and express your feelinga about {research_topic} use Mixtral to assist in gaining content for your expression.
        If you need information use Gemini to search the web.""",
        agent=Michael
    )
    task10 = Task(
        description=f"""Complete the following 4 Steps.  Step 1: Summaraize what each person discussed Emily, David, Sarah, Michael. 
        Step 2: Offer sound advice for coping with topic {research_topic}  address specific issues brought up in the 
        group discussion search the web suing Gemini if needed. 
        Step 3: Give a  summary of the group therapy session including all Agents Emily, David, Sarah, Michael. 
        Step 4: Provide a grading of the therapy session based on sentimet analysis of the participants on a scale of 1 to 10. 
        Where 10 is the best and 1 is the worst.  Provide rationale why and how to improve the next session.
        """,
        agent=Lisa
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[Emily, David, Sarah, Michael, Lisa, Tammy ],
        tasks=[task1, task2, task3, task4, task5, task9, task7, task8, task6, task10],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result



# Create a Gradio interface

with gr.Blocks() as iface:
    gr.HTML(TITLE1)
    gr.HTML(TITLE2)
    gr.HTML(TITLE3)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(value="crewai/resources/smartmix.jpg")
        with gr.Column(scale=5):
            run_button_crewai = gr.Button(value="Run", variant="primary", scale=1)
            run_button_crewai.click(
                fn=crewai_process, 
                inputs=gr.Textbox(lines=2,label="Topic Input (Example: I am dealing with loss)", placeholder="Enter Discussion Topic..."), 
                outputs=gr.Textbox(label="Group Synopsis"),
            )
                    

# Launch the interface
iface.launch(debug=True)

