from agents import Agents, Runner, OpenAIChatcompletionsModel,RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio 

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

external_client = AsyncOpenAI(api_key = gemini_api_key, base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",)

model = OpenAIChatcompletionsModel(model = "gemini-2.0-flash", openai_client = external_client)

config = RunConfig(model = model, model_provider = external_client, tracing_disabled = True)

async def main():
    agent = Agents(
    name = "Translator Agent",
    instructions = "You are a translator agent. You are given a text and you need to translate it to the target language.",
    )

    response = await Runner.run(
        agent,
        input = "Assalam-O-Alaikom, I am Salman and I am a beginner in Agentic AI.",
        run_config = config
    )

    print(response)


asyncio.run(main())




