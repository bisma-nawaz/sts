import os
import json
import logging
from dotenv import load_dotenv

import aiohttp
from livekit import agents
from livekit.agents.llm import function_tool
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext
from livekit.plugins import openai, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from openai import OpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# patch out any aiohttp proxy kwargs
_orig = aiohttp.ClientSession.__init__
def _patched(self, *a, **k):
    k.pop("proxy", None)
    return _orig(self, *a, **k)
aiohttp.ClientSession.__init__ = _patched

load_dotenv() 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speech-assistant")

# client setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
stt    = deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY"))
llm    = openai.LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
tts    = openai.TTS(instructions="You are a friendly voice assistant.", model="gpt-4o-mini-tts", api_key=os.getenv("OPENAI_API_KEY"))

# pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "your-index-name"
namespace = "your-namespace"
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings_model, namespace=namespace)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            # define your instructions for the agent here
            instructions="""
            You are a voice assistant. Speak naturally.
            You have a tool called 'lookup_info' that you can use to find information.
            When asked a question, use the lookup_info tool to search for an answer.
            """,
            stt=stt,
            llm=llm,
            tts=tts
        )

    # define the tool that the agent can use to search for information (based on RAG)
    @function_tool
    async def lookup_info(self, context: RunContext, query: str):
        await self.session.say("One moment please.")
        docs = retriever.invoke(query)
        if not docs:
            return None, "Sorry, I couldn't find an answer."
        ctx_text = "\n".join(d.page_content for d in docs)
        answer = client.responses.create(
            model="gpt-4o-mini",
            input=f"Answer using context:\n{ctx_text}\nQ: {query}"
        ).output_text.strip()
        return None, answer

# agent entrypoint and session setup
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection=EnglishModel()
    )
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )
    # generate a reply for the agent to say when they first join the room
    await session.generate_reply(instructions="Hi there! How can I help?")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))