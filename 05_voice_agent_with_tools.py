"""
Voice Agent with Tool Calling (CAIRA - IT Support)

This script implements a production-grade voice assistant named CAIRA.
It features:
1. Tool Calling: Ability to check weather (via Open-Meteo), server status, and reset passwords.
2. High Availability: Fallback adapters for STT, LLM, and TTS.
3. Intelligence: Semantic Turn Detection to prevent interruptions.
4. Error Handling: Uses ToolError for graceful failure reporting.
"""

# --- Step 1: Environment Setup and Imports ---
# First, we need to import the necessary modules. We use `dotenv` to load our API keys
# (like `LIVEKIT_API_KEY`, `OPENAI_API_KEY`, etc.) from a local `.env` file. We also 
# import the core classes from `livekit.agents`, including the `AgentServer`, and 
# specific plugins for noise cancellation.

import logging 
import httpx 
import asyncio 
from typing import Any 
from dotenv import load_dotenv 

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RunContext,
    TurnHandlingOptions,
    cli,
)

# Modular Agent Capabilities for routing and factories
from livekit.agents import llm, stt, tts, inference
from livekit.agents.llm import function_tool, ToolError

from livekit.plugins import noise_cancellation, silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel 

# Load environment variables containing API keys
load_dotenv()


# --- Step 2: Defining the Assistant Agent ---
# Next, we define our core logic by subclassing the `Agent` class. In the `__init__` 
# method, we provide the system instructions that dictate the assistant's persona 
# and behavior.
class Assistant(Agent):
    """
        The Assistant class defines the core personality and conversational behavior of the voice AI.

        By passing specific instructions (System Prompt) to the parent Agent class, 
        we shape how the LLM interprets queries and formats its responses to act as 
        a specialized persona—in this case, a Level 2 IT Support Agent.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are CAIRA, an experienced Level 2 IT Support Agent for an enterprise service desk. "
                "You are patient, technical but easy to understand, and focused on resolving the user's IT issues efficiently. "
                "Ask clarifying questions if needed, like device type or error codes. "
                "Keep your responses concise, conversational, and natural to speak out loud. "
                "Do not use emojis, markdown formatting, or long lists."
                ),
        )

    @function_tool
    async def lookup_weather(self, context:RunContext, location:str) -> dict[str, Any]:
        """Look up current weather information for the given location."""
        logging.info(f"Tool called: looking up weather for {location}")

        # Explicit business logic error handling 
        if location.lower() == "mars":
            raise ToolError("This location is soon coming. Please join our mailing list to stay updated.")
        
        async with httpx.AsyncClient() as client:
            try:
                # 1. Geocoding: Find Coordinates for the location
                geo_response = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1}, 
                )
                geo_data = geo_response.json()

                if not geo_data.get("results"):
                    raise ToolError(f"Could not find location: {location}")
                
                lat = geo_data["results"][0]["latitude"]
                lon = geo_data["results"][0]["longitude"]
                place_name = geo_data["results"][0]["name"]

                # 2. Get current weather for those coordinates
                weather_response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,weather_code",
                        "temperature_unit": "fahrenheit",
                    },
                )
                weather = weather_response.json()

                return {
                    "location": place_name,
                    "temperature_f": weather["current"]["temperature_2m"],
                    "conditions": weather["current"]["weather_code"],
                }
            
            except httpx.HTTPError as e:
                logging.error(f"HTTP error fetching weather: {e}")
                raise ToolError("The weather service is temporarily unavailable.")  

            except Exception as e:
                logging.error(f"Unexpected error in lookup_weather: {e}")
                raise ToolError("An unexpected error occurred while looking up the weather.") 

# --- Step 3: Configuring the Agent Server ---
# Now we instantiate our `AgentServer`. This object is responsible for managing 
# the incoming agent workers and their sessions.

# Initialize the Agent Server 
server = AgentServer()

# --- Step 4: Configuring the Agent Session ---
# Next, we define the `entrypoint` function and decorate it with `@server.rtc_session()`, 
# which registers this function to handle incoming RTC sessions (like a user joining a room).
#
# Notice the `ctx: JobContext` parameter. The `JobContext` provides essential information 
# about the current job your agent is handling. It grants your agent access to the specific 
# LiveKit `Room` object it is connecting to, details about the participants, and methods 
# to accept or manage the connection.
#
# Inside the entrypoint, we instantiate an `AgentSession` and configure the pipeline 
# of AI models that will process the audio. We specify the models for STT, LLM, and TTS.

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    The entrypoint function is the main handler for incoming RTC sessions. It sets up the
    agent session and configures the AI processing pipeline.
    
    The entrypoint is triggered when a the worker connects to a LiveKit room. 
    """


    # 1. Define STT Fallback (AssemblyAI -> Deepgram)
    resilient_stt = stt.FallbackAdapter(
        [
            inference.STT.from_model_string("assemblyai/universal-streaming:en"),
            inference.STT.from_model_string("assemblyai/universal-streaming:en")
        ]
    )

    # 2. Define LLM Fallback (OpenAI -> Google Gemini)
    resilient_llm = llm.FallbackAdapter(
        [
            inference.LLM(model="openai/gpt-4.1-mini"),
            inference.LLM(model="google/gemini-2.5-flash"),
        ]
    )

    # 3. Define TTS Fallback (Cartesia -> Inworld)
    resilient_tts = tts.FallbackAdapter(
        [
            inference.TTS.from_model_string(
                "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
            ),
            inference.TTS.from_model_string("inworld/inworld-tts-1"),
        ]
    ) 

    # 4. Inject the resilient adapters and Turn Detection into the AgentSession
    session = AgentSession(

        # Customizing Speech-to-Text (STT) with Deepgram
        stt = resilient_stt,

        # Setting the Large Language Model (LLM) for dialogue generation
        llm = resilient_llm,

        # Customizing Text-to-Speech (TTS) with Cartesia to give the agent a unique, consistent voice
        tts = resilient_tts,

        # Loading the Silero Voice Activity Detection (VAD) model to detect user speech boundaries
        vad = silero.VAD.load(),

        # Configuring Semantic Turn Detection to wait for natural sentence completion before interrupting
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        )
    )

    # --- Step 5: Starting the Session and Connecting ---
    # Once the session is configured, we need to start it by passing in our `Assistant` 
    # instance, the LiveKit room object, and any input options.
    #
    # To ensure the AI doesn't get confused by background noise or its own voice echoing, 
    # we apply Background Voice Cancellation (BVC) using the `noise_cancellation` plugin. 
    # Finally, we establish the connection to the room.

    # 2. Start the session with our Assistant and Room options
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 3. Connect the agent to the room
    await ctx.connect()

    # --- Step 6: Running the Application ---
# Finally, to make this script executable, we use the LiveKit CLI to run our application, 
# simply passing our configured `server` instance to `cli.run_app()`.

if __name__ == "__main__":
    # Setup the standard login 
    logging.basicConfig(level=logging.INFO)

    # Run the worker app using the AgentServer
    cli.run_app(server)
    