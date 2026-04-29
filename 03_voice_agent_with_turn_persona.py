"""
Building a Voice AI Assistant with LiveKit

In this tutorial, we will walk through the process of building a real-time voice AI assistant using the LiveKit Agents framework. Our assistant will listen to user input, process it using a Large Language Model (LLM), and respond with synthesized speech.

We will be integrating several powerful AI models:
* STT (Speech-to-Text): Deepgram Nova-3
* LLM (Large Language Model): OpenAI GPT-4.1-mini
* TTS (Text-to-Speech): Cartesia Sonic-2
"""

# --- Step 1: Environment Setup and Imports ---
# First, we need to import the necessary modules. We use `dotenv` to load our API keys
# (like `LIVEKIT_API_KEY`, `OPENAI_API_KEY`, etc.) from a local `.env` file. We also 
# import the core classes from `livekit.agents`, including the `AgentServer`, and 
# specific plugins for noise cancellation.

import logging 
from dotenv import load_dotenv 

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    TurnHandlingOptions,
    cli,
)

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

    # 1. Configure the AI Pipeline Models
    session = AgentSession(

        # Customizing Speech-to-Text (STT) with Deepgram
        stt = "deepgram/nova-3",

        # Setting the Large Language Model (LLM) for dialogue generation
        llm = "openai/gpt-4.1-mini",

        # Customizing Text-to-Speech (TTS) with Cartesia to give the agent a unique, consistent voice
        tts = "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",

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
    