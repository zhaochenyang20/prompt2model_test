import asyncio
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn
from zeno_build.models import lm_config
from zeno_build.models.providers.openai_utils import (
    generate_from_openai_chat_completion,
)


async def main():
    # Define the prompts
    prompt1 = "How old are you?"
    prompt2 = "What's your name?"

    # Define the model configuration
    model_config = lm_config.LMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
    )

    # Create chat turns for each prompt
    turn1 = ChatTurn(role="user", content=prompt1)
    turn2 = ChatTurn(role="user", content=prompt2)

    # Create full_contexts list with the chat messages
    full_contexts = [
        ChatMessages(messages=[turn1]),
        ChatMessages(messages=[turn2]),
    ]

    # Create a prompt template for generating responses
    prompt_template = ChatMessages(messages=[])

    # Set the generation parameters
    temperature = 0.8
    max_tokens = 50
    top_p = 0.9
    context_length = 2

    # Call the generation function
    responses = await generate_from_openai_chat_completion(
        full_contexts=full_contexts,
        prompt_template=prompt_template,
        model_config=model_config,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        context_length=context_length,
    )

    print(responses)


# Run the script within an event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
