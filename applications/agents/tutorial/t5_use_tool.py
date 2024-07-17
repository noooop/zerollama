import json
from zerollama.agents.use_tool.agent_use_tools import AgentUseTools


def get_current_temperature(location: str, unit: str) -> str:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units.
    """
    if "chicago" in location.lower():
        return json.dumps({"location": "Chicago", "temperature": "13", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "55", "unit": unit})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": "11", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!


model = "NousResearch/Hermes-2-Pro-Llama-3-8B"


for llm_config in [
    {"type": "zerollama", "model": model, "global_priority": False},
    {"type": "openai", "model": model, "base_url": 'http://localhost:8080/v1/', "global_priority": False},
    #{"type": "ollama", "model": model, "global_priority": False},

]:
    agent = AgentUseTools(
        system_message="You are a bot that responds to weather queries. You should reply with the unit used in the queried location.",
        tools=[get_current_temperature, get_current_wind_speed],
        llm_config=llm_config
    )

    messages = [
        {"role": "user", "content": "What is the current temperature of New York, San Francisco and Chicago?"}
    ]

    reply = agent.generate_reply(messages)

    print(reply)