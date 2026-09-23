import os
import json
import time
import datetime
import webbrowser
from printing import dark_print
from dotenv import load_dotenv
from libs.deepseek_web_client import DeepSeekWebClient, ModelType

# Cargar las variables del archivo .env en el entorno
load_dotenv()

userToken = os.getenv('userToken')  # Se puede obtener desde el LocalStorage.

dark_print("Setting up functions...")

def get_os_time():
    now = datetime.datetime.now()
    return str(now)

def open_url(url: str):
    webbrowser.open(url)
    return True

available_functions = {
    "get_os_time": get_os_time,
    "open_url": open_url
}

LOOP_STATES = {"think", "plan", "act", "review"}
TERMINAL_STATES = {"idle", "complete"}

dark_print("Starting client...")
client = DeepSeekWebClient(bearer_token=userToken, verify_ssl=False)

dark_print("Creating a new chat...")
chat = client.create_chat()

dark_print("Sending first context prompt...")
ctx = open('context_prompt.txt', 'r').read()
response = json.loads(chat.send(ctx))
dark_print("Waiting for model acknowledgement for the system prompt.")
if response.get("messages", [{}])[0].get("content") != "OK":
    print("Model did not acknowledge the system prompt or failed to deliver json format.")
    exit()
dark_print("Chat ready!")


def send_json(payload: dict) -> dict:
    """Sends a dict payload to the model as real JSON and parses the reply."""
    raw = chat.send(json.dumps(payload))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        dark_print(f"Model returned invalid JSON, ignoring turn: {raw!r}")
        return {"state": "idle", "messages": []}


def print_assistant_messages(messages):
    for msg in messages:
        if msg.get("role") == "assistant":
            print(msg.get("content", ""))


def run_function_calls(messages) -> dict:
    """Executes any function-role messages and returns a tool_response payload."""
    tool_response = {"messages": []}
    for msg in messages:
        if msg.get("role") != "function":
            continue
        fn_name = msg.get("name")
        fn_args = msg.get("arguments") or {}
        function_to_call = available_functions.get(fn_name)
        if not function_to_call:
            tool_result = {"error": f"Unknown tool: {fn_name}"}
            dark_print(f"Unknown tool: {fn_name}")
        else:
            dark_print(f"Running tool: {fn_name}...")
            try:
                tool_result = function_to_call(**fn_args)
            except Exception as e:
                tool_result = {"error": str(e)}
                dark_print(f"Error on tool: {str(e)}")
        tool_response["messages"].append({
            "role": "function",
            "name": fn_name,
            "arguments": fn_args,
            "result": tool_result
        })
    return tool_response


def send_and_process(payload: dict) -> str:
    """Sends a payload, prints assistant output, executes any function calls,
    feeds back tool results, and returns the resulting state."""
    response = send_json(payload)
    state = response.get("state", "idle")
    dark_print(f"[state: {state}]")
    print_assistant_messages(response.get("messages", []))

    tool_response = run_function_calls(response.get("messages", []))
    if tool_response["messages"]:
        dark_print("Sending tool results...")
        response2 = send_json(tool_response)
        state = response2.get("state", state)
        dark_print(f"[state: {state}]")
        print_assistant_messages(response2.get("messages", []))

    return state


current_state = "idle"

while True:
    if current_state in TERMINAL_STATES:
        usermsg = input("> ")
        formatted = {"messages": [{"role": "user", "content": usermsg}]}
        current_state = send_and_process(formatted)
    else:
        #input(f"[step '{current_state}' done — press Enter to continue] ")
        print(f"[step '{current_state}' done — sending ack] ")
        time.sleep(1)
        ack = {"messages": [{"role": "user", "content": "continue"}]}
        current_state = send_and_process(ack)
