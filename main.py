import os
import json
import datetime
import webbrowser
from printing import dark_print
from dotenv import load_dotenv
from libs.deepseek_web_client import DeepSeekWebClient, ModelType

# Cargar las variables del archivo .env en el entorno
load_dotenv()

userToken = os.getenv('userToken') # Se puede obtener desde el LocalStorage.

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

dark_print("Starting client...")
client =  DeepSeekWebClient(bearer_token=userToken, verify_ssl=False)

dark_print("Creating a new chat...")
chat = client.create_chat()

dark_print("Sending first context prompt...")
ctx = open('context_prompt.txt', 'r').read()
response = json.loads(chat.send(ctx))
dark_print("Waiting for model acknowledgement for the system prompt.")
if response["messages"][0]["content"] != "OK":
    print("Model did not acknowledge the system prompt or failed to deliver json format.")
    exit()
dark_print("Chat ready!")

while True:
    usermsg = input("> ")
    formattedChat = {"messages": [{"role": "user", "content": f"{usermsg}"}]}
    response = json.loads(chat.send(str(formattedChat)))
    tool_response = {"messages": []}
    for msg in response["messages"]:
        if (msg["role"] == "assistant"):
            print(msg["content"])
        elif (msg["role"] == "function"):
            fn_name = msg["name"]
            fn_args = msg["arguments"]
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
                    print(f"Error on tool: {str(e)}...")
            tool_response["messages"].append({"role": "function", "name": fn_name, "arguments": fn_args, "result": tool_result})
    if (len(tool_response["messages"]) > 0):
        dark_print("Sending tool results...")
        response = json.loads(chat.send(str(tool_response)))
        for msg in response["messages"]:
            if (msg["role"] == "assistant"):
                print(msg["content"])
