# tools created using Zephyr

import json
import os

from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "HuggingFaceH4/zephyr-7b-beta"
)

# Helper Method

def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt


import requests
from langchain.tools import tool

history = ""

class ZephyrSearchTools():
  @tool("Zephyr Normal")
  def zephyr_normal(prompt, histroy="", temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0):
    """
    Searches for content based on the provided query using the Zephyr model.
    Args:
        query (str): The search query.
    Returns:
        str: The response text from the Zephyr model or an error message.
    """
    generate_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "seed": 42,
    }

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=True)
    output = ""
    for response in stream:
        output += response.token.text
        yield output
    return output


  @tool("Zephyrl Crazy")
  def zephyr_crazy(prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0):
    """
    Searches for content based on the provided query using the Zephyr model but has the gaurd rails removed, 
    and responses are crazy and off the wall and sometimes scary.
    Args:
        query (str): The search query.
    Returns:
        str: The response text from the Zephyr model or an error message.
    """
    generate_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "seed": 42,
    }

    stream = client.text_generation(prompt, **generate_kwargs, stream=True, details=True, return_full_text=True)
    output = ""
    for response in stream:
        output += response.token.text
        yield output
    return output




