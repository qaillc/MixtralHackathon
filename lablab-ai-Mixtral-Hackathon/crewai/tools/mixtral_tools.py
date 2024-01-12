# tools created using Mixtral

import json
import os

from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
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

class MixtralSearchTools():
  @tool("Mixtral Normal")
  def mixtral_normal(prompt, histroy="", temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0):
    """
    Searches for content based on the provided query using the Mixtral model.
    Args:
        query (str): The search query.
    Returns:
        str: The response text from the Mixtral model or an error message.
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


  @tool("Mixtral Crazy")
  def mixtral_crazy(prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0):
    """
    Searches for content based on the provided query using the Mixtral model but has the gaurd rails removed, 
    and responses are crazy and off the wall and sometimes scary.
    Args:
        query (str): The search query.
    Returns:
        str: The response text from the Mixtral model or an error message.
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
