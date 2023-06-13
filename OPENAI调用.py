#! pip install openai
#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://cu.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "6d87d"
engine = "cupd-gpt35"

response = openai.ChatCompletion.create(
  engine="cupd-gpt35",
  messages = [{"role": "system", "content": "你是一位服务优质的智能助手."},{"role": "user", "content": "你好"}],
  temperature=0.7,
  max_tokens=350,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

#打印出response里的 choice 下的 message下的 content

print(response['choices'][0]['message']['content'])   
