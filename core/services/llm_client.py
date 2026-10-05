from django.conf import settings
from openai import OpenAI

client = OpenAI(
   api_key = settings.OPENROUTER_API_KEY,
   base_url = settings.OPENROUTER_URL,
   default_headers = {
       'X-title' : 'RAG Chatbot',
   },
)