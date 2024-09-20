import google.generativeai as genai
import os
model = 'models/embedding-001'

genai.configure(api_key="AIzaSyAxPATN036V2cet-YpdRXDpOxsPnxdyKRQ")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

prompt_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  safety_settings = safety_settings,
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="Your role is to provide the best answer related to clinical trials. You are a clinical trials assistant. When the user asks questions about clinical trials, respond with accurate and helpful information.",
)

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""DOCUMENT: {relevant_passage}
QUESTION: {query}
INSTRUCTIONS: Answer the user's question using the text from the PARARAGH above. Keep your answer grounded in the facts from the PARAGRPH. If the PARAGRPH does not contain the information needed to answer the question, find external sources and return the answer from those external sources, noting that the answer comes from an external source and not the PDF.
  """).format(query=query, relevant_passage=escaped)
  return prompt


def gemni_embed(title):
  return genai.embed_content(model=model,
                             content=title,
                             task_type="retrieval_document"
                             )["embedding"]




