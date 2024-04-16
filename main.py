import streamlit as st
import os
import dotenv
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import PyPDF2
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

api_key=os.environ.get("OPENAI_API_KEY")

st.title('Resume Analyzer')

case = st.file_uploader('File uploader',type="pdf")

if case is not None:
  # Read the file content as bytes
  bytes_data = case.read()

  # Process the bytes using PyPDF2
  with open("temp_file.pdf", "wb") as f:
      f.write(bytes_data)

  # Now you can use PyPDF2 with "temp_file.pdf"
  with open("temp_file.pdf", "rb") as f:
      pdf_reader = PyPDF2.PdfReader(f)

      text = ''
      for i, page in enumerate(pdf_reader.pages):
          content = page.extract_text()
          if content:
              text += content

      docs = [Document(page_content=text)]

      llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

      template = '''Write a concise and detailed summary of the following resume.
      Speech: `{text}`
      '''
      prompt = PromptTemplate(
          input_variables=['text'],
          template=template
      )

      chain = load_summarize_chain(
          llm,
          chain_type='stuff',
          prompt=prompt,
          verbose=False
      )
      output_summary = chain.run(docs)

      st.write(output_summary)

  # Clean up (optional)
  # os.remove("temp_file.pdf")
