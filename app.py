import streamlit as st
import validators
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.chains import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title='Langchain: Summarize text from YT or website')
st.subheader('Summarize URL')
with st.sidebar:
    groq_api_key=st.sidebar.text_input('Enter GROQ API key',type='password')
    
llm_model=ChatGroq(model='gemma2-9b-it', api_key=groq_api_key)
url=st.text_input('URL',label_visibility='collapsed')

if st.button('Summarize the context from YT or Website'):
    if not groq_api_key.strip() or not url.strip():
        st.error('Please Provide the information')
    elif not validators.url(url):
        st.error('please enter valid url')
    else:
        try:
            with st.spinner('waiting .........'):
                if 'youtube.com' in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    # st.write('Done with loading YT')
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False)
                    # st.write('Done with URL loading')
                data=loader.load()
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=1000)
                finding=text_splitter.split_documents(data)
                
                prompt_template='''Please summarize the below context with 300 words.
                context:{context}'''
                prompt=PromptTemplate(template=prompt_template,input_variables=['context'])
                
                final_prompt='''Provide the final summary of the entire context with the important points.
                Start it with what is it all about & then provide the summary in number points with example.
                context:{context}'''
                final_prompt_template=PromptTemplate(input_variables=['context'],template=final_prompt)
                
                chain=load_summarize_chain(llm=llm_model,
                           chain_type='map_reduce',
                           map_prompt=prompt,
                           combine_prompt=final_prompt_template,
                           combine_document_variable_name="context",
                           map_reduce_document_variable_name="context",
                           verbose=True)
                output_summary=chain.run(finding)
                st.success(output_summary)
        except Exception as e:
            st.exception(f'exception: {e}')
    


