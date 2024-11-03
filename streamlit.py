import sys
import os
import re
import io
import ast
import unicodedata
import streamlit as st
from fpdf import FPDF
from typing import Union, List, Optional, Dict
from googleapiclient.discovery import build
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from urllib.parse import urlparse, parse_qs
import tiktoken
from youtube import YoutubeLoader

# Constant Definitions
ALL_CHAPTERS = "All Chapters"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"


# 初始化 SQLite3 模块（如果有特殊需求）
__import__('sqlite3')
import sqlite3


# Remove and reload the sqlite3 module (based on original code requirements)
sys.modules['sqlite3'] = sys.modules.pop('sqlite3')


# ================ Utility Functions ================

def set_environment_keys(api_keys):
    for key, value in api_keys.items():
        os.environ[key] = value


def initialize_vector_store(video_url: str) -> Optional[Chroma]:
    """
    Initialize the vector store and content transcription based on the video
    """
    print(f"Loading transcript for video URL")

    transcript = load_transcript(video_url)
    print(f"Transcript loaded")

    if not transcript:
        print("Transcript is empty.")
        return None
    docs = chunk_transcript(transcript)
    if not docs:
        return None
    return Chroma.from_documents(docs, OpenAIEmbeddings())


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the video ID from the URL
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    elif parsed_url.hostname == 'youtube.googleapis.com':
        return parsed_url.path.lstrip('/v/')
    elif parsed_url.hostname == 'www.youtube-nocookie.com':
        return parsed_url.path.lstrip('/embed/')
    else:
        return None


def fetch_video_info(url: str, api_key: str) -> Dict[str, Union[str, List[str]]]:
    """
    Retrieve information about the YouTube video.
    """
    video_id = extract_video_id(url)
    print(f"Extracted video ID: {video_id}")  # Print the extracted video_id
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    youtube = build('youtube', 'v3', developerKey=api_key)
    response = youtube.videos().list(part="snippet", id=video_id).execute()

    items = response.get("items")
    if not items:
        raise ValueError("Video not found")

    snippet = items[0]["snippet"]
    return {
        'description': snippet.get('description', ''),
        'title': snippet.get('title', ''),
        'channelTitle': snippet.get('channelTitle', ''),
        'tags': snippet.get('tags', [])
    }


def load_transcript(video_url: str) -> Optional[str]:
    # # Retrieve google_api_key from st.session_state

    api_key = st.session_state.get('google_api_key', None)
    if not api_key:
        st.error("Google API Key is missing.")
        return None

    # Use the obtained API key to load the transcript of the YouTube video
    loader = YoutubeLoader.from_youtube_url(video_url, api_key=api_key, add_video_info=True)
    documents = loader.load()
    if not documents or not hasattr(documents[0], 'page_content') or not documents[0].page_content:
        return None
    return documents[0].page_content


def chunk_transcript(transcript: str) -> List[Document]:
    """
    Chunk the transcript content into segments.
    """
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=200)
    segmented_text = text_splitter.split_text(transcript)
    return [Document(page_content=text) for text in segmented_text]


def extract_chapters(description: str, video_url: str) -> List[str]:
    """
    Extract video chapters from the description or transcript.
    """
    # Try to extract chapters from the description
    output_dict = extract_chapters_from_description(description)
    chapters_from_description = output_dict.get('Chapters', [])

    # If extracted chapters from the description are less than expected, extract from transcript
    if not output_dict["Status"] or len(chapters_from_description) < 3:
        chapters_from_transcript = extract_chapters_from_transcript(video_url)
        # Merge chapters and remove duplicates
        chapters = list(set(chapters_from_description + chapters_from_transcript))
    else:
        chapters = chapters_from_description

    return chapters


def extract_chapters_from_description(description: str) -> Dict[str, Union[bool, List[str]]]:
    """
    Extract chapters from the video description.
    """
    import re

    # Regular expression to match timestamps and chapter titles
    chapter_regex = r'((?:\d{1,2}:)?\d{1,2}:\d{2})\s*-?\s*(.*)'
    matches = re.findall(chapter_regex, description)
    print(f"Number of matches found: {len(matches)}")

    chapters = [match[1].strip() for match in matches if match[1].strip()]

    if chapters:
        print("Chapters extracted successfully.")
        return {"Status": True, "Chapters": chapters}

    # If the regex fails to extract chapters, use LLM
    chapters_existence_schema = ResponseSchema(
        name="Status",
        description="Does the description contain chapters? Answer true or false."
    )
    chapters_schema = ResponseSchema(
        name="Chapters",
        description="Extract the list of chapters from the description in the format of a Python list."
    )
    response_schemas = [chapters_existence_schema, chapters_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = f"""
    Please extract the chapter titles from the following YouTube video description. Chapters are typically listed in the description and may include timestamps (e.g., '00:00', '0:00:00') and may be prefaced by headings like 'Chapters', 'Timestamps', etc.

    {format_instructions}

    Your task is to:

    1. Identify all chapter entries in the description, ignoring any non-chapter content such as ads.
    2. Extract the chapter titles without timestamps.
    3. Output a Python list of chapter titles.

    Example:

    Description:
    '''
    00:00 Start
    05:30 Chapter 1: Introduction
    12:45 Chapter 2: Main Content
    16:00 Conclusion
    '''

    Output:
    ["Start", "Chapter 1: Introduction", "Chapter 2: Main Content", "Conclusion"]

    Now, please process the following description:

    '''{description}'''
    """

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ.get(OPENAI_API_KEY_ENV, ''))
    response = llm([{"role": "system", "content": prompt_template}])

    try:
        return output_parser.parse(response.content)
    except Exception:
        return {"Status": False, "Chapters": []}


def extract_chapters_from_transcript(video_url: str) -> List[str]:
    # Call the load_transcript function, which retrieves the API key from st.session_state
    transcript = load_transcript(video_url)
    if not transcript:
        return []

    # Reduce chunk size and increase overlap
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=5000, chunk_overlap=1000)
    segmented_documents = text_splitter.split_text(transcript)
    docs = [Document(page_content=text) for text in segmented_documents]

    # GPT model
    prompt_template = """
    你是一位视频总结助手，现在需要从提供的转录内容中提取详细的章节标题。请按照以下要求：

    1. 请你阅读转录内容，识别其中的主要主题和次要主题。
    2. 请为每个主要主题创建一个章节标题，如果有次要主题，也请提取。
    3. 输出一个 Python 列表，包含所有章节标题，顺序与内容出现的顺序一致。

    示例输出：
    ["引言", "背景介绍", "方法论", "实验结果", "结论与展望"]

    现在，请处理以下转录内容：

    '''{text}'''
    """

    system_message = SystemMessagePromptTemplate.from_template(prompt_template)
    human_message = HumanMessagePromptTemplate.from_template("")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ.get(OPENAI_API_KEY_ENV, ''))

    chapters = []

    for doc in docs:
        response = llm(chat_prompt.format_messages(text=doc.page_content))
        try:
            extracted_chapters = ast.literal_eval(response.content)
            chapters.extend(extracted_chapters)
        except (SyntaxError, ValueError):
            continue

    # Remove duplicates while maintaining order
    unique_chapters = []
    seen = set()
    for chapter in chapters:
        if chapter not in seen:
            seen.add(chapter)
            unique_chapters.append(chapter)

    return unique_chapters


def generate_video_summary(model_name: str, chapter: str, vector_store: Chroma, summary_type: str) -> Dict[str, str]:
    """
    Generate a video summary.
    """
    if not vector_store:
        return {}
    if summary_type not in ["bullet points", "paragraph"]:
        raise ValueError("summary_type 必须是 'bullet points' 或 'paragraph'")

    system_template = f"""
    你需要根据提供的章节内容生成简洁明了的摘要。摘要类型为 {summary_type}，最多 100 字。

    ```{{context}}```
    """

    system_message = SystemMessagePromptTemplate.from_template(system_template)
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=os.environ.get(OPENAI_API_KEY_ENV, ''))
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={'prompt': chat_prompt}
    )

    summary = qa.run(chapter)
    return {'chapter': chapter, 'summary': summary}


def string_to_pdf_bytes(video_info: str, summary_text: str) -> bytes:
    """
    Convert string to PDF byte stream.
    """
    pdf_buffer = io.BytesIO()
    pdf = FPDF(format='A4')
    pdf.add_page()

    # Add fonts supporting Unicode
    # pdf.add_font("ArialUnicode", style="", fname="ARIALUNI.TTF", uni=True)
    # pdf.set_font("ArialUnicode", size=12)
    font_path = r"C:\Users\王公子\Desktop\py\SimHei.ttf"
    pdf.add_font('SimHei', '', font_path, uni=True)
    pdf.set_font('SimHei', '', 12)

    # Add video information
    pdf.cell(0, 10, "Video Information:", ln=True)
    pdf.multi_cell(0, 10, video_info)
    pdf.cell(0, 10, "-----", ln=True)

    # Add summary text
    pdf.multi_cell(0, 10, summary_text)

    # Save PDF content to BytesIO
    pdf.output(pdf_buffer)

    # Return PDF bytes
    return pdf_buffer.getvalue()


def display_summaries(summaries: List[Dict[str, str]]):
    """
    Display summaries in Streamlit.
    """
    for summary_dict in summaries:
        chapter_title = f"## Chapter: {summary_dict['chapter']}"
        st.markdown(chapter_title)

        if '\n- ' in summary_dict['summary']:
            # Handle bullet point formatting
            summary_text = summary_dict['summary'].replace('\n- ', '\n* ')
            st.markdown(summary_text)
        else:
            st.markdown(f"{summary_dict['summary']}")
        st.write("---")


# ================ Main Application Logic ================

def main():
    st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
    youtube_logo_url = "https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_(2017).svg"
    title_with_logo = f'<img src="{youtube_logo_url}" width="100" style="vertical-align: middle;"> <span style="font-size:50px; vertical-align: middle; font-weight: bold; margin-left: 10px;">YouTube Video Summarizer</span>'
    st.markdown(title_with_logo, unsafe_allow_html=True)

    # Initialize session state
    if 'summary' not in st.session_state:
        st.session_state['summary'] = []
    if 'chapters_list' not in st.session_state:
        st.session_state['chapters_list'] = []
    if 'video_info' not in st.session_state:
        st.session_state['video_info'] = {}
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    st.sidebar.header("User Input")

    # Input API keys and video URL
    openai_api_key = st.sidebar.text_input("Input OpenAI API Key", type="password")
    google_api_key = st.sidebar.text_input("Input Google API Key", type="password")
    video_url = st.sidebar.text_input("Input YouTube Video URL")

    if google_api_key:
        st.session_state['google_api_key'] = google_api_key
    # Check if all required fields are filled
    all_fields_filled = all([openai_api_key, google_api_key, video_url])

    if all_fields_filled:
        st.sidebar.success("All fields are completed")
        set_environment_keys({
            OPENAI_API_KEY_ENV: openai_api_key,
            GOOGLE_API_KEY_ENV: google_api_key
        })

    # Start button
    if st.sidebar.button("Start"):
        if not all_fields_filled:
            st.sidebar.error("Please fill out all fields")
            return

        with st.spinner("Retrieve video information..."):
            try:
                video_info = fetch_video_info(video_url, google_api_key)
                st.session_state['video_info'] = video_info
            except Exception as e:
                print(f"Error while fetching video info: {e}")
                st.error(f"Failed to retrieve video information: {e}")
                return

        with st.spinner("Extracting video chapters..."):
            try:
                chapters = extract_chapters(video_info.get('description', ''), video_url)
                if chapters:
                    st.session_state['chapters_list'] = [ALL_CHAPTERS] + chapters
                else:
                    st.warning("Failed to extract chapter information")
            except Exception as e:
                st.error(f"Chapter extraction failed: {e}")
                return

        with st.spinner("Initializing vector storage..."):
            try:
                vector_store = initialize_vector_store(video_url)
                if vector_store:
                    st.session_state['vector_store'] = vector_store
                else:
                    st.warning("Failed to initialize vector storage")
            except Exception as e:
                st.error(f"Vector storage initialization failed: {e}")
                return

    # Display video information
    if st.session_state['video_info']:
        st.header("Video Info")
        video_info = st.session_state['video_info']
        st.markdown(f"**Channel Title:** {video_info.get('channelTitle', 'N/A')}")
        st.markdown(f"**Video Title:** {video_info.get('title', 'N/A')}")
        st.markdown(f"**Tags:** {', '.join(video_info.get('tags', []))}")

    # Chapter selection and summary type selection
    if st.session_state['chapters_list']:
        st.header("Chapter Selection")
        selected_chapters = st.multiselect(
            "Select Chapter to Summarize",
            st.session_state['chapters_list'],
            default=[ALL_CHAPTERS]
        )
        is_valid_selection = not (ALL_CHAPTERS in selected_chapters and len(selected_chapters) > 1)
        if not is_valid_selection:
            st.warning("You cannot select both 'All Chapters' and specific chapters at the same time.")

        summary_type = st.selectbox("Select Summary Type", ["paragraph", "bullet points"])

        if st.button("Generate Summary") and is_valid_selection:
            st.session_state['summary'] = []
            with st.spinner("Generating Summary..."):
                try:
                    chapters_to_summarize = st.session_state[
                        'chapters_list'] if ALL_CHAPTERS in selected_chapters else selected_chapters
                    for chapter in chapters_to_summarize:
                        summary = generate_video_summary("gpt-3.5-turbo", chapter, st.session_state['vector_store'],
                                                         summary_type)
                        if summary:
                            st.session_state['summary'].append(summary)
                        else:
                            st.warning(f"Unable to generate summary for chapter '{chapter}'")
                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")

    # Display summary
    if st.session_state['summary']:
        st.header("Video Summary")
        display_summaries(st.session_state['summary'])

        # Prepare PDF content
        video_info_text = f"Channel Title: {st.session_state['video_info'].get('channelTitle', 'N/A')}\nVideo Title: {st.session_state['video_info'].get('title', 'N/A')}\nTags: {', '.join(st.session_state['video_info'].get('tags', []))}\n"
        full_summary_text = "\n\n".join(
            [f"Chapter: {s['chapter']}\n{s['summary']}" for s in st.session_state['summary']])

        # Generate PDF
        pdf_bytes = string_to_pdf_bytes(video_info_text, full_summary_text)

        st.download_button(
            label="Download Summary as PDF",
            data=pdf_bytes,
            file_name="video_summary.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
