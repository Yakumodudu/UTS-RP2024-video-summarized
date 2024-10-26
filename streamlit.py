
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


# 常量定义
ALL_CHAPTERS = "All Chapters"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# 初始化 SQLite3 模块（如果有特殊需求）
__import__('sqlite3')
import sqlite3

# 移除并重新加载 sqlite3 模块（根据原代码需求）
sys.modules['sqlite3'] = sys.modules.pop('sqlite3')

# ================= 工具函数 ===================


def set_environment_keys(api_keys):
    for key, value in api_keys.items():
        os.environ[key] = value


def initialize_vector_store(video_url: str) -> Optional[Chroma]:
    """
    初始化向量存储，和基于视频的转录内容。
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
    从URL中提取视频ID
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
    获取YouTube视频信息。
    """
    video_id = extract_video_id(url)
    print(f"Extracted video ID: {video_id}")  # 打印提取到的video_id
    if not video_id:
        raise ValueError("无效的 YouTube URL")

    youtube = build('youtube', 'v3', developerKey=api_key)
    response = youtube.videos().list(part="snippet", id=video_id).execute()

    items = response.get("items")
    if not items:
        raise ValueError("未找到视频")

    snippet = items[0]["snippet"]
    return {
        'description': snippet.get('description', ''),
        'title': snippet.get('title', ''),
        'channelTitle': snippet.get('channelTitle', ''),
        'tags': snippet.get('tags', [])
    }

def load_transcript(video_url: str) -> Optional[str]:
    """
    加载视频的转录内容。
    """
    loader = YoutubeLoader.from_youtube_url(video_url, api_key='AIzaSyC6J5sSZ0XkMfNSzUnnPlZw1DM-otpb88k', add_video_info=True)
    documents = loader.load()
    print(video_url)
    if not documents or not hasattr(documents[0], 'page_content') or not documents[0].page_content:
        return None
    return documents[0].page_content


def chunk_transcript(transcript: str) -> List[Document]:
    """
    将转录内容分块处理。
    """
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=200)
    segmented_text = text_splitter.split_text(transcript)
    return [Document(page_content=text) for text in segmented_text]


def extract_chapters(description: str, video_url: str) -> List[str]:
    """
    从描述或转录中提取视频章节。
    """
    # 尝试从描述中提取章节
    output_dict = extract_chapters_from_description(description)
    chapters_from_description = output_dict.get('Chapters', [])

    # 如果描述中提取的章节少于预期，则从转录中提取
    if not output_dict["Status"] or len(chapters_from_description) < 3:
        chapters_from_transcript = extract_chapters_from_transcript(video_url)
        # 合并章节，去重
        chapters = list(set(chapters_from_description + chapters_from_transcript))
    else:
        chapters = chapters_from_description

    return chapters

def extract_chapters_from_description(description: str) -> Dict[str, Union[bool, List[str]]]:
    """
    从视频描述中提取章节。
    """
    import re

    # 匹配时间戳和章节标题的正则表达式
    chapter_regex = r'((?:\d{1,2}:)?\d{1,2}:\d{2})\s*-?\s*(.*)'
    matches = re.findall(chapter_regex, description)
    print(f"Number of matches found: {len(matches)}")

    chapters = [match[1].strip() for match in matches if match[1].strip()]

    if chapters:
        print("Chapters extracted successfully.")
        return {"Status": True, "Chapters": chapters}

    # 如果正则表达式未能提取章节，则使用 LLM
    chapters_existence_schema = ResponseSchema(
        name="Status",
        description="描述中是否包含章节，回答 true 或 false。"
    )
    chapters_schema = ResponseSchema(
        name="Chapters",
        description="从描述中提取章节列表，格式为 Python 列表。"
    )
    response_schemas = [chapters_existence_schema, chapters_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = f"""
    请你从以下 YouTube 视频描述中提取视频的小章节标题。小章节通常列在描述中，且有可能带有时间戳（例如 '00:00'、'0:00:00' 等），并可能以 'Chapters'、'Timestamps' 等标题开头。

    {format_instructions}

    你的任务是：

    1. 查找描述中所有的章节条目，并忽略任何广告等非章节内容。
    2. 提取章节标题，不包括时间戳。
    3. 输出章节标题的 Python 列表。

    示例：

    描述：
    '''
    00:00 开始
    05:30 第一章：引言
    12:45 第二章：主要内容
    16:00 结束语
    '''

    输出：
    ["开始", "第一章：引言", "第二章：主要内容", "结束语"]

    现在，请处理以下描述：

    '''{description}'''
    """

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ.get(OPENAI_API_KEY_ENV, ''))
    response = llm([{"role": "system", "content": prompt_template}])

    try:
        return output_parser.parse(response.content)
    except Exception:
        return {"Status": False, "Chapters": []}


def extract_chapters_from_transcript(video_url: str) -> List[str]:
    """
    从转录内容中提取章节。
    """
    transcript = load_transcript(video_url)
    if not transcript:
        return []

    # 减小分块大小，增加覆盖
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=5000, chunk_overlap=1000)
    segmented_documents = text_splitter.split_text(transcript)
    docs = [Document(page_content=text) for text in segmented_documents]

    # gpt的具体模板
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

    # 去重并保持顺序
    unique_chapters = []
    seen = set()
    for chapter in chapters:
        if chapter not in seen:
            seen.add(chapter)
            unique_chapters.append(chapter)

    return unique_chapters


def generate_video_summary(model_name: str, chapter: str, vector_store: Chroma, summary_type: str) -> Dict[str, str]:
    """
    生成视频摘要。
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
    将字符串转换为 PDF 字节流。
    """
    pdf_buffer = io.BytesIO()
    pdf = FPDF(format='A4')
    pdf.add_page()

    # 添加支持 Unicode 的字体
    # pdf.add_font("ArialUnicode", style="", fname="ARIALUNI.TTF", uni=True)
    # pdf.set_font("ArialUnicode", size=12)
    font_path = r"C:\Users\王公子\Desktop\py\SimHei.ttf"
    pdf.add_font('SimHei', '', font_path, uni=True)
    pdf.set_font('SimHei', '', 12)

    # 添加视频信息
    pdf.cell(0, 10, "Video Information:", ln=True)
    pdf.multi_cell(0, 10, video_info)
    pdf.cell(0, 10, "-----", ln=True)

    # 添加摘要文本
    pdf.multi_cell(0, 10, summary_text)

    # 保存 PDF 内容到 BytesIO
    pdf.output(pdf_buffer)

    # 返回 PDF 字节
    return pdf_buffer.getvalue()


def display_summaries(summaries: List[Dict[str, str]]):
    """
    在 Streamlit 中显示摘要。
    """
    for summary_dict in summaries:
        chapter_title = f"## 章节: {summary_dict['chapter']}"
        st.markdown(chapter_title)

        if '\n- ' in summary_dict['summary']:
            # 处理子弹点格式
            summary_text = summary_dict['summary'].replace('\n- ', '\n* ')
            st.markdown(summary_text)
        else:
            st.markdown(f"{summary_dict['summary']}")
        st.write("---")


# =================== 主应用逻辑 ===================

def main():
    st.set_page_config(page_title="YouTube 视频摘要生成器", layout="wide")
    youtube_logo_url = "https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_(2017).svg"
    title_with_logo = f'<img src="{youtube_logo_url}" width="100" style="vertical-align: middle;"> <span style="font-size:50px; vertical-align: middle; font-weight: bold; margin-left: 10px;">视频摘要生成器</span>'
    st.markdown(title_with_logo, unsafe_allow_html=True)

    # 初始化会话状态
    if 'summary' not in st.session_state:
        st.session_state['summary'] = []
    if 'chapters_list' not in st.session_state:
        st.session_state['chapters_list'] = []
    if 'video_info' not in st.session_state:
        st.session_state['video_info'] = {}
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    st.sidebar.header("用户输入")

    # 输入API密钥和视频URL
    openai_api_key = st.sidebar.text_input("输入 OpenAI API 密钥", type="password")
    google_api_key = st.sidebar.text_input("输入 Google API 密钥", type="password")
    video_url = st.sidebar.text_input("输入 YouTube 视频 URL")

    # 检查是否填写所有部分
    all_fields_filled = all([openai_api_key, google_api_key, video_url])

    if all_fields_filled:
        st.sidebar.success("所有字段已填写")
        set_environment_keys({
            OPENAI_API_KEY_ENV: openai_api_key,
            GOOGLE_API_KEY_ENV: google_api_key
        })

    # 开始按钮
    if st.sidebar.button("开始"):
        if not all_fields_filled:
            st.sidebar.error("请填写所有字段")
            return

        with st.spinner("获取视频信息..."):
            try:
                video_info = fetch_video_info(video_url, google_api_key)
                st.session_state['video_info'] = video_info
            except Exception as e:
                print(f"Error while fetching video info: {e}")
                st.error(f"获取视频信息失败: {e}")
                return

        with st.spinner("提取视频章节..."):
            try:
                chapters = extract_chapters(video_info.get('description', ''), video_url)
                if chapters:
                    st.session_state['chapters_list'] = [ALL_CHAPTERS] + chapters
                else:
                    st.warning("未能提取到章节信息")
            except Exception as e:
                st.error(f"提取章节失败: {e}")
                return

        with st.spinner("初始化向量存储..."):
            try:
                vector_store = initialize_vector_store(video_url)
                if vector_store:
                    st.session_state['vector_store'] = vector_store
                else:
                    st.warning("未能初始化向量存储")
            except Exception as e:
                st.error(f"初始化向量存储失败: {e}")
                return

    # 显示视频信息
    if st.session_state['video_info']:
        st.header("视频信息")
        video_info = st.session_state['video_info']
        st.markdown(f"**频道标题:** {video_info.get('channelTitle', 'N/A')}")
        st.markdown(f"**视频标题:** {video_info.get('title', 'N/A')}")
        st.markdown(f"**标签:** {', '.join(video_info.get('tags', []))}")

    # 章节选择和摘要类型选择
    if st.session_state['chapters_list']:
        st.header("章节选择")
        selected_chapters = st.multiselect(
            "选择要摘要的章节",
            st.session_state['chapters_list'],
            default=[ALL_CHAPTERS]
        )
        is_valid_selection = not (ALL_CHAPTERS in selected_chapters and len(selected_chapters) > 1)
        if not is_valid_selection:
            st.warning("不能同时选择 'All Chapters' 和具体章节")

        summary_type = st.selectbox("选择摘要类型", ["paragraph", "bullet points"])

        if st.button("生成摘要") and is_valid_selection:
            st.session_state['summary'] = []
            with st.spinner("生成摘要中..."):
                try:
                    chapters_to_summarize = st.session_state[
                        'chapters_list'] if ALL_CHAPTERS in selected_chapters else selected_chapters
                    for chapter in chapters_to_summarize:
                        summary = generate_video_summary("gpt-3.5-turbo", chapter, st.session_state['vector_store'],
                                                         summary_type)
                        if summary:
                            st.session_state['summary'].append(summary)
                        else:
                            st.warning(f"无法生成章节 '{chapter}' 的摘要")
                except Exception as e:
                    st.error(f"生成摘要失败: {e}")

    # 显示摘要
    if st.session_state['summary']:
        st.header("视频摘要")
        display_summaries(st.session_state['summary'])

        # 准备 PDF 内容
        video_info_text = f"频道标题: {st.session_state['video_info'].get('channelTitle', 'N/A')}\n视频标题: {st.session_state['video_info'].get('title', 'N/A')}\n标签: {', '.join(st.session_state['video_info'].get('tags', []))}\n"
        full_summary_text = "\n\n".join([f"章节: {s['chapter']}\n{s['summary']}" for s in st.session_state['summary']])

        # 生成 PDF
        pdf_bytes = string_to_pdf_bytes(video_info_text, full_summary_text)

        st.download_button(
            label="下载摘要为 PDF",
            data=pdf_bytes,
            file_name="video_summary.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
