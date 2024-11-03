YouTube Video Summarizer

API Requirements: Google API Key, OpenAI API Key

Supervisor: Prof. Wei Liu

What it is

The YouTube Video Summarizer is a tool designed to generate concise summaries for YouTube videos that include timestamps in their descriptions. It is primarily intended for educational purposes, assisting users in quickly understanding the main content of lengthy videos. The tool extracts video information, chapters, and generates summaries using Google and GPT APIs.

How it works

Video Metadata Retrieval: Using the Google API, the tool retrieves video metadata, including the title, channel name, and tags.

Chapter Extraction: For videos with timestamped chapters, the tool extracts chapter names and descriptions to create segment-specific summaries.

Summary Generation: GPT API is used to summarize each chapter, providing a condensed version of the video’s key points.

Output: Summaries are displayed in the Streamlit interface, and users can download a PDF version of the summary.

Limitations

Requires valid Google and OpenAI API keys.

Can only process YouTube videos that include timestamps in the description for chapter extraction.

Limited to videos in English, as the GPT model may not support other languages effectively.

Performance and response time may vary based on the length of the video and the number of chapters.
