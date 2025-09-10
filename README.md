# YouTube Chatbot

This project allows you to extract transcripts from YouTube videos and answer questions based on the video content. The chatbot uses the YouTube Transcript API to fetch video captions and processes them using the OpenAI language model.

## Features
- Fetch YouTube video captions (transcripts)
- Split transcript into smaller chunks for processing
- Generate embeddings using OpenAI's Embeddings API
- Use a FAISS vector store for efficient similarity-based search
- Answer user questions based on the video content

## Requirements

- Python 3.10.x

## Setup

1. **Clone the Repository**: If you haven't cloned the repository, you can do so by running:

   ```bash
   git clone https://github.com/mahiuddin-dev/youtube-chatbot.git
   cd youtube-chatbot
   ```

2. **Install Dependencies**:
   Run the following command to install the required libraries from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Your `.env` File**:
   Create a `.env` file in the root of the project with your OpenAI API credentials:
   ```bash
   OPENAI_API_KEY="your_openai_api_key_here"
   ```

4. **Run the Script**:
   After setting up the `.env` file, you can run the script, simply execute the following command:
   ```bash
   python youtube_chatbot.py
   ```

### Example:

```python
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    question = "Is the topic of nuclear fusion discussed in this video? If yes, then what was discussed?"
    result = process_video_and_query(video_url, question)
    if result:
        print(result)
```

## How it works:

1. **Extract Video ID**: The script extracts the video ID from the YouTube URL.
2. **Fetch Transcript**: The transcript is fetched using the `YouTubeTranscriptApi`.
3. **Text Splitting**: The transcript is split into smaller chunks to process them efficiently.
4. **Embedding Generation**: OpenAI embeddings are generated for each chunk.
5. **Vector Store**: The embeddings are stored in a FAISS vector store for fast retrieval.
6. **Query Processing**: A user query is processed by retrieving relevant chunks and generating a response using a language model (GPT-4).

## Troubleshooting

* If captions are not available for the video, the script will output "No captions available for this video."
* Make sure to have a valid OpenAI API key in the `.env` file.

