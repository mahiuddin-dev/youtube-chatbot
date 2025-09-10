import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# get video ID from the YouTube URL
def get_video_id(url: str) -> str:
    group=1
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    regex = re.compile(pattern)
    results = regex.search(url)
    if not results:
        return None
    return results.group(group)


def process_video_and_query(video_url: str, question: str):
    # Extract video ID
    video_id = get_video_id(video_url)
    if not video_id:
        print("Invalid video URL.")
        return

    # Transcript video text
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embedding Generation and Storing in Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retrieve relevant context for the user query
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Model
    model = ChatOpenAI(model="gpt-4o", temperature=0.2)

    # Prompt
    prompt = PromptTemplate(
        template="You are a helpful assistant. Answer ONLY the provided transcript context. If the context is insufficient, just say you don't know. \n\n {context}\n\n Question: {question}",
        input_variables=["context", "question"]
    )

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    result = model.invoke(final_prompt)
    return result.content


if __name__ == "__main__":
    # Example usage:
    # video_url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    video_url = "https://www.youtube.com/watch?v=ry9SYnV3svc"
    # question = "Is the topic of nuclear fusion discussed in this video? If yes, then what was discussed?"
    question = "Is the topic of boss is hilarious discussed in this video? If yes, then what was discussed?"
    result = process_video_and_query(video_url, question)
    if result:
        print(result)
