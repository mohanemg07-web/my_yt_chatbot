# 🎬 YouTube AI Chatbot

> Ask any question about a YouTube video — powered by Retrieval-Augmented Generation (RAG)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## ✨ Features

- 🔗 **Paste any YouTube URL** — automatically extracts the transcript
- 🧠 **RAG Pipeline** — chunks the transcript, builds a FAISS vector index, and retrieves the most relevant context
- 💬 **AI-Powered Answers** — uses OpenAI GPT to generate grounded, cited answers
- 📊 **Video Metadata** — displays title, author, and chunk count
- 📚 **Context Transparency** — view the exact transcript chunks used to generate each answer
- 🎨 **Modern UI** — dark theme with gradient accents, progress indicators, and responsive layout

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
┌──────────────────┐
│  YoutubeLoader   │  ← Extract transcript via youtube-transcript-api
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Text Splitter   │  ← RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  FAISS Indexing   │  ← OpenAI text-embedding-3-large
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Retriever (k=4) │  ← Similarity search for top-4 relevant chunks
└────────┬─────────┘
         ▼
┌──────────────────┐
│  LLM (GPT-4o)    │  ← Generates answer from context + question
└────────┬─────────┘
         ▼
      Answer
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/youtube-ai-chatbot.git
cd youtube-ai-chatbot

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run youtube_chatbot.py
```

Enter your OpenAI API key in the sidebar and start chatting!

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set the main file to `youtube_chatbot.py`
4. Add your API key in **Settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy** — your app will be live at `https://your-app.streamlit.app`

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **LangChain** | RAG orchestration, prompt management, chain composition |
| **FAISS** | In-memory vector similarity search |
| **OpenAI** | Text embeddings (`text-embedding-3-large`) + LLM (`gpt-4o-mini`) |
| **Streamlit** | Web UI framework with reactive components |
| **youtube-transcript-api** | Transcript extraction from YouTube videos |

---

## 📁 Project Structure

```
youtube-ai-chatbot/
├── youtube_chatbot.py        # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── .streamlit/
│   ├── config.toml           # Streamlit theme configuration
│   └── secrets.toml.example  # Example secrets file
└── README.md                 # This file
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using LangChain & Streamlit
</p>
