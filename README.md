**sts** provides a boilerplate code for building a **Speech-to-Speech Assistant** using **LiveKit**, **Deepgram**, **OpenAI**, and **Pinecone**. 

The project enables Retrieval-Augmented Generation (RAG) to look up information efficiently. With minimal setup, you can have your STS Assistant running in minutes!

## Features

- **Speech-to-Speech Integration**: Seamless integration with LiveKit and Deepgram for real-time audio processing.
- **RAG Implementation**: Utilize Pinecone for efficient information retrieval.
- **Quick Setup**: Minimal configurations required to get started.

## Setup

1. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate  # For Linux/MacOS
   # OR
   venv\Scripts\activate  # For Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py console
   ```

4. **Configuration**:
   - Add your `.env` file with the required keys as mentioned in the `.env.example` file.
   - Configure your Pinecone index.

## Usage

- Once set up, the STS Assistant is ready to process speech-to-speech tasks with RAG for information lookup.
