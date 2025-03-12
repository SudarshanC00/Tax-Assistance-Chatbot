# Tax Assistant Chatbot

The **Tax Assistant Chatbot** is a web application designed to help users with tax-related queries. Powered by advanced natural language processing, it provides concise and precise responses to user questions about taxes. The chatbot is built using **Streamlit** for the frontend, **LangChain** for conversational AI, **Chroma** for vector database retrieval, and Google's **Gemini LLM** for generating responses.

---

## Features
- **Tax-Specific Expertise**: The chatbot specializes in answering tax-related questions.
- **Vector Search**: Uses a Chroma vector database for context-based document retrieval.
- **Interactive UI**: A modern, responsive chatbot interface with a dark theme.
- **Persistent Memory**: Maintains conversation history to provide context-aware answers.
- **Custom Prompts**: Tailored prompts to ensure accurate and focused responses.

---

## Prerequisites

### Software Requirements
- Python 3.9 or later
- Streamlit
- HuggingFace Transformers
- LangChain
- Chroma (VectorStore)

### API Requirements
- Google Generative AI API Key

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/tax-assistant-chatbot.git
   cd tax-assistant-chatbot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Key**:
   Replace `Your_API_KEY` in the code with your Google Generative AI API key.

4. **Set Up Vector Database**:
   Ensure that the `vector_db` directory exists in the project folder. This will store the Chroma vector database.

---

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Chatbot**:
   Open your web browser and navigate to `http://localhost:8501`.

3. **Ask Questions**:
   - Use the input box to type your tax-related queries.
   - The chatbot retrieves relevant context and provides concise answers.

---

## File Structure

```
Tax Assistant Chatbot
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── vector_db/            # Directory for Chroma vector database
└── README.md             # Project documentation
```

---

## Customization

### Modify the Prompt
- The chatbot's behavior is defined by a custom prompt template:
  ```python
  prompting = """You are a helpful and specialized tax assistant named Tax Assistant Bot...
  """
  ```
- Update the prompt text to change the chatbot's tone or focus.

### Change Styling
- Adjust the chatbot's appearance by modifying the CSS styles in the `st.markdown()` sections.

---

## Limitations
- This chatbot only answers **tax-related questions**. Non-tax queries may yield irrelevant or minimal responses.
- Requires a functional internet connection for the LLM API and vector retrieval.

---

## Future Enhancements
- Add support for multilingual queries.
- Integrate additional tax-related datasets for broader coverage.
- Deploy on a cloud platform like AWS or Google Cloud.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/)
- [Google Generative AI](https://cloud.google.com/vertex-ai/docs/generative-ai)

---

Feel free to fork, contribute, or reach out for suggestions and improvements!
# Tax-chatbot-flask
