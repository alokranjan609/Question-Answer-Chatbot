
# Question-Answer Chatbot with LangChain and Streamlit

This project is an end-to-end Question and Answer (QA) chatbot that accepts a PDF or paragraph, processes the input using LangChain, and allows users to ask questions based on the provided content. The model retrieves relevant sections and gives accurate answers accordingly.

## Features

- **PDF and Text Support**: Upload PDFs or paste paragraphs for analysis.
- **Interactive Q&A**: Ask questions based on the provided input and receive relevant answers.
- **Streamlit Web Interface**: User-friendly web application powered by Streamlit for easy interaction.

## Prerequisites

- **Python**: Version 3.10 or higher
- **Conda**: For environment management

## Setup Instructions

Follow these steps to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/alokranjan609/Question-Answer-Chatbot.git
cd Question-Answer-Chatbot
```

### 2. Create a Conda Environment

Create a Conda environment using Python 3.10:

```bash
conda create -p venv python==3.10
```

### 3. Activate the Environment

Activate the Conda environment:

```bash
conda activate ./venv
```

### 4. Install Required Dependencies

Install the necessary packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

Navigate to the folder containing the `end_to_end.py` file and run the Streamlit application:

```bash
python -m streamlit run end_to_end.py
```

### 6. Access the Application

The application will be running locally at:

```text
http://localhost:8501
```

## Usage

1. **Upload** a PDF file or **paste** a paragraph into the input area.
2. **Ask questions** about the provided content.
3. Get **relevant answers** based on the content you provided.

## Project Structure

- **`end_to_end.py`**: The main Streamlit app file for running the Q&A system.
- **`requirements.txt`**: Lists the dependencies required to run the application.
- **`chain.ipynb`**: Jupyter notebook for experimenting with the QA system.


