# Smart Research Notes

A lightweight local research note system for storing papers and searchable idea phrases.

This app helps you build a small personal research “second brain.” You can add paper metadata, store idea phrases from each paper, search ideas semantically, and browse papers grouped by subtopic.

The app runs locally on your Mac and does not require any external API.

---

## Installation

### 1. Create and enter the project folder

```bash
mkdir smart-research-notes
cd smart-research-notes
```

Place your `app.py` file inside this folder.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install streamlit chromadb sentence-transformers
```

### 5. Test the installation

```bash
python -c "import streamlit, chromadb, sentence_transformers; print('Setup OK')"
```

If everything is installed correctly, you should see:

```bash
Setup OK
```

---

## Running the App

Make sure you are inside the project folder:

```bash
cd smart-research-notes
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the app:

```bash
streamlit run app.py
```

The app should open automatically in your browser. If it does not, open:

```text
http://localhost:8501
```

---

## Features

- Add paper metadata:
  - Paper title
  - Paper link
  - Year
  - Subtopic

- Add idea phrases for each paper

- Search ideas semantically using local embeddings

- Browse papers grouped by subtopic

- Click a paper title to view:
  - Paper details
  - Existing ideas
  - Add-new-idea form
  - Edit/delete controls

- Add a new idea to an existing paper with paper metadata auto-filled

- Edit or delete existing ideas

- Store all ideas and embeddings locally using ChromaDB

---

## Tech Stack

- Python
- Streamlit
- ChromaDB
- SentenceTransformers
- `sentence-transformers/all-MiniLM-L6-v2` embedding model

---

## Project Structure

```bash
smart-research-notes/
│
├── app.py
├── README.md
└── chroma_store/        # Automatically created after running the app
```

The `chroma_store/` folder stores your local vector database and embeddings. It is created automatically when you run the app for the first time.

---

## Usage

### Add an Idea

Open the **Add idea** tab and fill in:

- Paper title
- Paper link
- Year
- Subtopic
- Idea phrase

Then click **Save idea**.

The idea will be embedded locally and saved in ChromaDB.

### Search Ideas

Open the **Search ideas** tab and search using natural language.

Example searches:

```text
generated data for task separation
```

```text
robustness through attribution similarity
```

```text
concept explanations for spurious correlations
```

The search is semantic, so the exact words do not need to match.

### Browse Papers

Open the **Browse / edit papers** tab.

Papers are grouped by subtopic. Click a paper title to expand it and view paper details, existing ideas, add-new-idea form, and edit/delete controls.

### Add an Idea to an Existing Paper

In the **Browse / edit papers** tab:

1. Click a subtopic.
2. Click a paper title.
3. Open **Add new idea to this paper**.
4. Type the new idea.
5. Click **Save new idea under this paper**.

The app automatically uses the existing paper title, link, year, and subtopic.

---

## Using macOS Dictation

You can use macOS built-in Dictation with any text box in the app.

Click inside an idea text box and press:

```text
Fn / Globe key twice
```

or use:

```text
Edit → Start Dictation
```

---

## Resetting the Database

To delete all saved ideas and start fresh, stop the Streamlit app first.

Then run:

```bash
rm -rf chroma_store
```

Restart the app:

```bash
streamlit run app.py
```

---

## Troubleshooting

### `streamlit: command not found`

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then reinstall Streamlit:

```bash
pip install streamlit
```

### Package import error

Reinstall the dependencies:

```bash
pip install --upgrade streamlit chromadb sentence-transformers
```

### First run is slow

The first run may be slow because SentenceTransformers downloads the embedding model. Later runs should be faster.