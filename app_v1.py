import uuid

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load embedding model once
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(name="idea_phrases")
    return collection


model = load_model()
collection = load_chroma()


# -----------------------------
# Demo seed data
# -----------------------------
def seed_demo_data():
    if collection.count() > 0:
        return

    demo_ideas = [
        {
            "idea": "Auxiliary classes reduce shortcut sharing across tasks.",
            "title": "CMDA",
            "year": 2026,
            "subtopic": "continual learning",
            "link": "https://example.com/cmda",
        },
        {
            "idea": "Task-ID prediction improves when generated data separates task-specific shortcuts.",
            "title": "CMDA",
            "year": 2026,
            "subtopic": "continual learning",
            "link": "https://example.com/cmda",
        },
        {
            "idea": "Attribution sharing can be used as a proxy for robustness in visual classifiers.",
            "title": "CAS Robustness",
            "year": 2026,
            "subtopic": "adversarial robustness",
            "link": "https://example.com/cas",
        },
        {
            "idea": "Feature-map smoothing improves the stability of gradient-based saliency maps.",
            "title": "Feature Map Smoothing",
            "year": 2025,
            "subtopic": "explainable AI",
            "link": "https://example.com/smoothing",
        },
        {
            "idea": "Concept-based explanations can reveal spurious correlations such as background color.",
            "title": "Concept Debiasing",
            "year": 2025,
            "subtopic": "concept explanations",
            "link": "https://example.com/concepts",
        },
    ]

    texts = [x["idea"] for x in demo_ideas]
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=[str(uuid.uuid4()) for _ in demo_ideas],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "title": x["title"],
                "year": x["year"],
                "subtopic": x["subtopic"],
                "link": x["link"],
            }
            for x in demo_ideas
        ],
    )


seed_demo_data()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Research Notes", layout="wide")

st.title("Smart Research Notes")
st.caption("Local semantic search over research idea phrases.")

tab_search, tab_add = st.tabs(["Search ideas", "Add idea"])


# -----------------------------
# Search tab
# -----------------------------
with tab_search:
    st.subheader("Semantic search")

    query = st.text_input(
        "Search query",
        placeholder="Example: generated data that improves task separation",
    )

    n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if query:
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        st.write(f"Found {len(docs)} related ideas:")

        for doc, meta, distance in zip(docs, metas, distances):
            with st.container(border=True):
                st.markdown(f"### {doc}")
                st.write(f"**Paper:** {meta.get('title', '')}")
                st.write(f"**Year:** {meta.get('year', '')}")
                st.write(f"**Subtopic:** {meta.get('subtopic', '')}")
                st.write(f"**Link:** {meta.get('link', '')}")
                st.caption(f"Distance: {distance:.4f}")


# -----------------------------
# Add idea tab
# -----------------------------
with tab_add:
    st.subheader("Add one idea phrase")

    title = st.text_input("Paper title")
    link = st.text_input("Paper link")
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2026)
    subtopic = st.text_input("Subtopic", placeholder="continual learning, XAI, robustness, etc.")

    idea = st.text_area(
        "Idea phrase",
        placeholder="Example: Auxiliary classes reduce shortcut sharing across tasks.",
    )

    if st.button("Save idea"):
        if not idea.strip():
            st.error("Please enter an idea phrase.")
        else:
            embedding = model.encode([idea.strip()]).tolist()

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[idea.strip()],
                embeddings=embedding,
                metadatas=[
                    {
                        "title": title,
                        "year": int(year),
                        "subtopic": subtopic,
                        "link": link,
                    }
                ],
            )

            st.success("Idea saved and embedded locally.")
            st.write("You can now search for it semantically in the Search tab.")