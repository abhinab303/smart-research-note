import uuid
import hashlib
from collections import defaultdict

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
# Helper functions
# -----------------------------
def stable_key(*parts):
    raw = "||".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def clean_text(x, fallback=""):
    if x is None:
        return fallback
    x = str(x).strip()
    return x if x else fallback


def normalize_metadata(title, link, year, subtopic):
    return {
        "title": clean_text(title, "Untitled Paper"),
        "link": clean_text(link, ""),
        "year": int(year),
        "subtopic": clean_text(subtopic, "Uncategorized"),
    }


def add_idea(idea, title, link, year, subtopic):
    idea = clean_text(idea)

    if not idea:
        raise ValueError("Idea phrase cannot be empty.")

    metadata = normalize_metadata(title, link, year, subtopic)
    embedding = model.encode([idea]).tolist()

    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[idea],
        embeddings=embedding,
        metadatas=[metadata],
    )


def update_idea(idea_id, idea, title, link, year, subtopic):
    idea = clean_text(idea)

    if not idea:
        raise ValueError("Idea phrase cannot be empty.")

    metadata = normalize_metadata(title, link, year, subtopic)
    embedding = model.encode([idea]).tolist()

    collection.update(
        ids=[idea_id],
        documents=[idea],
        embeddings=embedding,
        metadatas=[metadata],
    )


def delete_idea(idea_id):
    collection.delete(ids=[idea_id])


def get_all_ideas():
    if collection.count() == 0:
        return []

    data = collection.get(include=["documents", "metadatas"])

    rows = []
    for idea_id, doc, meta in zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    ):
        meta = meta or {}

        rows.append(
            {
                "id": idea_id,
                "idea": doc,
                "title": meta.get("title", "Untitled Paper"),
                "link": meta.get("link", ""),
                "year": int(meta.get("year", 2026)),
                "subtopic": meta.get("subtopic", "Uncategorized"),
            }
        )

    return rows


def group_ideas_by_subtopic_and_paper(rows):
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        subtopic = row["subtopic"] or "Uncategorized"

        paper_key = (
            row["title"] or "Untitled Paper",
            row["link"] or "",
            int(row["year"]),
            subtopic,
        )

        grouped[subtopic][paper_key].append(row)

    return grouped


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
            normalize_metadata(
                title=x["title"],
                link=x["link"],
                year=x["year"],
                subtopic=x["subtopic"],
            )
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

tab_search, tab_add, tab_browse = st.tabs(
    ["Search ideas", "Add idea", "Browse / edit papers"]
)


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
            meta = meta or {}

            with st.container(border=True):
                st.markdown(f"### {doc}")
                st.write(f"**Paper:** {meta.get('title', '')}")
                st.write(f"**Year:** {meta.get('year', '')}")
                st.write(f"**Subtopic:** {meta.get('subtopic', '')}")

                link = meta.get("link", "")
                if link:
                    st.write(f"**Link:** {link}")

                st.caption(f"Distance: {distance:.4f} — lower is more similar")


# -----------------------------
# Add idea tab
# -----------------------------
with tab_add:
    st.subheader("Add one idea phrase")

    title = st.text_input("Paper title", key="add_title")
    link = st.text_input("Paper link", key="add_link")
    year = st.number_input(
        "Year",
        min_value=1900,
        max_value=2100,
        value=2026,
        key="add_year",
    )
    subtopic = st.text_input(
        "Subtopic",
        placeholder="continual learning, XAI, robustness, etc.",
        key="add_subtopic",
    )

    idea = st.text_area(
        "Idea phrase",
        placeholder="Example: Auxiliary classes reduce shortcut sharing across tasks.",
        key="add_idea",
    )

    if st.button("Save idea", key="save_single_idea"):
        try:
            add_idea(
                idea=idea,
                title=title,
                link=link,
                year=year,
                subtopic=subtopic,
            )
            st.success("Idea saved and embedded locally.")
            st.info("You can now search for it semantically or browse it by subtopic.")
        except ValueError as e:
            st.error(str(e))


# -----------------------------
# Browse / edit tab
# -----------------------------
with tab_browse:
    st.subheader("Browse papers by subtopic")

    rows = get_all_ideas()

    if not rows:
        st.info("No ideas saved yet.")
    else:
        grouped = group_ideas_by_subtopic_and_paper(rows)

        st.write(f"Total ideas: **{len(rows)}**")
        st.write(f"Total subtopics: **{len(grouped)}**")

        for subtopic in sorted(grouped.keys()):
            papers = grouped[subtopic]

            with st.expander(
                f"{subtopic} — {len(papers)} paper(s)",
                expanded=True,
            ):
                for paper_key, paper_ideas in sorted(
                    papers.items(), key=lambda item: item[0][0].lower()
                ):
                    paper_title, paper_link, paper_year, paper_subtopic = paper_key
                    paper_box_key = stable_key(
                        "paper",
                        paper_title,
                        paper_link,
                        paper_year,
                        paper_subtopic,
                    )

                    # Only the paper title is visible at first.
                    # Clicking it opens all details.
                    with st.expander(paper_title, expanded=False):
                        st.write(f"**Year:** {paper_year}")
                        st.write(f"**Subtopic:** {paper_subtopic}")

                        if paper_link:
                            st.write(f"**Link:** {paper_link}")

                        st.write(f"**Ideas:** {len(paper_ideas)}")

                        # -----------------------------
                        # Add idea to this paper
                        # -----------------------------
                        with st.expander("Add new idea to this paper"):
                            with st.form(key=f"add_to_paper_{paper_box_key}"):
                                new_idea = st.text_area(
                                    "New idea phrase",
                                    placeholder="Write a new idea from this paper...",
                                    key=f"new_idea_{paper_box_key}",
                                )

                                submitted = st.form_submit_button(
                                    "Save new idea under this paper"
                                )

                                if submitted:
                                    try:
                                        add_idea(
                                            idea=new_idea,
                                            title=paper_title,
                                            link=paper_link,
                                            year=paper_year,
                                            subtopic=paper_subtopic,
                                        )
                                        st.success("New idea added to this paper.")
                                        st.rerun()
                                    except ValueError as e:
                                        st.error(str(e))

                        # -----------------------------
                        # Existing ideas
                        # -----------------------------
                        st.markdown("#### Existing ideas")

                        for idx, row in enumerate(paper_ideas, start=1):
                            idea_id = row["id"]
                            idea_key = stable_key("idea", idea_id)

                            preview = row["idea"][:90]
                            if len(row["idea"]) > 90:
                                preview += "..."

                            with st.expander(f"Idea {idx}: {preview}"):
                                with st.form(key=f"edit_form_{idea_key}"):
                                    edited_idea = st.text_area(
                                        "Idea text",
                                        value=row["idea"],
                                        key=f"edit_idea_{idea_key}",
                                    )

                                    c1, c2 = st.columns(2)

                                    with c1:
                                        edited_title = st.text_input(
                                            "Paper title",
                                            value=row["title"],
                                            key=f"edit_title_{idea_key}",
                                        )
                                        edited_year = st.number_input(
                                            "Year",
                                            min_value=1900,
                                            max_value=2100,
                                            value=int(row["year"]),
                                            key=f"edit_year_{idea_key}",
                                        )

                                    with c2:
                                        edited_subtopic = st.text_input(
                                            "Subtopic",
                                            value=row["subtopic"],
                                            key=f"edit_subtopic_{idea_key}",
                                        )
                                        edited_link = st.text_input(
                                            "Paper link",
                                            value=row["link"],
                                            key=f"edit_link_{idea_key}",
                                        )

                                    save_col, delete_col = st.columns(2)

                                    with save_col:
                                        save_changes = st.form_submit_button(
                                            "Save changes"
                                        )

                                    with delete_col:
                                        delete_this = st.form_submit_button(
                                            "Delete idea"
                                        )

                                    if save_changes:
                                        try:
                                            update_idea(
                                                idea_id=idea_id,
                                                idea=edited_idea,
                                                title=edited_title,
                                                link=edited_link,
                                                year=edited_year,
                                                subtopic=edited_subtopic,
                                            )
                                            st.success("Idea updated.")
                                            st.rerun()
                                        except ValueError as e:
                                            st.error(str(e))

                                    if delete_this:
                                        delete_idea(idea_id)
                                        st.warning("Idea deleted.")
                                        st.rerun()