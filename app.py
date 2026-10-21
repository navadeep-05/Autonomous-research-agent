import os
import streamlit as st

from agent.graph import claim_status, source_kind, stream_graph
from RAG.loader import DATA_DIR
from RAG.retriever import build_index, index_size

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown("""
<style>
    /* Always display the primary Run Research button in vibrant red */
    div.stButton > button,
    div.stButton > button:disabled,
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(255, 75, 75, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        background-color: #e03b3b !important;
        border-color: #e03b3b !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        background-color: #c72c2c !important;
        border-color: #c72c2c !important;
        transform: translateY(0) !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

NODE_LABELS = {
    "planner": "🧭 Planning sub-tasks with Groq LLM",
    "researcher": "🌐 Executing Web Search (Tavily) & 📚 Local RAG Retrieval",
    "gap_detector": "🕳️ Auditing evidence & detecting research gaps",
    "gap_researcher": "🔁 Conducting targeted re-search on identified gaps",
    "source_recorder": "🏷️ Deduplicating sources & assigning SRC-xxx identifiers",
    "verifier": "✅ Ground-truth auditing: classifying claim validity",
    "citation_tracker": "🔗 Computing semantic vector embeddings for citation matching",
    "reporter": "📝 Synthesizing grounded final report with inline citations",
}

ICON = {
    "SUPPORTED": "✅",
    "PARTIAL": "⚠️",
    "UNSUPPORTED": "❌",
    "UNKNOWN": "❔"
}

# ---------- sidebar: local knowledge base ----------
os.makedirs(DATA_DIR, exist_ok=True)
with st.sidebar:
    st.header("📚 Local Knowledge Base")
    st.caption("Documents in `data/` are parsed, chunked, and embedded into local vector storage.")
    
    uploads = st.file_uploader(
        "Upload reference files (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    if uploads:
        for f in uploads:
            dest = os.path.join(DATA_DIR, os.path.basename(f.name))
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s).")
    
    col_btn, col_info = st.columns([1, 1])
    with col_btn:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Indexing chunks..."):
                n = build_index()
            st.success(f"Indexed {n} chunks")
            st.rerun()

    files = [n for n in os.listdir(DATA_DIR) if n.lower().endswith((".pdf", ".txt", ".md"))]
    st.markdown(f"**Index Status**: `{index_size()} chunks` from `{len(files)} file(s)`")
    if files:
        with st.expander("📁 View indexed files", expanded=False):
            for fname in files:
                st.write(f"• {fname}")

    st.markdown("---")
    st.markdown("### ⚙️ Guardrails & Constraints")
    st.markdown("""
    - **Model**: `openai/gpt-oss-20b`
    - **Max Tasks**: `2`
    - **Max Gap Rounds**: `1`
    - **Dense Embedding**: `all-MiniLM-L6-v2`
    - **Local Similarity Cutoff**: `0.25`
    - **Citation Alignment Cutoff**: `0.30`
    """)

# ---------- main app interface ----------
st.title("🔎 Autonomous Research Agent")
st.caption("Deterministic Multi-Agent Coordination via LangGraph: Dual Web/Local Retrieval · Gap Analysis · Semantic Verification")

question = st.text_area(
    "Research Question",
    placeholder="e.g. What are the key architectural differences between LangGraph and traditional chain-based agents?",
    height=85
)

col_run, col_clear = st.columns([1, 6])
with col_run:
    run_clicked = st.button("🚀 Run Research", type="primary", use_container_width=True)

if run_clicked:
    if not question.strip():
        st.warning("⚠️ Please enter a research question first.")
    else:
        final = None
        try:
            with st.status("Executing LangGraph research pipeline...", expanded=True) as status:
                step_container = st.container()
                for node, final in stream_graph(question.strip()):
                    step_container.write(NODE_LABELS.get(node, f"⚙️ {node}"))
                status.update(label="Research complete and report generated!", state="complete", expanded=False)
            st.session_state["result"] = dict(final)
        except Exception as e:
            st.session_state["result"] = None
            st.error(f"Execution failed: {e}")

state = st.session_state.get("result")
if state:
    report = state.get("report", "")
    if isinstance(report, list):
        report = "\n\n".join(str(x) for x in report)
    sources = state.get("sources", [])
    cites = state.get("citations", [])
    by_id = {s["source_id"]: s for s in sources}
    n_local = sum(1 for s in sources if s.get("url", "").startswith("local://"))
    n_web = len(sources) - n_local

    # Calculate claim verification breakdown
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0, "UNKNOWN": 0}
    for c in cites:
        k = claim_status(c.get("claim", ""))
        counts[k] = counts.get(k, 0) + 1

    st.markdown("### 📊 Research Execution Summary")
    cols = st.columns(6)
    cols[0].metric("Tasks Planned", len(state.get("plan", [])))
    cols[1].metric("Web Sources", n_web)
    cols[2].metric("Local Documents", n_local)
    cols[3].metric("Evidence Extracted", len(state.get("evidence", [])))
    cols[4].metric("Supported Claims", counts["SUPPORTED"])
    cols[5].metric("Gap-Loop Rounds", state.get("iteration", 0))

    if state.get("gaps"):
        st.warning(f"⚠️ **{len(state['gaps'])} unaddressed gap(s)** remain after max iterations ({state.get('iteration', 0)}/2). See the 'Plan & Gaps' tab.")
    else:
        st.success("✅ **Research Thorough**: No remaining gaps identified after evaluation.")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📄 Research Report",
        "🔀 LangGraph Workflow",
        "🎯 Plan & Gaps",
        "💡 Evidence Items",
        "🌐 Verified Sources",
        "🔍 Claim Verification & Citations"
    ])
    
    with t1:
        st.markdown(report)
        st.download_button(
            "📥 Download Research Report (.md)",
            report,
            file_name="research_report.md",
            mime="text/markdown"
        )
        
    with t2:
        st.subheader("🔀 Autonomous LangGraph Orchestration Workflow")
        st.caption("Sequential state transition trace executed autonomously by the research agent from start to finish:")

        st.markdown("""
```mermaid
graph TD
    UQ["1. User Question"] --> RP["2. Research Planner"]
    RP --> WS["3a. Web Search (Tavily)"]
    RP --> LR["3b. Local RAG (all-MiniLM-L6-v2)"]
    WS --> WE["Web Evidence"]
    LR --> LD["Local Documents"]
    WE --> EL["4. Evidence Layer"]
    LD --> EL
    EL --> SE["5. Source Evaluation"]
    SE --> GD["6. Gap Detection"]
    GD -->|"Gaps found & iterations < 2"| RS["7. Targeted Re-search"]
    RS -->|"Loop back"| GD
    GD -->|"No gaps OR max rounds (2)"| CV["8. Claim Verification"]
    CV --> CT["9. Semantic Citation Tracking"]
    CT --> FR["10. Final Cited Report (Downloadable)"]

    style UQ fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RP fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style WS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style LR fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style EL fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style SE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style GD fill:#fbe9e7,stroke:#d84315,stroke-width:2px
    style RS fill:#ffebee,stroke:#c62828,stroke-width:2px
    style CV fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style CT fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style FR fill:#e8eaf6,stroke:#283593,stroke-width:2px
```
""")

        st.markdown("#### 🔄 Sequential State Execution Log")
        wf_cols = st.columns(2)
        with wf_cols[0]:
            st.markdown(f"""
            - **1. User Question**: `{state.get('question', '')[:60]}...`
            - **2. Research Planner**: Generated `{len(state.get('plan', []))}` sub-tasks
            - **3. Dual Retrieval**:
              - Web Search: `{n_web}` sources
              - Local RAG: `{n_local}` passages
            - **4. Evidence Layer**: `{len(state.get('evidence', []))}` verified facts extracted
            - **5. Source Evaluation**: Evaluated quality, authority, recency
            """)
        with wf_cols[1]:
            st.markdown(f"""
            - **6. Gap Detection**: Identified completeness & missing context
            - **7. Re-Search Loop**: Executed `{state.get('iteration', 0)}` round(s) *(max 2)*
            - **8. Claim Verification**: `{len(cites)}` claims audited (`{counts['SUPPORTED']}` supported)
            - **9. Citation Tracking**: Dense vector cosine matching $\\ge 0.30$
            - **10. Final Cited Report**: Formatted markdown with references
            """)
        
    with t3:
        col_p, col_g = st.columns(2)
        with col_p:
            st.subheader("🧭 Research Plan")
            for i, t in enumerate(state.get("plan", []), 1):
                st.markdown(f"**{i}.** {t}")
        with col_g:
            st.subheader("🕳️ Identified Gaps")
            gaps = state.get("gaps", [])
            if gaps:
                for i, g in enumerate(gaps, 1):
                    st.markdown(f"**{i}.** {g}")
            else:
                st.info("No gaps detected. Evidence comprehensively answers the core question.")

    with t3:
        st.subheader(f"💡 Extracted Evidence ({len(state.get('evidence', []))} facts)")
        for i, ev in enumerate(state.get("evidence", []), 1):
            st.markdown(f"**{i}.** {ev}")

    with t4:
        st.subheader(f"📚 Sources Ledger ({len(sources)} unique references)")
        for s in sources:
            url = s.get("url", "")
            sid = s.get("source_id", "SRC")
            kind = source_kind(url)
            title = s.get("title") or url or "Untitled Document"
            
            if url.startswith("local://"):
                st.markdown(f"**[{sid}]** 📚 `{title}` — *{kind}*")
                with st.expander("View local excerpt", expanded=False):
                    st.write(s.get("content", ""))
            else:
                icon = "📄" if "pdf" in kind.lower() or "arxiv" in kind.lower() else "🌐"
                st.markdown(f"**[{sid}]** {icon} [{title}]({url}) — *{kind}*")
                with st.expander("View web snippet", expanded=False):
                    st.write(s.get("content", ""))

    with t5:
        st.subheader("🔍 Claim Verification & Semantic Source Alignment")
        st.caption("Each claim is categorized by truth groundness and matched to the closest source passage using SentenceTransformers cosine similarity.")
        
        for c in cites:
            claim_text = c.get("claim", "")
            sid = c.get("source_id")
            score = c.get("score", 0.0)
            status = claim_status(claim_text)
            
            src = by_id.get(sid)
            if src:
                url = src.get("url", "")
                title = src.get("title") or url
                ref = f"[{sid}: {title}]({url})" if not url.startswith("local://") else f"**{sid}** (`{title}`)"
            else:
                ref = "*(No source passage met $\\ge 0.30$ semantic threshold)*"
            
            st.markdown(f"{ICON.get(status, '❔')} **{claim_text}**")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **Source Match**: {ref} (Similarity: `{score:.2f}`)")
            st.markdown("---")