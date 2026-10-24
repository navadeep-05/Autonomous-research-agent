import os
import re
import streamlit as st

from agent.graph import MAX_GAP_ROUNDS, MAX_TASKS, claim_status, source_kind, stream_graph
from RAG.loader import DATA_DIR
from RAG.retriever import build_index

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
    "UNKNOWN": "❔",
    "UNVERIFIED": "⚪",
}

# ---------- sidebar: local knowledge base ----------
os.makedirs(DATA_DIR, exist_ok=True)
with st.sidebar:
    st.header("📚 Local Knowledge Base")
    st.caption(
        "Upload documents related to your question, then rebuild the index so the "
        "agent can search them alongside web results."
    )
    
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

    st.markdown("---")
    st.markdown("### ⚙️ Guardrails & Constraints")
    st.markdown(f"""
    - **Model**: `openai/gpt-oss-20b`
    - **Research Subtasks**: `{MAX_TASKS}` (planning is a separate phase)
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
    verified_claims = state.get("verified_claims", [])
    evidence_matches = state.get("evidence_matches", [])
    by_id = {
        s.get("source_id"): s
        for s in sources
        if s.get("source_id")
    }
    n_local = sum(1 for s in sources if s.get("url", "").startswith("local://"))
    n_web = len(sources) - n_local

    # Calculate claim verification breakdown from verifier results, even if
    # citation matching did not produce a row for every verified claim.
    counts = {
        "SUPPORTED": 0,
        "PARTIAL": 0,
        "UNSUPPORTED": 0,
        "UNKNOWN": 0,
        "UNVERIFIED": 0,
    }
    if cites:
        for citation in cites:
            status = citation.get("status") or claim_status(citation.get("claim", ""))
            counts[status] = counts.get(status, 0) + 1
    else:
        for claim in verified_claims:
            status = claim_status(claim)
            counts[status] = counts.get(status, 0) + 1

    st.markdown("### 📊 Research Execution Summary")
    cols = st.columns(6)
    cols[0].metric("Research Subtasks", len(state.get("plan", [])))
    cols[1].metric("Web Sources", n_web)
    cols[2].metric("Local Documents", n_local)
    cols[3].metric("Evidence Extracted", len(state.get("evidence", [])))
    cols[4].metric("Supported Claims", counts["SUPPORTED"])
    cols[5].metric("Gap-Loop Rounds", state.get("iteration", 0))

    if state.get("gaps"):
        st.warning(
            f"⚠️ **{len(state['gaps'])} unaddressed gap(s)** remain after "
            f"max iterations ({state.get('iteration', 0)}/{MAX_GAP_ROUNDS}). "
            "See the 'Plan & Gaps' tab."
        )
    else:
        st.success("✅ **Research Thorough**: No remaining gaps identified after evaluation.")

    t1, t2, t3, t4, t5 = st.tabs([
        "📄 Research Report",
        "🔀 LangGraph Workflow",
        "🎯 Plan & Gaps",
        "💡 Evidence Items",
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
        st.caption(
            "Sequential state transition trace executed autonomously by the research agent from start to finish:"
        )

        workflow_diagram = """
    ```mermaid
    %%{init: {
    "theme": "base",
      "themeVariables": {
        "primaryTextColor": "#000000",
        "textColor": "#000000",
        "lineColor": "#555555"
    }
    }}%%

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

        GD -->|"Gaps found & iterations < <MAX_GAP_ROUNDS>"| RS["7. Targeted Re-search"]
        RS -->|"Loop back"| GD

        GD -->|"No gaps OR max rounds (<MAX_GAP_ROUNDS>)"| CV["8. Claim Verification"]

        CV --> CT["9. Semantic Citation Tracking"]
        CT --> FR["10. Final Cited Report (Downloadable)"]

        style UQ fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000000
        style RP fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000000

        style WS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000000
        style LR fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000000

        style WE fill:#263238,stroke:#546e7a,stroke-width:2px,color:#ffffff
        style LD fill:#263238,stroke:#546e7a,stroke-width:2px,color:#ffffff

        style EL fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000000
        style SE fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000

        style GD fill:#fbe9e7,stroke:#d84315,stroke-width:2px,color:#000000
        style RS fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000000

        style CV fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000000
        style CT fill:#ede7f6,stroke:#4527a0,stroke-width:2px,color:#000000

        style FR fill:#e8eaf6,stroke:#283593,stroke-width:2px,color:#000000
        ```
        """
        st.markdown(workflow_diagram.replace("<MAX_GAP_ROUNDS>", str(MAX_GAP_ROUNDS)))

        
        st.markdown("#### 🔄 Sequential State Execution Log")
        wf_cols = st.columns(2)
        with wf_cols[0]:
            st.markdown(f"""
            - **1. User Question**: `{state.get('question', '')[:60]}...`
            - **2. Research Planner**: Generated `{len(state.get('plan', []))}` research subtasks (max {MAX_TASKS})
            - **3. Dual Retrieval**:
              - Web Search: `{n_web}` sources
              - Local RAG: `{n_local}` passages
            - **4. Evidence Layer**: `{len(state.get('evidence', []))}` candidate evidence items extracted
            - **5. Source Evaluation**: Evaluated quality, authority, recency
            """)
        with wf_cols[1]:
            st.markdown(f"""
            - **6. Gap Detection**: Identified completeness & missing context
            - **7. Re-Search Loop**: Executed `{state.get('iteration', 0)}` round(s) *(max {MAX_GAP_ROUNDS})*
            - **8. Claim Verification**: `{len(cites)}` claims audited (`{counts['SUPPORTED']}` supported)
            - **9. Citation Tracking**: Dense vector cosine matching $\\ge 0.30$
            - **10. Final Cited Report**: Formatted markdown with references
            """)
        
    with t3:
        col_p, col_g = st.columns(2)
        with col_p:
            st.subheader(f"🧭 Research Subtasks (up to {MAX_TASKS})")
            st.caption("Planning is a separate phase; these are the named research queries the agent executes.")
            for i, t in enumerate(state.get("plan", []), 1):
                st.markdown(f"**{i}.** {t}")
            if not state.get("plan"):
                st.info("No research subtasks were generated.")
        with col_g:
            st.subheader("🕳️ Identified Gaps")
            gaps = state.get("gaps", [])
            if gaps:
                for i, g in enumerate(gaps, 1):
                    st.markdown(f"**{i}.** {g}")
            else:
                st.info("No gaps detected. Evidence comprehensively answers the core question.")

    with t4:
        st.subheader(f"💡 Extracted Evidence ({len(state.get('evidence', []))} facts)")
        for i, ev in enumerate(state.get("evidence", []), 1):
            st.markdown(f"**{i}.** {ev}")

    with t5:
        st.markdown("### 🔍 Claim Verification & Semantic Source Alignment")
        st.caption(
            "Claims are categorized by how strongly the retrieved research evidence supports them. "
            "SentenceTransformers cosine similarity matches claims to recorded source content; "
            "matches below 0.30 are left uncited. Similarity is a retrieval match, not proof of truth."
        )

        st.subheader("Claim verification results")
        def claim_key(value):
            value = re.sub(r"^\W*\d+[.)]\s*", "", str(value))
            value = re.sub(
                r"\[(SUPPORTED|PARTIALLY SUPPORTED|PARTIAL|UNSUPPORTED|UNKNOWN|UNVERIFIED)\]",
                "",
                value,
                flags=re.I,
            )
            return " ".join(value.casefold().split())

        verified_by_key = {claim_key(claim): claim for claim in verified_claims}
        cited_keys = {claim_key(c.get("claim", "")) for c in cites}
        unmatched_claims = [
            claim for claim in verified_claims if claim_key(claim) not in cited_keys
        ]
        represented_evidence = {
            citation.get("evidence_index")
            for citation in cites
            if isinstance(citation.get("evidence_index"), int)
        }
        evidence = state.get("evidence", [])
        unrepresented_evidence = [
            (index, item)
            for index, item in enumerate(evidence)
            if index not in represented_evidence
        ]
        if not cites:
            unrepresented_evidence = list(enumerate(evidence))
        needs_evidence_fallback = bool(unrepresented_evidence)

        if state.get("verification_error"):
            st.error(state["verification_error"])
        if not cites:
            st.warning(
                "No claim citations were produced. Research evidence below is marked UNVERIFIED; "
                "no claim status is inferred."
            )
        elif unmatched_claims:
            st.warning(
                "Some verifier claims have no matching citation. Their verifier status is preserved, "
                "but no source match is inferred."
            )

        for c in cites:
            claim_text = c.get("claim", "")
            sid = c.get("source_id")
            score = c.get("score")
            verified_claim = verified_by_key.get(claim_key(claim_text))
            status = c.get("status") or claim_status(verified_claim or claim_text)
            display_status = "UNVERIFIED" if status in {"UNKNOWN", "UNVERIFIED"} else status
            
            src = by_id.get(sid)
            if src:
                url = src.get("url", "")
                title = src.get("title") or url
                ref = f"[{sid}: {title}]({url})" if not url.startswith("local://") else f"**{sid}** (`{title}`)"
            else:
                ref = "*(No source met the 0.30 similarity threshold)*"
            
            st.markdown(f"{ICON.get(display_status, '❔')} **{display_status}** — {claim_text}")
            score_label = f"`{score:.2f}`" if isinstance(score, (int, float)) else "unavailable"
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **Closest source match**: {ref} (Cosine similarity: {score_label})")
            if c.get("quote"):
                st.markdown(f"**Verifier quote (exact):**\n\n> {c['quote']}")
            if c.get("verifier_status") and c["verifier_status"] != status:
                st.caption(
                    f"Verifier assessment: {c['verifier_status']}; "
                    f"final status is {status} because the source did not meet alignment requirements."
                )
            if src:
                with st.expander("View matched source excerpt", expanded=False):
                    st.write(src.get("content", ""))
            elif c.get("origin_source_id") in by_id:
                origin_source = by_id[c["origin_source_id"]]
                with st.expander("View originating source (not cited)", expanded=False):
                    st.write(origin_source.get("content", ""))
            st.markdown("---")

        for claim in unmatched_claims:
            status = claim_status(claim)
            st.markdown(f"{ICON.get(status, '❔')} **{claim}**")
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;↳ **Source match**: unavailable (no aligned citation)")
            st.markdown("---")

        if needs_evidence_fallback:
            st.markdown("#### Extracted evidence — UNVERIFIED")
            st.caption(
                "The closest-source score below helps locate relevant research text; "
                "it does not verify the evidence or establish that source as its definitive origin."
            )
            if not unrepresented_evidence:
                st.info("No extracted evidence is available to display.")
            for index, item in unrepresented_evidence:
                st.markdown(f"⚪ **UNVERIFIED** — {item}")
                match = evidence_matches[index] if index < len(evidence_matches) else {}
                src = by_id.get(match.get("source_id"))
                score = match.get("score")
                if src:
                    title = src.get("title") or src.get("url") or src["source_id"]
                    score_label = f"`{score:.2f}`" if isinstance(score, (int, float)) else "unavailable"
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **Origin source (not a verification)**: "
                        f"**{src['source_id']}** ({title}; cosine similarity: {score_label})"
                    )
                    with st.expander("View closest source passage", expanded=False):
                        st.write(src.get("content", ""))
                else:
                    st.markdown(
                        "&nbsp;&nbsp;&nbsp;&nbsp;↳ **Source provenance**: unavailable"
                    )
                st.markdown("---")