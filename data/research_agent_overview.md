# Autonomous Research Agent Knowledge Base: Architecture & Workflow Guide

## Overview
The Autonomous Research Agent is designed to conduct end-to-end multi-step factual research without hallucination. It pairs real-time web retrieval with a local RAG pipeline to ensure verified, grounded synthesis.

## 1. Dual Retrieval Strategy
- **Web Search**: Dynamic querying via Tavily API for recent real-world data and external references.
- **Local RAG**: Dense vector retrieval utilizing local SentenceTransformers (`all-MiniLM-L6-v2`) embeddings and cosine similarity over segmented document chunks (800 chars, 100 overlap).

## 2. Iterative Gap-Filling Loop
- After initial evidence extraction, the agent runs a dedicated gap detector.
- If missing perspectives, unsupported claims, or information deficits are identified, the agent triggers focused gap-research queries (up to 2 iterations).

## 3. Claim Verification & Grounded Citation Matching
- Extracted claims are independently audited and labeled as SUPPORTED, PARTIALLY SUPPORTED, or UNSUPPORTED.
- Every claim is matched against source chunks using semantic vector cosine similarity. Claims meeting or exceeding the similarity threshold are anchored with definitive source IDs ([SRC-xxx]), preventing unreferenced assertions.
