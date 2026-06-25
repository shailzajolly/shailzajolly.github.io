---
layout: page
title: Sift
description: Agentic retrieval for technical documentation. Pairs BM25 with semantic search to handle dense specs, part numbers, and product codes that break naive vector search. Every answer carries a provenance trail back to the source.
img:
importance: 1
category: work
github: https://github.com/shailzajolly/Sift
---

Technical documentation is a hard retrieval problem. Datasheets, manuals, and selection guides are dense with out-of-vocabulary tokens (product codes like `WL12-3P2431`, part numbers, specification values) that semantic embeddings handle poorly. A naive vector search returns plausible-sounding passages that miss the exact token match. The answer looks right and is wrong.

Sift is built specifically for this. It pairs **BM25 lexical retrieval** (which nails exact-token matches) with **semantic search**, fuses the two rankings, reranks with a cross-encoder, and serves the result to an AI agent that answers only from the retrieved passages, with a citation back to the source document and page for every claim.

```
PDFs → chunk → index (BM25 + Chroma) → MCP search_docs → grounded agent
```

The retrieval layer is exposed as an **MCP server**, so it plugs directly into Claude Code as a tool. The agent is instructed to refuse to answer from memory: if the answer isn't in the retrieved passages, it says so.

**Stack:** Python, Docling (PDF chunking), Chroma (vector store), BM25, cross-encoder reranking, Model Context Protocol, Claude.

**[View on GitHub](https://github.com/shailzajolly/Sift)**
