---
layout: post
title: "Understanding Attention from the Bottom Up"
date: 2025-03-11 12:00:00
description: >
  A researcher's journey from RNN bottlenecks to FlashAttention-3 —
  with runnable code at every step.
tags: [attention, transformers, deep-learning, nlp]
categories: research
thumbnail: assets/img/attention-cover.png
related_posts: true
toc:
  sidebar: left
citation: false
related_publications: false
---

Attention is a word interpreted quite differently across communities.
For a toddler, it signals caution; for an athlete, it means posture
and sharp focus. But in the ML community, the word immediately conjures
Transformers [Vaswani et al., 2017]. What intrigues me is that
the word's Latin root _attendere_, meaning _"to stretch toward"_
maps almost perfectly onto what the mechanism does mathematically: it
lets a model reach across a sequence and selectively pull in whatever
is most relevant at each step. This blog traces how that idea was born,
why it became indispensable, and how it continues to evolve today.

My goal is to reconstruct the **researcher's internal monologue**: what
was breaking in the systems of the time, what intuition pointed toward
a fix, and how each fix then revealed the next limitation. By the end,
you should have a complete mental model of attention from its origins
in sequence-to-sequence translation all the way to the hardware-aware
algorithms that power today's large language models.

> **A note on why this blog matters now.**
> Perplexity AI CEO Aravind Srinivas has [argued](https://www.firstpost.com/tech/perplexity-ceo-believes-ai-could-brings-computer-science-back-to-its-mathematical-roots-13989960.html) [Srinivas, 2024]
> that AI is not making computer science obsolete. It is driving the
> field _back_ to its mathematical roots. As AI automates routine coding,
> competitive advantage shifts toward deep understanding of linear algebra,
> calculus, and first-principles engineering. Every code cell in this blog
> is therefore written in **plain PyTorch mathematics**. No high-level
> attention libraries, no black-box wrappers because understanding the
> fundamentals is what separates someone who can _use_ these tools from
> someone who can _improve_ or _rethink_ them.

Every section is paired with a code cell. All cells are part of a single
Jupyter notebook, linked below, that you can clone and run end-to-end.

{% if site.data.repositories.github_repos %}

<div class="repositories d-flex flex-wrap flex-md-row flex-column
            justify-content-between align-items-center">
  {% include repository/repo.liquid
     repository="shailzajolly/attention-notebook" %}
</div>
{% endif %}

---

## Section 1 — The World Before Attention

{: #section-1}

To understand why attention was needed, we first need to understand
what researchers were working with before it and precisely where
it broke.

### The Running Example

Throughout this blog I will use a single sentence as a test bed:

> _"The poet published many poems but is still grounded."_

This sentence is deliberately tricky. The verb _is_ must agree with
_poet_ (singular), not with _poems_ (plural), even though _poems_ is
the nearer noun. A model must track **syntactic recency** which is the true
grammatical subject and resist **sequential recency**, the temptation
to agree with the most recently seen noun. Getting this right across a
long sequence is exactly what pre-attention architectures struggled with.

### Seq2Seq: The Bottleneck

The dominant approach for sequence tasks before attention was the
**sequence-to-sequence (seq2seq)** encoder–decoder architecture
[Sutskever et al., 2014]. An encoder, typically an LSTM,
[Hochreiter & Schmidhuber, 1997] or a GRU [Cho et al., 2014] reads the
source sentence token by token and updates a hidden state at each step.
After the final token, the **last hidden state** $$\mathbf{h}_n$$ is
handed to the decoder as its initial state, and the decoder generates
the output sequence one token at a time.

This single vector must compress the _entire meaning_ of the input
sequence. For short sentences this works adequately. For longer ones,
it becomes a critical bottleneck: information from early tokens is
progressively overwritten as later tokens are processed
[Sutskever et al., 2014]. In translation benchmarks, this
manifests as a sharp drop in BLEU score for sentences longer than roughly
20 words [Bahdanau et al., 2015].

The problem is compounded during training. Error gradients must flow
backward through every recurrent step; for long sequences they either
shrink toward zero (**vanishing gradients**) or explode, making it
effectively impossible for the encoder to learn that a word from 30 steps
ago is still relevant [Hochreiter & Schmidhuber, 1997]. LSTMs and GRUs
mitigate this with gating mechanisms but do not eliminate the fundamental
bottleneck: the decoder still receives only $$\mathbf{h}_n$$, regardless
of how much useful information sits in the intermediate encoder states.

### Code — Cell 1: The Bottleneck

> ##### Design philosophy — why no LSTM here?
>
> A framework LSTM is a helpful engineering tool, but its internal gates,
> weight initialisations, and optimiser state obscure the core idea we
> are trying to illustrate. Instead, the setup below builds the encoder with
> **three explicit interpretability hacks** that trade realism for
> transparency:
>
> 1. **Structured embeddings instead of random ones.**
>    Rather than letting an optimiser learn embeddings over thousands of
>    steps, we hand-craft vectors that simulate what training would
>    eventually converge to. `poet` and `is` both receive
>    `[1, 0, 0, 0, 0, 0]` — a strong "subject / verb-needing-subject"
>    signal on the first dimension. `poems` receives `[0.2, 0, 0, 0, 0, 0]`,
>    a weaker distractor on the same axis. This makes the expected
>    behaviour of attention legible before we even run it.
> 2. **Noise scaled to 0.05.**
>    All other words are initialised with `torch.randn(d_model) * 0.05`.
>    Multiplying by 0.05 keeps distractor values very small, so signal
>    dominates over noise and the demo remains numerically stable across
>    random seeds.
> 3. **`torch.eye` as the encoder transformation.**
>    The encoder applies `W_enc = torch.eye(d_model)`, an identity matrix,
>    followed by `tanh`. This preserves every embedding dimension without
>    mixing or rotating it. Using the identity means encoded vectors are
>    directly interpretable: we know exactly what information is available
>    at each position, with no hidden rotations introduced by a learned
>    weight matrix. In a real trained model `W_enc` would be learned; here
>    we keep it transparent so attention's behaviour is fully attributable
>    to the embeddings.

```python
# ── Cell 1: shared setup — run once, all later cells reuse these ──────────────
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# -------------------------------------------------------
# Sentence
# -------------------------------------------------------

sentence = ["the", "poet", "published", "many", "poems", "but", "is", "still", "grounded"]

word_2_idx = {word: idx for idx, word in enumerate(sentence)}
tokens = torch.tensor([word_2_idx[word] for word in sentence])

seq_len = len(sentence)
d_model = 6

# -------------------------------------------------------
# Structured embeddings (not random) — Interpretability Hack #1
# -------------------------------------------------------
# poet  → strong "subject" signal on dim-0
# poems → weaker distractor on dim-0 (same axis, lower magnitude)
# is    → identical to poet; it needs to find its subject
# All other words get tiny noise so signal dominates.

embedding = torch.zeros(seq_len, d_model)

embedding[word_2_idx["poet"]]  = torch.tensor([1.0, 0, 0, 0, 0, 0])   # subject signal
embedding[word_2_idx["poems"]] = torch.tensor([0.2, 0, 0, 0, 0, 0])   # plural distractor
embedding[word_2_idx["is"]]    = torch.tensor([1.0, 0, 0, 0, 0, 0])   # verb needing subject

# Interpretability Hack #2: scale noise to 0.05
for w in sentence:
    if embedding[word_2_idx[w]].sum() == 0:
        embedding[word_2_idx[w]] = torch.randn(d_model) * 0.05

x = embedding[tokens]

# -------------------------------------------------------
# Encoder — Interpretability Hack #3: torch.eye
# -------------------------------------------------------
# W_enc = identity matrix. No mixing, no rotation.
# tanh is applied for consistency with real encoders, but because
# inputs are near zero (noise) or exactly ±1 (signals),
# tanh ≈ identity here too. Everything remains fully inspectable.

W_enc = torch.eye(d_model)
encoder_outputs = torch.tanh(x @ W_enc)   # shape: [seq_len, d_model]

# -------------------------------------------------------
# Bottleneck: only the last encoder state reaches the decoder
# -------------------------------------------------------

h_n = encoder_outputs[-1]   # "grounded" — the final token

print("Encoder outputs shape:", encoder_outputs.shape)
print("\nBottleneck h_n — the only vector the decoder sees:")
print(h_n)
print(f"\nThe decoder receives {h_n.numel()} numbers to represent")
print(f"a {seq_len}-token sentence. All other encoder states are DISCARDED.")
```

```
Encoder outputs shape: torch.Size([9, 6])

Bottleneck h_n — the only vector the decoder sees:
tensor([ 0.0259, -0.0655,  0.0096,  0.0271, -0.1105,  0.0129])

The decoder receives 6 numbers to represent
a 9-token sentence. All other encoder states are DISCARDED.
```

**What to observe.** The vector `h_n` corresponds to _"grounded"_ i.e., the final
token. Its first dimension is `0.026`, far from the `1.0` that `poet` had nine
steps earlier. The bottleneck is not a theoretical concern; it is visible
directly in these six numbers.

---

> **Researcher's question:** Every intermediate encoder hidden state is
> discarded after the final encoding step. But those states capture
> position-specific context that $$\mathbf{h}_n$$ has already partially
> forgotten. What if the decoder could query _all_ of them and decide
> which ones matter at each output step?

## Section 2 — Attention as the Solution

{: #section-2}

The question above contains two observations that, when combined, produce
the attention mechanism.

**Observation 1: The dot product measures similarity.**
Given two vectors $$\mathbf{a}$$ and $$\mathbf{b}$$, their dot product
$$\mathbf{a} \cdot \mathbf{b}$$ is large when the vectors point in the same
direction and small (or negative) when they diverge. This gives us a cheap,
differentiable measure of alignment between any pair of vectors.

**Observation 2: We already have all the encoder states.**
The bottleneck is a _choice_, not a necessity. An RNN encoder computes a
hidden state at every time step; only the last one is traditionally passed
to the decoder. The others sit unused in memory.

Combining these: what if the decoder used its current hidden state as a
**query**, computed a dot product against every encoder state (the **keys**),
and then formed a weighted sum of the encoder states (the **values**)
proportional to those scores?

That is exactly dot-product attention [Bahdanau et al., 2015].
Instead of receiving a single bottleneck vector, the decoder now receives a
**context vector** $$\mathbf{c}$$ that is a soft mixture of _all_ encoder
states, weighted by relevance:

$$
e_i = \mathbf{h}_\text{dec} \cdot \mathbf{h}_i^\text{enc}, \qquad
\alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}, \qquad
\mathbf{c} = \sum_i \alpha_i \, \mathbf{h}_i^\text{enc}
$$

The weights $$\alpha_i$$ are the **attention distribution**: they specify how
much each encoder position contributes to the current decoder step.
Crucially, these weights are **recomputed at every decoder step**. The
decoder can attend to entirely different parts of the source at each output
token. The bottleneck is gone.

### Code — Cell 2: Dot-Product Attention

```python
# ── Cell 2: dot-product attention — continues from Cell 1 ─────────────────────

# Query  = encoder output at the position of "is"
#          (in a real seq2seq model this would be the decoder's hidden state;
#           here we approximate it with the encoder output at "is" to keep
#           the example self-contained)
query  = encoder_outputs[word_2_idx["is"]]   # shape: [d_model]

keys   = encoder_outputs   # shape: [seq_len, d_model]
values = encoder_outputs   # shape: [seq_len, d_model]

# Step 1 — Dot-product scores: how well does each key match the query?
scores = keys @ query                          # shape: [seq_len]

# Step 2 — Softmax → attention weights (sum to 1)
attn_weights = F.softmax(scores, dim=0)        # shape: [seq_len]

# Step 3 — Weighted sum of values → context vector
context = attn_weights @ values                # shape: [d_model]

print("Attention weights (sorted by score):")
ranking = sorted(
    zip(sentence, attn_weights.tolist()),
    key=lambda pair: -pair[1]
)
for word, score in ranking:
    print(f"  {word:12s}  {score:.4f}")

focus = sentence[attn_weights.argmax()]
print(f"\nModel focuses on: '{focus}'")
print("\nContext vector:")
print(context)
```

```
Attention weights (sorted by score):
  poet          0.1659
  is            0.1659
  poems         0.1079
  the           0.0980
  grounded      0.0952
  published     0.0937
  still         0.0935
  many          0.0900
  but           0.0899

Model focuses on: 'poet'
```

`poet` and `is` tie at the top because their embeddings are identical
(`[1, 0, 0, 0, 0, 0]`). `poems` scores next — it shares the same non-zero
dimension but at a weaker magnitude (`0.2`). All other tokens score near the
uniform baseline. Without any learned parameters, the dot product already
recovers the correct grammatical subject.

### Bahdanau's Additive Attention: A Nonlinear Scoring Function

The original attention paper [Bahdanau et al., 2015] did not use
a raw dot product. Bahdanau et al. proposed replacing it with a small
feedforward network:

$$
e_i = \mathbf{v}^\top \tanh\!\left(\mathbf{W}_1 \mathbf{h}_\text{dec}
      + \mathbf{W}_2 \mathbf{h}_i^\text{enc}\right)
$$

The intuition is that a nonlinear scoring function can capture asymmetric
relationships that a dot product misses. If $$\mathbf{h}_\text{dec}$$ needs
to be _similar to but not identical to_ $$\mathbf{h}_i^\text{enc}$$ in a
way that depends on their sum, tanh can express that while a dot product
cannot.

In the simplified form below (no learned $$\mathbf{W}_1, \mathbf{W}_2$$),
this reduces to `tanh(query + key).sum()` for each key — enough to
illustrate the structural difference.

> ##### Additive vs. multiplicative scoring
>
> Additive attention is more expressive (a nonlinear score) but requires a
> forward pass through a scoring network for every (query, key) pair.
> Dot-product attention maps directly to a matrix multiply, making it
> dramatically more GPU-efficient. This is one reason the Transformer
> chose multiplicative attention — with a scaling fix to handle large
> dimensions, described next.

```python
# ── Cell 3: additive attention (Bahdanau 2015) ────────────────────────────────

query  = encoder_outputs[word_2_idx["is"]]
keys   = encoder_outputs
values = encoder_outputs

# Nonlinear scoring: tanh(query + key).sum() for each key
scores = torch.stack([torch.tanh(query + k).sum() for k in keys])

attn_weights = F.softmax(scores, dim=0)
context = attn_weights @ values

print("Additive Attention weights (sorted):")
ranking = sorted(zip(sentence, attn_weights.tolist()), key=lambda x: -x[1])
for word, score in ranking:
    print(f"  {word:12s}  {score:.4f}")
print("\nFocus:", sentence[attn_weights.argmax()])
```

```
Additive Attention weights (sorted):
  poet          0.1392
  is            0.1392
  poems         0.1184
  but           0.1136
  published     0.1037
  many          0.1034
  still         0.0988
  grounded      0.0953
  the           0.0885

Focus: poet
```

The correct subject is still recovered, but the distribution is flatter
than with dot-product attention. The `tanh` nonlinearity compresses extreme
values, which reduces contrast between `poet` and the distractors. In a
fully trained model, the learned weight matrices would sharpen this
distribution; here we see the baseline behavior of the scoring function alone.

---

> **Researcher's question:** Bahdanau attention solves the bottleneck problem,
> but it still requires a sequential RNN encoder. Each token's representation
> depends on processing all prior tokens first, which prevents parallelisation.
> Token 2 cannot be processed until token 1 is done. What if we could remove
> the recurrence entirely and keep only the attention?

## Section 3 — Self-Attention and the Transformer

{: #section-3}

The insight that led to the Transformer [Vaswani et al., 2017] was
deceptively simple: **recurrence is not necessary**. If attention can replace
the bottleneck vector, can it also replace the sequential encoder?

In Bahdanau's setup, the query came from the decoder which is a hidden state
_outside_ the encoder's sequence. Self-attention takes the next logical step:
**every word in a sequence acts as both a query and a key-value pair for
every other word in the same sequence**. There is no separate decoder; each
token enriches itself by attending to the full context.

This gives us two fundamental properties that seq2seq could never achieve
simultaneously:

1. **Direct connections.** Every pair of tokens is connected in a single
   layer, regardless of their distance. `is` can directly attend to `poet`
   without routing information through the eight intervening tokens.

2. **Full parallelism.** Because no token's representation depends on a
   sequentially computed predecessor, all tokens can be processed at once.
   The entire sentence fits into a single matrix multiply.

The sentence `"The poet published many poems but is still grounded"` now
computes nine attention distributions simultaneously i.e., one per token, each
reading the full context. _That_ is how modern LLMs process hundreds of
thousands of tokens in parallel.

### Scaled Dot-Product Attention

There is a numerical problem lurking in plain dot-product attention. When
the embedding dimension $$d_k$$ is large, dot products can grow very large
in magnitude. If the query and key vectors are random with zero mean and unit
variance, each element of $$\mathbf{q} \cdot \mathbf{k}$$ has variance 1,
so the sum of $$d_k$$ elements has variance $$d_k$$. With $$d_k = 64$$
(a modest Transformer dimension), this gives a standard deviation of 8,
which pushes softmax into extremely peaked regions where gradients vanish and
the model stops learning.

The fix [Vaswani et al., 2017] is to divide scores by
$$\sqrt{d_k}$$ before softmax:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

Division by $$\sqrt{d_k}$$ keeps the variance of dot products at 1
regardless of dimension, which keeps softmax in a well-behaved gradient
region throughout training.

```python
# ── Cell 3b: scaled dot-product attention ─────────────────────────────────────

query  = encoder_outputs[word_2_idx["is"]]
keys   = encoder_outputs
values = encoder_outputs

# Divide by sqrt(d_model) before softmax
scores       = (keys @ query) / (d_model ** 0.5)   # shape: [seq_len]
attn_weights = F.softmax(scores, dim=0)
context      = attn_weights @ values

print("Scaled Dot-Product Attention weights (sorted):")
ranking = sorted(zip(sentence, attn_weights.tolist()), key=lambda x: -x[1])
for word, score in ranking:
    print(f"  {word:12s}  {score:.4f}")
print("\nFocus:", sentence[attn_weights.argmax()])
```

```
Scaled Dot-Product Attention weights (sorted):
  poet          0.1321
  is            0.1321
  poems         0.1113
  the           0.1070
  grounded      0.1054
  published     0.1053
  still         0.1044
  many          0.1033
  but           0.1033

Focus: poet
```

The ranking is stable, but the distribution is slightly flatter than without
scaling. In a high-dimensional setting ($$d_k = 512$$ or $$1024$$), this
difference is decisive: unscaled attention produces near-one-hot distributions
that give essentially zero gradient signal.

### Code — Cell 4: Self-Attention (Full Matrix)

> ##### A note on projection matrices
>
> In a real Transformer, queries, keys, and values are _projected_ into
> separate learned subspaces via parameter matrices
> $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$$:
>
> $$
> \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad
>   \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad
>   \mathbf{V} = \mathbf{X}\mathbf{W}_V
> $$
>
> These matrices are initialised randomly and updated by gradient descent.
> Over training, $$\mathbf{W}_Q$$ and $$\mathbf{W}_K$$ learn to project
> inputs into a space where semantically related tokens align; $$\mathbf{W}_V$$
> extracts information most useful downstream. Here we set all three to
> `encoder_outputs` directly (identity projections) so the attention
> behaviour is attributable entirely to the hand-crafted embeddings.

```python
# ── Cell 4: self-attention — all tokens attend to all tokens ──────────────────

Q = encoder_outputs   # shape: [seq_len, d_model]
K = encoder_outputs
V = encoder_outputs

# Full attention matrix: every token queries every other token
scores      = (Q @ K.T) / (d_model ** 0.5)   # shape: [seq_len, seq_len]
attn_matrix = F.softmax(scores, dim=1)        # normalise each row independently

output = attn_matrix @ V                      # shape: [seq_len, d_model]

print("Self-attention matrix shape:", attn_matrix.shape)

# Inspect where "is" attends
is_idx  = word_2_idx["is"]
weights = attn_matrix[is_idx]

print("\nWhere 'is' attends (sorted):")
ranking = sorted(zip(sentence, weights.tolist()), key=lambda x: -x[1])
for word, score in ranking:
    print(f"  {word:12s}  {score:.4f}")
```

```
Self-attention matrix shape: torch.Size([9, 9])

Where 'is' attends (sorted):
  poet          0.1321
  is            0.1321
  poems         0.1113
  the           0.1070
  grounded      0.1054
  published     0.1053
  still         0.1044
  many          0.1033
  but           0.1033
```

The `attn_matrix` is a $$9 \times 9$$ matrix: every row is a probability
distribution over all nine tokens. Row 6 (`is`) shows the familiar
`poet`dominant pattern. But now _every_ row is computed simultaneously i.e.,
all nine token representations are enriched in a single parallel pass.
That is the computational breakthrough that makes Transformers scalable.

### Multi-Head Attention

One head of self-attention asks a single question: "given the full context,
what is most relevant for each position?" But natural language requires
_many_ types of relevance simultaneously. A single head cannot learn a
subject-verb dependency in the same subspace as a pronoun-referent
dependency or a semantic paraphrase relationship. These correspond to
different geometric structures in the representation space.

Multi-head attention [Vaswani et al., 2017] runs $$h$$
independent attention functions in parallel on _different projected
subspaces_ of the input, then concatenates the results:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,\mathbf{W}_O
$$

where $$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_{Q_i},
\mathbf{K}\mathbf{W}_{K_i}, \mathbf{V}\mathbf{W}_{V_i})$$.

Each head operates on dimension $$d_k = d_\text{model} / h$$, so total
compute stays roughly constant. Crucially, each head can specialise: one
learns syntactic dependencies (subject-verb), another semantic similarity,
a third long-range co-reference. In our running example, the six embedding
dimensions are split across two heads of three dimensions each. Think of
it as assigning separate analysts to separate aspects of the problem.

```python
# ── Cell 5: multi-head attention (conceptual split) ───────────────────────────

num_heads = 2
head_dim  = d_model // num_heads   # 6 // 2 = 3

# Split the d_model dimension across heads
# Shape: [seq_len, num_heads, head_dim]
Q = encoder_outputs.view(seq_len, num_heads, head_dim)
K = encoder_outputs.view(seq_len, num_heads, head_dim)
V = encoder_outputs.view(seq_len, num_heads, head_dim)

is_idx = word_2_idx["is"]

for h in range(num_heads):
    q       = Q[is_idx, h]                           # shape: [head_dim]
    scores  = (K[:, h] @ q) / (head_dim ** 0.5)     # shape: [seq_len]
    weights = F.softmax(scores, dim=0)

    print(f"\nHead {h} — where 'is' attends:")
    ranking = sorted(zip(sentence, weights.tolist()), key=lambda x: -x[1])
    for word, score in ranking:
        print(f"  {word:12s}  {score:.4f}")
```

```
Head 0 — where 'is' attends:
  poet          0.1410
  is            0.1410
  poems         0.1100
  the           0.1041
  grounded      0.1021
  published     0.1018
  still         0.1015
  many          0.0995
  but           0.0990

Head 1 — where 'is' attends:
  the           0.1111
  poet          0.1111
  published     0.1111
  many          0.1111
  poems         0.1111
  but           0.1111
  is            0.1111
  still         0.1111
  grounded      0.1111
```

Head 0 sharpens around `poet` as it has found the subject-verb signal in
dims 0–2. Head 1 produces a perfectly uniform distribution. dims 3–5 are
near-zero noise for all tokens, so all keys look the same from Head 1's
perspective. In a trained model, learned projection matrices
$$\mathbf{W}_{Q_1}, \mathbf{W}_{K_1}$$ would rotate Head 1 into a subspace
where a _different_ linguistic property dominates. The uniform output here
is a property of our toy embeddings, not a flaw in the mechanism.

The key takeaway: multi-head attention does not increase per-head complexity,
but it multiplies the number of relationship types the model can discover
simultaneously. This is one of the core design choices that made Transformers
generalist models rather than task-specific ones.

---

> **Researcher's question:** We now have an $$O(n^2)$$ computation. For
> $$n = 10{,}000$$ tokens, the attention matrix has $$10^8$$ entries.
> Storing this at FP32 requires 400 MB _per layer_. With 96 layers and
> large batches, we are talking about terabytes. Is the bottleneck now
> memory rather than expressiveness?

## Section 4 — FlashAttention: Attention Meets Hardware

{: #section-4}

By 2022, the Transformer had become the dominant architecture across language,
vision, and speech. But scaling it to longer contexts was hitting a wall. It was
not a mathematical one, but a hardware one.

### The Memory Wall

Modern GPUs are organised into a two-tier memory hierarchy:

- **SRAM (Shared Memory / L2 cache):** Very fast, about hundreds of TB/s of
  bandwidth. Very small, about a few MB total on-chip.
- **HBM (High Bandwidth Memory):** Much slower, around ~2 TB/s. Much larger, around
  tens of GB.

Standard attention loads $$\mathbf{Q}$$, $$\mathbf{K}$$, $$\mathbf{V}$$ from
HBM, materialises the full $$n \times n$$ attention score matrix in HBM, then
reads it back to compute the weighted sum. Every read and write to HBM is
comparatively slow; the GPU's compute cores sit idle waiting for data. The
attention computation is **memory-bound**, not compute-bound.

Dao et al. [Dao et al., 2022] made a critical observation:
**we never actually need the full $$n \times n$$ matrix**. We only need the
final output, which is a weighted sum of values. The full matrix is written to
HBM and then immediately read back which is an enormous round-trip that achieves
nothing except satisfying the sequential structure of the naive algorithm.

### Tiling and the Online Softmax Trick

FlashAttention splits $$\mathbf{Q}$$, $$\mathbf{K}$$, $$\mathbf{V}$$ into
**tiles** that fit in SRAM and processes them one at a time:

1. Load a tile of $$\mathbf{K}$$ and $$\mathbf{V}$$ into SRAM.
2. Load a tile of $$\mathbf{Q}$$ into SRAM.
3. Compute partial $$\mathbf{Q}\mathbf{K}^\top$$ scores for this tile.
4. Update a **running maximum** and **running normaliser** using the online
   softmax algorithm.
5. Accumulate $$\exp(s - m) \cdot \mathbf{V}_\text{block}$$ into a running
   context vector, rescaling the previous accumulation when a new maximum is
   found.
6. Repeat for all tiles; divide by the final running sum.

The running maximum is the critical ingredient. Softmax requires dividing
each exponential by the sum of all exponentials, but we cannot compute that
sum until we have seen all scores. The online softmax maintains a running
maximum $$m$$ and rescales previous partial sums when a larger value arrives,
ensuring numerical stability without ever writing the full matrix to HBM.

The result: FlashAttention computes **exact** attention (no approximation,
no quality loss) with $$O(n)$$ HBM reads instead of $$O(n^2)$$. In practice
this yields 2–4× wall-clock speedups and reduces memory from quadratic to
linear in sequence length [Dao et al., 2022].

```python
# ── Cell 6: Flash Attention (block-wise tiled computation) ────────────────────

block_size = 3   # tile size (in a real GPU kernel, matches SRAM capacity)

Q = encoder_outputs
K = encoder_outputs
V = encoder_outputs

q = Q[word_2_idx["is"]]   # single query for illustration

running_max     = None
running_sum     = None
running_context = None

for start in range(0, seq_len, block_size):
    end = min(start + block_size, seq_len)

    K_block = K[start:end]
    V_block = V[start:end]

    scores_block = (K_block @ q) / (d_model ** 0.5)
    block_max    = scores_block.max()

    if running_max is None:
        running_max     = block_max
        exp_scores      = torch.exp(scores_block - running_max)
        running_sum     = exp_scores.sum()
        running_context = exp_scores @ V_block
    else:
        new_max = torch.maximum(running_max, block_max)

        # Rescale previous accumulation to account for new maximum
        running_context *= torch.exp(running_max - new_max)
        running_sum     *= torch.exp(running_max - new_max)

        exp_scores       = torch.exp(scores_block - new_max)
        running_context += exp_scores @ V_block
        running_sum     += exp_scores.sum()

        running_max = new_max

# Final normalisation
context = running_context / running_sum

print("Flash Attention context vector:")
print(context)
```

```
Flash Attention context vector:
tensor([ 0.2277,  0.0004, -0.0246,  0.0077, -0.0187, -0.0030])
```

> ##### Verifying correctness
>
> You can verify this matches the standard scaled dot-product result by
> running both cells on the same inputs. FlashAttention is _exact_. The
> tiling is a scheduling optimisation, not an approximation. The strong
> first dimension (`0.2277`) reflects the dominance of `poet`
> (embedding `[1, 0, ...]`) in the weighted sum, as expected.

FlashAttention-2 [Dao, 2024] redesigned the
parallelisation strategy for the backward pass, minimised non-matmul FLOPs
in the rescaling steps, and improved work partitioning across GPU thread
blocks. This yields roughly 2× the throughput of FlashAttention-1 on A100 GPUs.
FlashAttention-3 [Shah et al., 2024] targets H100
hardware specifically, exploiting its asynchronous memory pipeline to overlap
data movement with compute and using FP8 arithmetic for the inner matrix
multiply, pushing throughput toward the theoretical hardware limit.

---

## Section 5 — From Theory to Production: Batches, Masking, and Training

{: #section-5}

Every code cell above processes a single sentence of nine tokens with
hand-crafted embeddings. Real language models train on batches of thousands
of sequences simultaneously, handle variable-length inputs, enforce causal
visibility constraints, and manage learned parameters that evolve over
billions of gradient steps. This section maps the single-example mechanics
to the production setting.

### 5.1 — Batching

In practice, $$\mathbf{Q}$$, $$\mathbf{K}$$, $$\mathbf{V}$$ carry a batch
dimension:

$$
\mathbf{Q} \in \mathbb{R}^{B \times T \times d_k}, \quad
\mathbf{K} \in \mathbb{R}^{B \times T \times d_k}, \quad
\mathbf{V} \in \mathbb{R}^{B \times T \times d_v}
$$

where $$B$$ is the batch size and $$T$$ is the sequence length. The
$$T \times T$$ attention matrix becomes $$B \times T \times T$$. For
$$B = 32$$, $$T = 2048$$, and FP16, this is $$32 \times 2048 \times 2048
\times 2 \approx 268 \text{ MB}$$ per layer. It is multiplied across all layers
and all heads, this is precisely why FlashAttention's avoidance of
materialising this matrix is decisive at production scale.

Sequences in a batch rarely have the same length. They are padded to a
common length (usually the longest sequence in the batch), and a
**padding mask** is applied before softmax so that padded positions
neither send nor receive attention weight. Without this mask, the model
would attend to meaningless padding tokens and learn spurious correlations.

A sketch of the batched setup:

```python
# Conceptual batched setup (not runnable without data)
B, T, d_model = 32, 512, 768
X = torch.randn(B, T, d_model)           # batch of token embeddings

W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)

Q = X @ W_Q   # shape: [B, T, d_model]
K = X @ W_K
V = X @ W_V

scores = (Q @ K.transpose(-2, -1)) / (d_model ** 0.5)   # [B, T, T]

# Apply padding mask (True where padding tokens are)
# padding_mask: [B, T] — True for positions that should be ignored
# scores = scores.masked_fill(padding_mask[:, None, :], float('-inf'))

attn_weights = F.softmax(scores, dim=-1)   # [B, T, T]
output = attn_weights @ V                  # [B, T, d_model]
```

### 5.2 — Causal Masking (Autoregressive Generation)

Language models generate text left to right: when predicting token $$t$$,
the model must not see tokens $$t+1, t+2, \ldots$$ This is enforced with a
**causal (lower-triangular) mask** applied to the score matrix before softmax:

```
Position:   0    1    2    3    4
  0       [ 0   -∞   -∞   -∞   -∞ ]
  1       [ 0    0   -∞   -∞   -∞ ]
  2       [ 0    0    0   -∞   -∞ ]
  3       [ 0    0    0    0   -∞ ]
  4       [ 0    0    0    0    0 ]
```

Positions marked $$-\infty$$ become exactly zero after softmax, blocking
attention to future tokens. This single design choice i.e., a triangular mask
applied at training time is what makes the same Transformer architecture
work as an encoder (BERT, no mask, bidirectional context) or a decoder
(GPT, causal mask, left-to-right generation). The architecture is identical;
only the mask changes.

```python
# Causal mask for a sequence of length T
T = 9
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
# True = blocked (will be set to -inf before softmax)
# shape: [T, T], upper triangle = True
```

### 5.3 — Learned Projection Matrices

Throughout the earlier sections, we set $$\mathbf{Q} = \mathbf{K} = \mathbf{V}
= \text{encoder_outputs}$$ which is equivalent to using identity projection matrices.
In a trained model, each attention layer has three learnable parameter matrices:

$$
\mathbf{W}_Q \in \mathbb{R}^{d_\text{model} \times d_k}, \quad
\mathbf{W}_K \in \mathbb{R}^{d_\text{model} \times d_k}, \quad
\mathbf{W}_V \in \mathbb{R}^{d_\text{model} \times d_v}
$$

plus an output projection $$\mathbf{W}_O \in \mathbb{R}^{h \cdot d_v \times
d_\text{model}}$$ that merges the multi-head outputs. These are updated by
gradient descent over millions of steps. Over training, $$\mathbf{W}_Q$$ and
$$\mathbf{W}_K$$ learn to project inputs into a space where semantically
aligned tokens have high dot products, and $$\mathbf{W}_V$$ learns to extract
the features most useful for the downstream task. The identity projections in
our demo are the special case where training would happen to converge to the
identity which is a simplification that makes the effect of embedding design
directly observable, with no learned transformation in the way.

### 5.4 — KV Cache During Inference

During autoregressive inference (generating one token at a time), recomputing
$$\mathbf{K}$$ and $$\mathbf{V}$$ for all previous tokens at every step would
be wasteful: each step processes an ever-growing prefix. The **KV cache**
solves this by storing the key and value tensors for all tokens generated so
far. At step $$t$$, only the new token's $$\mathbf{K}_t$$ and $$\mathbf{V}_t$$
are computed and appended to the cache; the query $$\mathbf{Q}_t$$ then
attends to the full cached history in one operation.

The cache grows linearly with context length. For a 70B-parameter model
with 32 layers, 64 heads, and 8K context at FP16:

$$
\text{KV cache size} = 2 \times 32 \times 64 \times 8192 \times 128 \times 2
\approx 8 \text{ GB per sequence}
$$

At batch size 32 for serving, this is 256 GB in KV cache alone which exceeds
the capacity of a single node. Managing this cache efficiently (paged
allocation, prefix sharing, quantisation) is one of the primary engineering
challenges in deploying large language models.

### 5.5 — Positional Encoding

Self-attention is **permutation-equivariant**: swap the order of input tokens
and the output permutes in the same way. The model has no inherent notion of
sequence order. Positional encodings inject order information by adding a
position-dependent signal to each token's embedding before it enters the
attention layers.

The original Transformer used fixed sinusoidal functions. Modern large models
typically use **Rotary Position Embeddings (RoPE)**, which apply
position-dependent rotations directly to the query and key vectors before
their dot product is computed. This has two practical advantages: it encodes
_relative_ position (how far apart are these two tokens?) rather than
_absolute_ position, and it generalises more gracefully to sequences longer
than those seen during training. RoPE is the positional encoding used in
LLaMA, Mistral, and most contemporary open-source models.

---

> **Researcher's question:** Even with FlashAttention and all the engineering
> optimisations described above, the fundamental $$O(n^2)$$ compute complexity
> in sequence length limits practical context windows. At $$n = 1{,}000{,}000$$
> tokens, a full novel, the attention computation alone would require
> terabytes of compute. Are there fundamentally better alternatives?

## Section 6 — Is Attention Still Evolving?

{: #section-6}

The attention mechanism described in this blog has driven a decade of
progress in NLP, vision, and multimodal AI. Yet the boundaries of what is
practical continue to shift, and researchers continue to push at those
boundaries from multiple directions.

### 6.1 — The Quadratic Bottleneck

Standard self-attention requires $$O(n^2)$$ time and memory in sequence
length $$n$$. For $$n = 128{,}000$$ tokens (a common production context
length in 2024–25), the raw attention matrix contains $$\sim 1.6 \times
10^{10}$$ entries. FlashAttention eliminates the memory cost of _storing_
this matrix, but the compute cost remains quadratic.

Two broad strategies have emerged to attack this:

#### Sparse Attention

Sparse attention methods restrict each token to attending only to a
structured subset of other tokens. **Longformer**
[Beltagy et al., 2020] combines a local sliding-window
pattern (each token attends to a fixed window of neighbours) with global
attention on a small set of task-specific tokens (e.g. the `[CLS]` token),
achieving $$O(n)$$ complexity for document-length inputs while retaining
most of the expressiveness of full attention. **BigBird** [Zaheer et al., 2020]
extends this with
random attention on top of the local and global patterns, adding theoretical
coverage guarantees.

The trade-off is architectural: the sparsity pattern must be designed or
learned, and some long-range dependencies may fall outside the pattern.

#### Linear Attention Approximations

Linear attention methods approximate the softmax attention matrix using
kernel functions or low-rank decompositions, reducing complexity to
$$O(n \cdot d)$$. The trade-off is that the approximation introduces error.
Tasks requiring precise retrieval of a specific fact from a long context which is
the needle-in-a-haystack test expose linear attention's weakness, because
such tasks rely on attending to a very small number of tokens with high
sharpness, which the approximation diffuses.

### 6.2 — Multi-Query and Grouped-Query Attention

During inference with the KV cache, memory cost grows linearly with the
number of attention heads: each head stores its own $$\mathbf{K}$$ and
$$\mathbf{V}$$ matrices for all past tokens. At long contexts and large
batch sizes, this dominates total GPU memory usage.

**Multi-Query Attention (MQA)** [Shazeer, 2019]
dramatically reduces this by having all query heads share a _single_ key-value
head. This cuts KV cache memory by a factor of $$h$$ (the number of query
heads) at the cost of a modest quality degradation, because each query head
can no longer attend to information projected into its own private key-value
space.

**Grouped-Query Attention (GQA)** [Ainslie et al., 2023]
interpolates: query heads are divided into $$g$$ groups, each sharing one
KV head. With $$g = h$$ this is standard MHA; with $$g = 1$$ this is MQA.
Models like Llama 3, Mistral, and Gemma use GQA with $$g = 4$$ or $$8$$
to balance inference speed against model quality. This is not a theoretical
curiosity, rather it is an engineering decision that directly determines how many
users a deployed model can serve per second on a given hardware budget.

### 6.3 — State-Space Models: A Different Paradigm

Perhaps the most significant challenge to attention comes from a completely
different direction. **Mamba** [Gu & Dao, 2023] is a selective
state-space model (SSM) that achieves linear-time sequence processing by
learning to _selectively_ retain or forget information at each step. It is a
learned gating mechanism applied to a fixed-size hidden state. It is
philosophically reminiscent of an LSTM, but derived from continuous-time
signal processing theory rather than RNN practice, and implemented with
hardware-aware convolutions rather than sequential recurrence.

In language modelling benchmarks, Mamba matches or approaches Transformer
quality at a fraction of the inference cost for long sequences. Whether SSMs
can fully replace attention particularly on tasks requiring precise
retrieval of specific facts from a long context, or compositional reasoning
that benefits from all-pairs comparison remains an active and open research
question.

The broader lesson: the competition between attention and its alternatives is
ultimately a competition between different **inductive biases**. Attention
assumes all tokens may be relevant to all others and allocates quadratic
resources accordingly. SSMs assume relevant information can be compressed
into a fixed-size state and process linearly. Neither assumption is
universally correct, which is why hybrid architectures (Transformer layers
interleaved with SSM or MLP layers) are among the most actively investigated
designs today.

---

## Section 7 — Conclusion

{: #section-7}

We began with a single broken sentence and a single broken architecture. The
sentence was _"The poet published many poems but is still grounded"_ a
perfectly ordinary construction that exposed the limit of every seq2seq model
of its era. The architecture was the bottleneck: a single fixed-size vector
forced to carry the entire meaning of a sequence to the decoder.

The fix was to stop discarding the encoder's intermediate states and instead
let the decoder query all of them. That query-and-retrieve operation, dot
product, softmax, weighted sum is attention in its most elementary form.

From that starting point, each improvement revealed the next limitation.
Dot-product attention showed that recurrence could be eliminated entirely
(self-attention). Self-attention showed that a single relationship type was
insufficient (multi-head attention). The full attention matrix showed that
hardware efficiency was the new bottleneck (FlashAttention). The KV cache
showed that inference economics are dominated by how many key-value heads
you maintain (MQA, GQA). And the quadratic compute cost has prompted the
search for fundamentally different architectures (Mamba, linear attention).

This is the pattern of research: each solution exposes the next limitation,
and the limitation points toward the next insight. The mathematics at every
step is not exotic. It is a matrix multiply, a softmax, or a running
maximum. That is exactly the point.

Aravind Srinivas's observation [Srinivas, 2024] was not
a nostalgic appeal to a pre-AI era. It was a prediction about where
competitive advantage will live in an AI-accelerated field. The engineers
who understand _why_ a scaled dot product becomes numerically unstable at
high dimensions, or _why_ tiling allows exact attention without writing the
full matrix to HBM, will be the ones who see the next limitation before the
benchmark announces it. The fundamentals do not become obsolete when the
tools improve rather they become more important.

---

## References

- Ainslie, J., Lee-Thorp, J., de Jong, M., Zeiler, M., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. _EMNLP 2023_.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. _ICLR 2015_.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. _arXiv:2004.05150_.
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. _EMNLP 2014_.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. _NeurIPS 2022_.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. _ICLR 2024_.
- Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. _arXiv:2312.00752_.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. _Neural Computation_, 9(8), 1735–1780.
- Shah, J., Bikshandi, G., Zhang, Y., Diamos, V., Tripuraneni, N., & Dao, T. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. _arXiv:2407.08608_.
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. _arXiv:1911.02150_.
- Srinivas, A. (2024). Perplexity CEO Believes AI Could Bring Computer Science Back to Its Mathematical Roots. _Firstpost_.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. _NeurIPS 2014_.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. _NeurIPS 2017_.
- Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., … & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. _NeurIPS 2020_.
