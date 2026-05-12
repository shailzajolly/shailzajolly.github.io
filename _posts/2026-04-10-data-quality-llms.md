---
layout: post
title: "Data Quality in the Age of LLMs: A Comprehensive Survey of Synthetic Data Generation"
date: 2026-04-10 00:00:00
description: "A survey-style blog post covering definitions, risks, current research, and open challenges in data quality for large language model training and fine-tuning."
tags: [data-quality, llms, synthetic-data, machine-learning, nlp]
categories: research
thumbnail: assets/img/data_quality.png
toc:
  sidebar: left
math: true
---

*A survey-style blog post covering definitions, risks, current research, and open challenges in data quality for large language model training and fine-tuning.*

## Table of Contents

1. [Introduction](#introduction)
2. [What Is Data Quality? Classical Definitions and Dimensions](#what-is-data-quality)
3. [An ML-Centric View: From Databases to Language Models](#ml-centric-view)
4. [Why Data Quality Matters: The Risks of Getting It Wrong](#why-it-matters)
5. [The Synthetic Data Paradigm](#synthetic-data-paradigm)
6. [Current Research Landscape](#current-research)
   - 6.1 [Phi-1: Textbooks Are All You Need](#phi1)
   - 6.2 [FineWeb-Edu and DCLM: Industrial-Scale Curation](#fineweb-dclm)
   - 6.3 [Constitutional AI and RLAIF](#constitutional-ai)
   - 6.4 [Model Collapse: The Curse of Recursion](#model-collapse)
   - 6.5 [Breaking the Curse: Accumulation as Mitigation](#breaking-collapse)
   - 6.6 [Evaluating Quality in Synthetic Data for Tool-Using LLMs](#tool-using)
   - 6.7 [LLMs as Data Generators: AGORABENCH](#agorabench)
   - 6.8 [The LLM Data Auditor: A Metric-Oriented Survey](#data-auditor)
   - 6.9 [Beyond Synthetic Benchmarks: Real-World Code Generation](#beyond-benchmarks)
   - 6.10 [Learning Under Strict Data Constraints: SLlama](#sllama)
7. [Open Challenges and Emerging Solutions](#open-challenges)
8. [Conclusion](#conclusion)
9. [References](#references)

## 1. Introduction <a name="introduction"></a>

There is an old adage in computing: *garbage in, garbage out*. In the era of large language models (LLMs), scaling to hundreds of billions of parameters on trillions of web-scraped tokens has made training data quality one of the most critical and least discussed determinants of model behavior.

The urgency is compounded by a looming supply crisis: high-quality human-generated text is a finite resource. Projections suggest that at current scaling trajectories, we will exhaust the available supply of human-written web text within this decade. Synthetic data generation, using LLMs, to produce training data for other (or the same) models has emerged as the most promising solution to this supply constraint. Yet it introduces its own quality challenges. If we use a model trained on low-quality data to generate more data, do we compound the problem? How do we define, measure, and ensure quality in LLM-generated synthetic data?

This blog post surveys the current state of research on data quality (DQ) in the context of LLMs and synthetic data generation. We ask four foundational questions:

1. **What is data quality** in the context of language models?
2. **Why does it matter**, and what are the risks when it is neglected?
3. **What has the research community done** to address these challenges?
4. **What remains unsolved**, and where should the community focus next?

## 2. What Is Data Quality? Classical Definitions and Dimensions <a name="what-is-data-quality"></a>

Data quality is not a monolithic concept. In its classical database formulation, it is described as *fitness for use* i.e., the degree to which data meets the requirements of its intended application. The seminal work of Wang and Strong (1996) organized data quality into four categories of properties:

- **Intrinsic DQ**: Accuracy, objectivity, believability, reputation
- **Contextual DQ**: Relevancy, value-added, timeliness, completeness, appropriate volume
- **Representational DQ**: Interpretability, ease of understanding, representational consistency, concise representation
- **Accessibility DQ**: Accessibility, access security

Modern treatments of data quality, particularly in the ML and AI context, have refined and extended this taxonomy. A recurring framework organizes data quality around **five key facets**:

| Facet | Description |
|-------|-------------|
| **Accuracy** | Does the data correctly represent the real-world entity or phenomenon it describes? |
| **Completeness** | Is all necessary information present? Are there missing values, fields, or samples? |
| **Consistency** | Is the data free from contradictions, both internally and with respect to other data sources? |
| **Timeliness** | Is the data current and up-to-date for the intended use case? |
| **Uniqueness / Diversity** | Are duplicate records removed? Is the data representative of the full distribution? |

These five facets provide a valuable foundation, but as we will see, they are insufficient on their own when applied to the complex, high-dimensional world of language model training data. A dataset can score perfectly on all five classical dimensions and still produce a poorly-performing or harmful language model.

## 3. An ML-Centric View: From Databases to Language Models <a name="ml-centric-view"></a>

The machine learning perspective on data quality departs from classical database-centric definitions in important ways. In the database world, quality is evaluated against a *schema*; a predefined structure with typed fields and integrity constraints. In machine learning, and especially in NLP, data quality is fundamentally **task-relative**: what counts as high-quality depends on what you want the model to learn.

This ML-centric view identifies several dimensions of quality that are either absent from or underweighted in classical frameworks:

### 3.1 Label Quality and Annotation Consistency

For supervised learning, the quality of the label is as important as the quality of the feature. Annotation errors are the systematic biases introduced by crowdworkers and noisy labels from heuristic labeling pipelines directly corrupt the learning signal. In instruction tuning for LLMs, the "label" is the intended response, and inconsistent or low-quality responses degrade alignment.

### 3.2 Distribution Shift and Representativeness

A dataset may be accurate and complete with respect to its own distribution but fail to represent the diversity of real-world inputs. This is especially problematic in NLP, where the long tail of language—rare linguistic constructions, underrepresented dialects, domain-specific vocabulary is often systematically excluded from training corpora.

### 3.3 Difficulty and Complexity Calibration

For post-training (instruction tuning, RLHF), the *difficulty* of training examples matters enormously. A dataset composed entirely of trivial instructions teaches the model nothing; a dataset with adversarially difficult instructions may be too noisy to learn from efficiently. The right balance of instruction difficulty is a data quality dimension unique to the alignment phase of LLM training.

### 3.4 Noise Tolerance and Learnability

Machine learning models are, to a degree, noise-tolerant. They can learn from imperfect data. But there are thresholds. When noise exceeds a certain level, or when errors are systematic rather than random, model performance degrades sharply. For LLMs, this translates to a concept we can call *learnability* which is the extent to which a training instance enables the student model to improve.

### 3.5 Data Provenance and Ethical Dimensions

Who generated the data? Under what conditions? Does it reflect demographic biases, harmful content, or intellectual property violations? The *provenance* of training data, especially web-scraped text, is increasingly recognized as a data quality dimension with legal and ethical implications.

## 4. Why Data Quality Matters: The Risks of Getting It Wrong <a name="why-it-matters"></a>

Poor data quality in LLM training is not merely an academic concern, rather, it has concrete and observable consequences:

### 4.1 Performance Degradation

The most direct consequence of poor training data is a less capable model. Low-quality instruction-response pairs teach the model incorrect associations, erode generalization, and reduce benchmark scores across all capability dimensions.

### 4.2 Hallucination and Factual Errors

LLMs trained on factually inaccurate or inconsistent data learn to generate text that *sounds* plausible but is factually wrong. This hallucination problem is, at its root, a data quality problem: the model cannot distinguish accurate from inaccurate information if both are equally represented in its training corpus.

### 4.3 Bias Amplification

Training data reflects the biases of its sources. Models trained on web text inherit and can amplify societal biases like gender stereotypes, racial prejudice, political slant that are present in the raw data. Without deliberate curation and quality control, these biases become encoded in the model's parameters.

### 4.4 Safety and Alignment Failures

For alignment-focused post-training, data quality is existential. A model fine-tuned on misaligned demonstrations (responses that violate safety principles or human values) will learn to behave unsafely. The alignment tax paid for poor-quality RLHF data can be catastrophic.

### 4.5 Model Collapse: The Recursive Trap

Perhaps the most alarming consequence of poor synthetic data quality and one that has received significant recent attention is the phenomenon of **model collapse**. As we discuss in detail in Section 6.4, models trained on LLM-generated synthetic data can undergo catastrophic forgetting of the true data distribution, with the tails of that distribution disappearing entirely from the model's learned representation [1]. Importantly, this collapse is a property of the *data replacement regime*, not an inevitable outcome: as Section 6.5 shows, accumulation-based pipelines can avoid it [11].

### 4.6 Benchmark Gaming and Evaluation Distortion

High-quality training data must not contaminate test sets. When synthetic data is generated using models that have been exposed to evaluation benchmarks, or when training data overlaps with test data, benchmark scores become inflated and meaningless. This evaluation distortion is a data quality problem with field-wide implications.

## 5. The Synthetic Data Paradigm <a name="synthetic-data-paradigm"></a>

Human annotation is the gold standard for LLM training data, but it is slow, expensive, and limited in scale. An expert annotator produces dozens of examples per day while an LLM produces millions. This asymmetry has driven an explosion in **synthetic data generation**: using LLMs to produce training data for future (or the same) LLMs.

The synthetic data paradigm encompasses several distinct methodological approaches:

### 5.1 Instance Generation

A small seed dataset of high-quality, human-crafted examples is used as in-context demonstrations. The data generator LLM is prompted to produce new examples that follow the same format and quality standard. The seed dataset is iteratively expanded until the desired volume is reached. This approach pioneered by Self-Instruct [2] and used in systems like Alpaca allows scaling from hundreds of human examples to tens of thousands of synthetic ones.

### 5.2 Response Generation

A large set of instructions (prompts) is collected from diverse sources, but without corresponding responses. A generator LLM is then used to produce high-quality responses for each instruction, creating complete instruction-response pairs. The Magpie approach [3] extends this by extracting instructions directly from aligned LLMs via empty chat templates, requiring no seed prompts whatsoever.

### 5.3 Quality Enhancement

Existing instruction-response pairs of modest quality are systematically improved by prompting a generator LLM to rewrite them with higher complexity, accuracy, or pedagogical value. WizardLM's Evol-Instruct [4] exemplifies this approach, prompting a generator to make existing instructions progressively harder through in-breadth and in-depth evolution operations.

### 5.4 Knowledge Distillation via Synthetic Data

Stronger "teacher" LLMs generate training data for weaker "student" models. This approach, used in systems like Orca [5], has shown that student models can achieve performance approaching their teachers but, as we will see, the relationship between teacher capability and synthetic data quality is more nuanced than it appears.

### 5.5 AI Feedback and Constitutional Generation

LLMs can be used not just to generate data, but to *evaluate and filter* synthetic data according to principled criteria. Constitutional AI [13] formalizes this by having a model critique and revise its own outputs against a set of stated principles, then using the resulting AI-generated preference labels to train a reward model which is a paradigm called RLAIF (Reinforcement Learning from AI Feedback). This approach demonstrates that synthetic preference data, when systematically constructed, can substitute for human annotation in alignment pipelines.

## 6. Current Research Landscape <a name="current-research"></a>

### 6.1 Phi-1: Textbooks Are All You Need <a name="phi1"></a>

**Paper**: Gunasekar et al. (2023), "Textbooks Are All You Need" [12]

One of the most influential demonstrations of synthetic data quality in practice is the **phi-1** language model from Microsoft Research. Phi-1 is a 1.3B-parameter Transformer trained on only ~7B tokens which is dramatically fewer than competing models and yet it achieves 50.6% pass@1 on HumanEval and 55.5% on MBPP, rivaling models an order of magnitude larger.

The key insight is that **the distribution of training data matters far more than its volume**. The training corpus was composed of:

- ~6B tokens of "textbook quality" code-adjacent web text filtered by GPT-4 quality scores
- ~1B tokens of GPT-3.5-generated synthetic Python textbooks, specifically designed to promote reasoning and algorithmic skills

The synthetic textbook component was designed with explicit diversity constraints i.e., varying topic coverage and target audience in the generation prompt to avoid redundancy. The resulting model displays emergent properties, including the ability to solve novel problems not present in its training data.

**Implication for data quality**: Phi-1 established the *quality-over-quantity* principle for pretraining. A carefully curated 7B-token dataset beat models trained on hundreds of billions of tokens. This finding directly motivates the industrial data curation pipelines described next.

### 6.2 FineWeb-Edu and DCLM: Industrial-Scale Curation <a name="fineweb-dclm"></a>

**Papers**: Penedo et al. (2024), FineWeb-Edu [HuggingFace]; Li et al. (2024), DCLM [arXiv:2406.11794]

Phi-1's quality-over-quantity insight prompted the community to ask: *how do we scale principled data curation to trillion-token pretraining corpora?* Two concurrent efforts address this directly.

#### FineWeb-Edu

FineWeb-Edu is a 1.3 trillion token dataset filtered from the 15 trillion token FineWeb corpus (itself built from 96 Common Crawl releases). Its defining innovation is an **educational quality classifier** trained on 460,000 annotations generated by Llama-3-70B-Instruct, which assigns each document a score from 0 to 5. Only documents scoring ≥3 are retained. A 1.82B model trained on 350B FineWeb-Edu tokens outperforms models trained on all of FineWeb, demonstrating that LLM-generated quality labels can be used to effectively curate pretraining data at scale.

#### DCLM (DataComp for Language Models)

DCLM is a systematic benchmark for controlled data curation experiments, providing a standardized 240-trillion-token corpus from Common Crawl, a fixed training recipe, and 53 downstream evaluation tasks. Participants propose curation pipelines like deduplication, filtering strategies, data mixing ratios and the resulting models are compared under identical training conditions. DCLM's key empirical finding: **model-based filtering is the most important component of an effective curation pipeline**, and details of the filtering model have a large impact (ranging from 35% to 44% MMLU 5-shot accuracy at 7B scale). DCLM-Baseline, using a fastText quality classifier trained on OpenHermes-2.5, achieves 64% MMLU 5-shot with a 7B model—competitive with state-of-the-art models while using fewer compute resources.

**Implication**: Both FineWeb-Edu and DCLM demonstrate that classifier-based filtering using LLM-generated quality labels is a practical, scalable, and reproducible solution to pretraining data quality directly addressing the question of how to operationalize "quality" at industrial scale.

### 6.3 Constitutional AI and RLAIF <a name="constitutional-ai"></a>

**Paper**: Bai et al. (2022), "Constitutional AI: Harmlessness from AI Feedback" [13]

Constitutional AI (CAI) from Anthropic introduced a paradigm in which a language model critiques and revises its own outputs against a stated **constitution** a list of explicit principles about helpfulness, harmlessness, and honesty. This self-critique process generates synthetic preference data (which response better follows the principles?) that can then be used to train a reward model via RLHF-style optimization. The resulting paradigm; **RLAIF** (Reinforcement Learning from AI Feedback) is the first large-scale demonstration that synthetic preference data can replace human preference labels in alignment pipelines.

**Why this matters for data quality**: CAI decouples alignment data quality from human annotation throughput. Instead of depending on crowdworkers who may be inconsistent or fatigued, the quality of the synthetic preference data is governed by the constitution's clarity and the model's ability to apply it. This shifts the data quality problem from "did humans label this correctly?" to "is the constitution well-specified, and is the model capable of applying it faithfully?" CAI is now the backbone of Anthropic's Claude model family and has been widely adopted in open-source alignment pipelines.

**Key considerations**:
- Constitutional quality becomes a data quality variable: a poorly written or inconsistent constitution generates noisy preference data
- The approach requires a sufficiently capable base model; very weak models cannot reliably apply constitutional principles
- AI feedback can reflect biases from the model's pretraining distribution, potentially encoding subtle misalignments at scale

### 6.4 Model Collapse: The Curse of Recursion <a name="model-collapse"></a>

**Paper**: Shumailov et al. (2023/2024), "AI models collapse when trained on recursively generated data" [1]

> **Note on citation**: The paper was posted to arXiv in May 2023 (arXiv:1) under the title *"The Curse of Recursion: Training on Generated Data Makes Models Forget"* and published in *Nature* in July 2024 under the revised title *"AI models collapse when trained on recursively generated data."* Both titles refer to the same work.

One of the most striking findings in recent machine learning research is that **models trained on synthetic data generated by other models gradually forget the tails of the original data distribution**; a phenomenon called *model collapse*. Shumailov et al. demonstrate this effect empirically and theoretically across Variational Autoencoders, Gaussian Mixture Models, and Large Language Models.

The mechanism is intuitive once stated: a generator model trained on real data has some error in representing that data. When a new model is trained on synthetic data from this generator, it inherits and compounds this error. Over successive generations under a **data-replacement regime**, where each generation trains only on the previous generation's synthetic output, the representation of rare but important aspects of the original distribution degrades and eventually disappears. The most common patterns dominate; the tails vanish.

**Why this matters for synthetic data quality**: Model collapse establishes a fundamental quality constraint on naively constructed synthetic data pipelines. Without systematic controls, iterative use of a fixed generator causes statistical drift. This finding motivates several lines of active research covered in the next section.

### 6.5 Breaking the Curse: Data Accumulation as Mitigation <a name="breaking-collapse"></a>

**Papers**: Gerstgrasser et al. (2024), "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data" [11]; Kazdan et al. (2024/2025), "Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World" [14]

The Shumailov et al. finding triggered a wave of follow-up work asking whether collapse is truly inevitable. The answer, established both theoretically and empirically, is: **it depends on the training workflow**.

#### The Accumulation Fix (Gerstgrasser et al., 2024)

Gerstgrasser et al. demonstrate that the collapse observed by Shumailov et al. is specific to the *replacement regime* where real data is discarded after each generation and the model trains only on synthetic outputs. When real data is **accumulated alongside synthetic data** (i.e., each generation trains on all previous real and synthetic data combined), a critical theoretical result holds:

- **Replacement**: Test error increases linearly with the number of iterations → collapse
- **Accumulation**: Test error is bounded by a finite constant independent of the number of iterations → no collapse

This result holds across linear models (analytically), VAEs for images, and diffusion models for molecular conformation generation. For language models, the implication is clear: **synthetic data pipelines that retain access to original human data do not suffer model collapse**, even as the proportion of real data in each training batch becomes arbitrarily small.

#### Nuanced Conditions (Kazdan et al., 2024)

Kazdan et al. extend this analysis to study three training workflows—replace, accumulate, and accumulate with fixed compute budget—across three generative settings. Their key finding on the fixed-budget scenario is particularly important for practice: when total dataset size is capped (as it often is in real training pipelines), accumulating synthetic data leads to *slow and gradual* rather than *explosive* degradation across generations. This suggests that practical pipelines operating at fixed compute budgets may experience a milder, manageable form of drift rather than catastrophic collapse.

**Practical synthesis**: The research community now understands model collapse as a spectrum of risk controlled primarily by data workflow:

| Workflow | Collapse Risk | Practical Implication |
|----------|---------------|----------------------|
| Pure replacement (train only on synthetic) | High — linear error growth | Avoid in production pipelines |
| Mixed accumulation (real + synthetic) | Negligible — bounded error | Safe at any synthetic fraction |
| Fixed-budget accumulation | Low-moderate — gradual drift | Monitor across generations; periodic re-anchoring recommended |

### 6.6 Evaluating Quality in Synthetic Data for Tool-Using LLMs <a name="tool-using"></a>

**Paper**: Iskander et al. (2024), "Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs" [6]

Training LLMs to use external tools (APIs, calculators, search engines) requires specialized instruction-response data. Because collecting this data from real human interactions is difficult, synthetic data generation is especially attractive in this domain. Iskander et al. identify a critical gap: **the absence of systematic data quality checks in synthetic tool-use data pipelines causes downstream complications in both training and evaluation**.

The paper proposes two complementary approaches for assessing the reliability of synthetic tool-use data:

1. **Human-defined correctness criteria**: Explicit, interpretable rules derived from the structure of tool-use interactions i.e., checking whether API calls are syntactically valid, whether required parameters are present, whether the response actually uses the tool's output.

2. **Model-driven assessment with in-context evaluation**: Using a capable LLM as a quality judge, providing it with examples of correct and incorrect tool-use chains and asking it to evaluate new synthetic instances.

**Key findings**:
- Naive synthetic data generation for tool use produces a substantial fraction of structurally invalid instances—API calls with missing parameters, responses that hallucinate tool outputs, inconsistent function signatures.
- Both human-defined and model-driven quality filters significantly improve downstream model performance, but the two approaches catch different types of errors.
- **Quality, not quantity, is the primary driver of downstream task performance**: a small set of verified high-quality tool-use examples outperforms a large set of unfiltered synthetic data.

This finding that quality dominates quantity in the synthetic data regime becomes a recurring theme across the research literature and is central to understanding what "data quality" means for LLM post-training.

### 6.7 LLMs as Data Generators: AGORABENCH <a name="agorabench"></a>

**Paper**: Kim et al. (2025), "Evaluating Language Models as Synthetic Data Generators" (ACL 2025) [7]

As multiple frontier LLMs emerge with comparable benchmark performance, a natural question arises: **which LLM is the best data generator?** And crucially, *is the best problem-solver also the best data generator?* Kim et al. address this with AGORABENCH; the first systematic benchmark for evaluating LMs specifically in their role as data generators.

#### The Framework

AGORABENCH evaluates six data generators (GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet, and Llama-3.1-Instruct at 8B, 70B, and 405B scale) across nine experimental settings: three data generation methods × three capability domains.

**Data Generation Methods**:
- *Instance Generation*: Creating new instruction-response pairs from a small seed set
- *Response Generation*: Producing responses for a large set of existing instructions
- *Quality Enhancement*: Refining existing instruction-response pairs to improve quality

**Domains**: Mathematics, Code, and Instruction Following

The key metric is **Performance Gap Recovered (PGR)**—measuring the percentage of the performance gap between a pre-trained base model and a fully instruction-tuned reference model that is recovered by training on the synthetic data:

$$\text{PGR}(G, B) = \frac{\text{score}_B(S_{D_G}) - \text{score}_B(S_\emptyset)}{\text{score}_B(S_\text{ref}) - \text{score}_B(S_\emptyset)} \times 100$$

This is an *extrinsic* metric that directly measures what matters: how much does training on the generated data actually improve the student model?

#### Key Findings

**1. LMs exhibit distinct data generation strengths.**
GPT-4o is the overall best data generator, achieving the highest PGR in five of nine settings. Its advantage is clearest in *instance generation*, where it outperforms all competitors across all three domains. Claude-3.5-Sonnet, meanwhile, excels at *quality enhancement*, demonstrating that the best generator for one method is not necessarily the best for another.

| Data Generator | Instance Gen. Avg | Response Gen. Avg | Quality Enh. Avg |
|----------------|-------------------|-------------------|------------------|
| GPT-4o         | **46.8%**         | **35.2%**         | 6.7%             |
| Claude-3.5-Sonnet | 24.1%          | 28.8%             | **17.9%**        |
| GPT-4o-mini    | 25.3%             | 26.9%             | 5.5%             |
| Llama-3.1-8B   | 22.8%             | 19.4%             | 5.6%             |

**2. Problem-solving ability does not predict data generation ability.**
This is perhaps the most surprising finding. Linear regression between benchmark problem-solving scores and AGORABENCH PGR scores reveals no significant correlation at either coarse or fine granularity (R² < 0.1). In fact, in the code domain, Llama-3.1-70B-Instruct and Llama-3.1-8B-Instruct outperform Claude-3.5-Sonnet and Llama-3.1-405B-Instruct for instance generation—despite being dramatically weaker problem-solvers.

> **Implication**: Using a more capable (and expensive) LLM as a data generator does not guarantee better training data. Task-specific evaluation of data generation capability is necessary.

**3. Intrinsic quality metrics collectively explain data generation ability.**
Since problem-solving ability fails to predict data generation quality, the authors investigate *intrinsic* data properties. A Principal Component Analysis (PCA) over nine intrinsic metrics: response quality (multiple judges), instruction difficulty, response perplexity, and instruction/response diversity—finds that the **top five principal components explain 93.4% of the variance** in PGR scores. All intrinsic metrics contribute approximately equally (loading strengths ranging from 0.189 to 0.256), confirming that data quality is multidimensional and cannot be reduced to any single metric.

**4. Cheap models generating more data can outperform expensive models generating less.**
When scaling generation from 10K to 50K instances, GPT-4o-mini (17× cheaper than GPT-4o) generating 50K instances achieves higher PGR than GPT-4o generating 10K instances in instruction following and math domains. This cost-efficiency finding has direct practical implications for practitioners building synthetic data pipelines.

**5. Meta-prompt quality matters significantly.**
Carefully crafted meta-prompts (developed over 2+ hours of iterative refinement) outperform hastily written ones by ~4% PGR on average. Additionally, free-form generation achieves ~4.45% higher PGR than JSON-structured output—supporting the hypothesis that structured format constraints impair the LLM's reasoning and generative quality.

#### Broader Implications for Data Quality

AGORABENCH establishes that **data quality for LLM post-training cannot be assessed without measuring downstream effect on the student model**. Intrinsic metrics are valuable but must be paired with extrinsic evaluation i.e., the PGR framework or equivalent—to validate what actually matters.

### 6.8 The LLM Data Auditor: A Metric-Oriented Survey <a name="data-auditor"></a>

**Paper**: Zhang et al. (2026), "The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data" [8]

As LLMs have transformed data from a scarce resource into a (theoretically) controllable asset, the research community has accumulated a large but fragmented collection of quality metrics for evaluating synthetic data. Zhang et al. provide the first systematic, metric-oriented survey of this landscape.

#### Two-Dimensional Quality Framework

The paper organizes synthetic data evaluation metrics along two primary axes:

**Axis 1: Quality**
- *Fluency and coherence*: Is the generated text grammatically correct and semantically coherent?
- *Factual accuracy*: Does the content align with verifiable facts?
- *Instruction adherence*: Does the response correctly follow the given instruction?
- *Task-specific correctness*: For specialized domains (math, code, QA), is the answer correct?

**Axis 2: Trustworthiness**
- *Faithfulness*: Does the synthetic data accurately reflect its claimed source or context?
- *Safety*: Is the content free from harmful, biased, or offensive material?
- *Fairness*: Is the data demographically balanced and representative?
- *Privacy preservation*: Does the synthetic data avoid leaking personally identifiable information from the training corpus?

#### Six Modalities of Synthetic Data

The survey documents LLM-based synthetic data generation across six distinct modalities:

1. **Text**: Instruction-response pairs, conversational data, long-form documents
2. **Code**: Programs, unit tests, docstrings, code-explanation pairs
3. **Mathematical reasoning**: Step-by-step solutions, proof sketches
4. **Structured data**: Tables, JSON, database records
5. **Multimodal data**: Image-text pairs, video descriptions
6. **Knowledge graphs**: Entity-relation triples, factual assertions

#### The Evaluation Gap

A central finding of the survey is that **existing research has invested disproportionately in generation methodology while providing limited direct attention to the quality of the resulting data**. The field has many sophisticated generators but relatively few principled evaluation frameworks that apply consistently across modalities, domains, and use cases.

The paper calls for:
- **Standardized quality benchmarks** for synthetic data evaluation
- **Automatic quality scoring pipelines** that go beyond surface-level fluency metrics
- **Trustworthiness auditing tools** that systematically check safety, fairness, and privacy dimensions

This survey provides the conceptual scaffolding for future data quality infrastructure in the LLM ecosystem.

### 6.9 Beyond Synthetic Benchmarks: Real-World Code Generation <a name="beyond-benchmarks"></a>

**Paper**: Rahman et al. (2025), "Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation" [9]

While much of the data quality literature focuses on training data, Rahman et al. highlight a parallel quality problem on the **evaluation** side: standard code generation benchmarks (HumanEval, MBPP) consist primarily of small, self-contained functions that do not reflect the complexity of real-world software development.

The paper introduces a benchmark derived from production-level open-source repositories, specifically targeting **class-level code generation** i.e., implementations that integrate multiple methods, attributes, and dependencies within authentic project contexts. To ensure data quality, they apply rigorous filtering:

- Retaining only projects classified as "engineered software" (not scripts or one-off experiments)
- Separating seen (pre-cutoff) and unseen (post-cutoff) repositories to test genuine generalization
- Verifying functional correctness through test suite execution

#### Key Findings for Data Quality

**1. There is a significant gap between synthetic benchmark performance and real-world code quality.** LLMs that achieve strong scores on HumanEval and MBPP show substantially weaker performance on class-level, multi-file code generation—suggesting that the narrow task distribution of synthetic benchmarks fails to capture the full complexity of real-world programming.

**2. Data quality in evaluation benchmarks has a direct effect on what we think LLMs can do.** Overfit to synthetic benchmarks creates a distorted picture of model capability that leads practitioners to deploy LLMs in situations where they will fail.

**3. Real-world data diversity is essential for both training and evaluation.** The class-level benchmark requires models to handle dependencies, inheritance hierarchies, and cross-file context which are the dimensions of programming language understanding that are systematically absent from function-level synthetic benchmarks.

**Data quality is not only a training data problem**—it extends equally to evaluation data; both training and test sets require careful curation.

### 6.10 Learning Under Strict Data Constraints: SLlama <a name="sllama"></a>

**Paper**: Omolaoye et al. (2025), "SLlama: Parameter-Efficient Language Model Architecture for Enhanced Linguistic Competence Under Strict Data Constraints" (EMNLP 2025) [10]

The SLlama paper explores an extreme data scarcity regime—the BabyLM Challenge—where all models are constrained to 10 million training tokens (roughly what a 5-year-old child hears).

In this extreme data constraint setting, the paper makes a counter-intuitive finding with important implications for data quality research:

#### Embedding Weight Tying Degrades Linguistic Competence

Embedding weight tying i.e., sharing parameters between the input embedding matrix and the output language model head is a standard parameter-reduction technique used widely in small language models. The authors find that this technique **dramatically reduces linguistic competence** under data-constrained training.

The mechanism, drawing on prior theoretical work:
- Weight tying forces the input and output embeddings to share a single representational space
- This shared space aligns more closely to the output embedding (prediction distribution) rather than the input embedding (token identity encoding)
- As a result, the model loses the rich input representations that carry syntactic and morphological information which is exactly the information needed for linguistic competence

**The BLiMP results are striking**: A small untied model with 4.4M parameters achieves 91.9% BLiMP accuracy. A tied model with 2.4M parameters achieves only 56.0%. Note that this comparison involves models of different sizes as well as different tying strategies; the paper's controlled experiments attribute the primary deficit to weight tying rather than parameter count, though both factors contribute.

#### The SLlama Architecture

To preserve parameter efficiency *without* weight tying, SLlama introduces four targeted modifications to the Llama-3 architecture:

1. **Repeated Reduced Hidden Size and Projection (RRHP)**: Reduces the embedding dimension by 4×, then *repeats* the reduced embedding rather than projecting it linearly. This preserves the learned representation without the lossy linear projection.

2. **Permutated Weight Attention (PWA)**: Replaces the standard Q/K/V weight matrices with a single shared parameter matrix indexed by different permutations, reducing attention parameters substantially.

3. **Shared Projection MLP (SPMLP)**: Ties the weights of the MLP expansion and reduction layers (using the transpose relationship), saving parameters per layer.

4. **Layer Weight Sharing**: Shares weights across groups of decoder layers, drastically reducing total parameter count.

With 2.6M parameters and 20× fewer than the Baby Llama baseline; SLlama achieves 91.94% BLiMP accuracy (31.72% improvement over baseline) without any knowledge distillation.

#### The Deeper Lesson for Data Quality

SLlama's results illuminate a principle that applies broadly: **when data quality or quantity is constrained, architectural choices that seem neutral in the data-rich regime can have catastrophic effects on learning**. The model's capacity to absorb and retain high-quality signals from limited data depends critically on its representational structure.

This is a data quality concern at the architecture-data interface: even with a fixed dataset, the "quality" of signal that a model can extract depends on whether its architecture is well-suited to the data regime. For researchers building synthetic data pipelines, this suggests that:

- Synthetic data quality evaluation must be architecture aware
- Small, resource-constrained models require especially high data quality i.e., fewer parameters means less noise tolerance
- Parameter efficiency and linguistic expressivity are not freely exchangeable

SLlama also demonstrates impressive out-of-domain generalization: despite being trained on child-directed speech, it outperforms much larger models on MMLU (general world knowledge) when evaluated relative to the same training data constraint.

## 7. Open Challenges and Emerging Solutions <a name="open-challenges"></a>

The research surveyed above, while substantial, leaves a rich landscape of unsolved problems. For each challenge, we summarize the state of existing solutions where they exist, and identify the remaining open question.

### Challenge 1: A Unified Theory of Synthetic Data Quality

The field lacks a principled, unified theory that explains *what makes synthetic data good*. The AGORABENCH finding that multiple intrinsic metrics collectively explain 93.4% of PGR variance (but no single metric dominates) suggests that quality is multidimensional but the precise relationships between these dimensions, and how they interact with model architecture and training objectives, remain poorly understood [7].

**Existing work**: The LLM Data Auditor [8] provides a taxonomy; AGORABENCH [7] provides an empirical characterization via PCA. IFD (Instruction Following Difficulty) scoring offers a single-dimensional heuristic for instruction selection.

**Open question**: Can we derive a theoretical framework analogous to information theoretic bounds in learning theory that characterizes the quality requirements for synthetic data as a function of the desired downstream task and model architecture?

### Challenge 2: Preventing Model Collapse at Scale

The Curse of Recursion [1] demonstrates that iterative training on synthetic data under a replacement regime causes distributional collapse. The research community now has a robust solution for the worst-case scenario; data accumulation [11] but important practical questions remain about how to implement this in real pipelines where total data budgets are constrained.

**Existing work**: Gerstgrasser et al. [11] prove that accumulation prevents collapse theoretically and empirically. Kazdan et al. [14] show that fixed-budget accumulation causes only gradual drift. Verification pipelines (e.g., arXiv:2510.16657) use independent models to validate synthetic distribution coverage.

**Open question**: What are the optimal real to synthetic mixing ratios as a function of domain, model scale, and generation quality? Can adaptive mixing strategies that monitor distributional drift and adjust the real data anchor in real time prevent gradual fixed-budget drift indefinitely?

### Challenge 3: Evaluating Trustworthiness at Scale

The LLM Data Auditor survey [8] documents a systematic gap between quality metrics and trustworthiness metrics. While fluency and factual accuracy have received significant attention, dimensions like privacy preservation, demographic fairness, and safety in synthetic data pipelines remain poorly instrumented.

**Existing work**: Constitutional AI [13] provides a scalable mechanism for safety-focused filtering. Automated red-teaming frameworks (e.g., Perez et al., 2022) use LLMs to generate adversarial prompts that expose safety failures. LLM-as-judge frameworks (MT-Bench, Alpaca-Eval) offer scalable quality evaluation, though they primarily target quality rather than trustworthiness.

**Open question**: How do we build automated, scalable trustworthiness auditing tools that can evaluate millions of synthetic training examples for safety, fairness, and privacy violations before model training begins?

### Challenge 4: Cross-Architecture Generalization of Data Quality

AGORABENCH uses Llama-3.1-8B as the student model for all experiments [7]. SLlama highlights that different architectures have fundamentally different data quality requirements [10]. Yet virtually all synthetic data quality research optimizes for a single architecture or family.

**Existing work**: SLlama [10] and the broader BabyLM literature characterize how data constraints interact with architecture in small-model regimes. FineWeb-Edu and DCLM provide architecture-agnostic pretraining benchmarks, but post-training data quality remains understudied across architectures.

**Open question**: Are high-quality synthetic datasets architecture agnostic, or must data quality be evaluated and optimized separately for each target architecture? Can we design "universally high-quality" synthetic data that improves all model architectures?

### Challenge 5: The Generator-Student Capability Gap

AGORABENCH reveals that weaker LMs can sometimes produce better training data than stronger ones [7]. Meanwhile, research on knowledge distillation shows that teachers that are *too* capable relative to their students can be ineffective because the learning signal is too far from the student's current ability.

**Existing work**: AGORABENCH [7] empirically documents the non-monotonic relationship between generator capability and data generation quality. Orca [5] demonstrates that chain-of-thought traces from GPT-4 can effectively teach smaller models, suggesting that *explanation style*, not just answer quality, mediates the capability gap.

**Open question**: Is there a principled way to select the optimal generator model for a given student model? Can we formalize a "Goldilocks" principle for generator-student capability matching?

### Challenge 6: Domain-Specific Quality Standards

Most data quality research operates at the level of general instruction tuning. But specialized domains like medicine, law, scientific reasoning, code have domain-specific quality criteria that general metrics fail to capture.

**Existing work**: Phi-1 [12] demonstrates domain-specific curation for code. Domain-specific benchmarks like MedPALM's evaluation suite, LegalBench, and SciEval provide evaluation infrastructure. DCLM [15] establishes that model-based filtering generalizes across domains, but the filtering classifiers themselves require domain-specific training data.

**Open question**: How do we develop and maintain domain-specific data quality standards and automated verification tools for high-stakes application domains particularly those where ground truth is contested (law, clinical medicine) or rapidly evolving (scientific research)?

### Challenge 7: Evaluation Benchmark Quality

Rahman et al. [9] demonstrate that evaluation benchmarks themselves suffer from data quality problems. The community's reliance on narrow synthetic benchmarks creates a distorted picture of model capability. Yet constructing truly representative, high-quality evaluation benchmarks is expensive and time-consuming.

**Existing work**: Rahman et al. [9] provide a production-code benchmark. LiveCodeBench uses temporally-isolated problems to prevent contamination. BIG-Bench Hard and MATH provide harder evaluation sets. Contamination detection methods (Min et al., 2023) help identify training-test overlap post hoc.

**Open question**: How do we systematically assess and improve the quality of evaluation benchmarks, and how do we ensure that new benchmarks remain uncontaminated as LLMs are increasingly used in their construction?

### Challenge 8: Diversity-Quality Trade-offs in Constrained Settings

SLlama shows that in data-constrained settings, architectural choices dominate data quality effects [10]. But it remains unclear how data diversity should be prioritized relative to individual example quality when total data volume is limited.

**Existing work**: Phi-1 [12] uses diversity constraints in synthetic textbook generation. DCLM [15] uses embedding-space deduplication and MinHash-based near-deduplication to maximize diversity. Coverage-based sampling methods (e.g., SemDeDup) cluster embeddings and sample uniformly across clusters to ensure distribution coverage.

**Open question**: What is the optimal diversity-quality trade-off in synthetic data generation as a function of total data budget? Can we derive adaptive sampling strategies that jointly maximize coverage and per-example quality?

### Challenge 9: Transparency and Reproducibility in Synthetic Data Pipelines

The provenance and quality of synthetic data used in commercial LLM training is rarely disclosed. This opacity makes it impossible for the research community to understand, replicate, or improve upon these pipelines. Furthermore, proprietary data generators (GPT-4o, Claude-3.5-Sonnet) raise concerns about intellectual property, usage rights, and the independence of evaluation [7].

**Existing work**: FineWeb-Edu and DCLM release both datasets and curation code openly. AGORABENCH [7] is fully reproducible with open student models. Statistical watermarking research (Kirchenbauer et al., 2023) offers tools for detecting synthetic data provenance in deployed content.

**Open question**: How do we establish norms and infrastructure for transparent synthetic data reporting including model cards for synthetic datasets, standardized provenance metadata, and contamination disclosure that enables reproducibility without compromising legitimate proprietary interests?

## 8. Conclusion <a name="conclusion"></a>

The research surveyed in this post converges on a small set of durable principles that cut across pretraining, post-training, and evaluation.

**Quality dominates quantity.** The clearest empirical lesson of the past three years is that the distribution of training data matters far more than its volume. Phi-1 [12], FineWeb-Edu, DCLM [15], and the tool-use work of Iskander et al. [6] all demonstrate, in different regimes, that a carefully curated small dataset consistently outperforms a large undifferentiated one. This finding is now sufficiently well-established to be treated as a design constraint rather than a hypothesis.

**Synthetic data is viable but requires deliberate engineering.** The field has moved past the naive assumption that LLM-generated data is either universally good or universally dangerous. The evidence shows it is neither: model collapse is real [1] but avoidable through data accumulation [11]; generator capability does not predict data quality [7]; and the best generator for one task (GPT-4o for instance generation) is not the best for another (Claude-3.5-Sonnet for quality enhancement) [7]. Effective synthetic data pipelines require deliberate choices about generation method, generator selection, filtering strategy, and mixing ratio—none of which can be defaulted to safely.

**Evaluation quality is as important as training data quality.** Rahman et al. [9] make the uncomfortable point that the benchmarks the field uses to measure progress are themselves a data quality problem. Narrow, function-level synthetic benchmarks systematically overestimate model capability on real-world tasks. Improving training data quality while leaving benchmark quality unchanged means measuring the wrong thing more accurately.

**Trustworthiness remains a second-class citizen.** The LLM Data Auditor [8] documents a persistent imbalance: the field has invested heavily in fluency and accuracy metrics, while safety, fairness, and privacy dimensions of synthetic data remain poorly instrumented. As synthetic data pipelines scale to industrial volumes, this gap will have increasingly concrete consequences. Constitutional AI [13] provides a path forward for safety-focused data generation, but scalable trustworthiness auditing tools across all dimensions do not yet exist.

**Architecture and data quality are coupled.** SLlama [10] establishes that data quality is not an abstract property of a dataset. It is always relative to the architecture consuming it. A design choice that is neutral in the data-rich regime (embedding weight tying) can be catastrophic in the data-scarce regime. This coupling implies that the field cannot optimize data quality in isolation from model design, particularly as practitioners deploy LLMs in resource-constrained settings.

The open challenges in Section 7 are not peripheral research curiosities; they are the engineering blockers that limit how much of the quality-over-quantity principle can be realized in practice. A unified theory of synthetic data quality, cross-architecture generalization, domain-specific standards, and transparent provenance infrastructure are the infrastructure investments the field needs to build on the empirical progress of the past few years.

We began with *garbage in, garbage out*. The research summarized here suggests a more nuanced restatement: *the right data, in the right quantities, from the right sources, filtered by the right criteria, accumulated rather than replaced, and evaluated against representative rather than convenient benchmarks, produces models that genuinely reflect what we asked them to learn*. The difficulty is that every clause of that sentence is an open research problem. That is not a pessimistic conclusion; it is a research agenda.

## 9. References <a name="references"></a>

[1] **Shumailov, I., et al. (2023/2024).** "AI models collapse when trained on recursively generated data." *Nature* (2024); arXiv preprint "The Curse of Recursion: Training on Generated Data Makes Models Forget," *arXiv:2305.17493* (2023). — Demonstrates model collapse under data-replacement regimes across VAEs, GMMs, and LLMs.

[2] **Wang, Y., Kordi, Y., et al. (2023).** "Self-Instruct: Aligning Language Models with Self-Generated Instructions." *ACL 2023*. — Foundational work on instance generation: expanding small seed datasets via LLM-generated instruction-response pairs.

[3] **Xu, Z., Jiang, F., et al. (2024).** "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing." *arXiv:2406.08464*. — Response generation approach: eliciting instructions from aligned LLMs via empty chat templates.

[4] **Xu, C., Sun, Q., et al. (2024).** "WizardLM: Empowering Large Pre-trained Language Models to Follow Complex Instructions." *ICLR 2024*. — Quality enhancement via Evol-Instruct: prompting LLMs to progressively increase instruction complexity.

[5] **Mukherjee, S., et al. (2023).** "Orca: Progressive Learning from Complex Explanation Traces of GPT-4." *arXiv:2306.02707*. — Knowledge distillation via synthetic data: using GPT-4 chain-of-thought traces to train smaller student models.

[6] **Iskander, S., et al. (2024).** "Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs." *arXiv:2409.16341*. — Proposes human-defined and model-driven quality filters for synthetic tool-use training data; shows quality dominates quantity.

[7] **Kim, S., Suk, J., Yue, X., Viswanathan, V., et al. (2025).** "Evaluating Language Models as Synthetic Data Generators." *ACL 2025* (acl-long.320). — AGORABENCH: systematic benchmark comparing LMs as data generators across methods and domains; reveals that problem-solving ability does not predict data generation quality.

[8] **Zhang, K., et al. (2026).** "The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data." *arXiv:2601.17717*. — Comprehensive metric survey covering quality and trustworthiness dimensions of synthetic data across six modalities.

[9] **Rahman, M., et al. (2025).** "Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation." *arXiv:2510.26130*. — Demonstrates the quality gap between narrow synthetic benchmarks and real-world class-level code generation.

[10] **Omolaoye, V.A., Owoyele, B.A., de Melo, G. (2025).** "SLlama: Parameter-Efficient Language Model Architecture for Enhanced Linguistic Competence Under Strict Data Constraints." *EMNLP 2025* (emnlp-main.1198). — Introduces SLlama; demonstrates that embedding weight tying degrades linguistic competence in small models; achieves 31.72% BLiMP improvement with 20× fewer parameters.

[11] **Gerstgrasser, M., Schaeffer, R., et al. (2024).** "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data." *arXiv:2404.01413*. — Proves theoretically and demonstrates empirically that data accumulation (vs. replacement) prevents model collapse, bounding test error independently of iteration count.

[12] **Gunasekar, S., Zhang, Y., et al. (2023).** "Textbooks Are All You Need." *arXiv:2306.11644*, Microsoft Research. — Introduces phi-1; demonstrates that 1.3B model trained on ~7B "textbook quality" tokens rivals models trained on hundreds of billions of tokens on code generation.

[13] **Bai, Y., Jones, A., et al. (2022).** "Constitutional AI: Harmlessness from AI Feedback." *arXiv:2212.08073*, Anthropic. — Introduces Constitutional AI (CAI) and RLAIF: using LLMs to generate synthetic preference data governed by a stated constitution, enabling alignment without human annotators.

[14] **Kazdan, J., et al. (2024).** "Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World." *arXiv:2410.16713*. — Studies three data training-workflows (replace, accumulate, fixed-budget accumulate) and shows that accumulation avoids collapse while fixed-budget accumulation causes only gradual drift.

[15] **Li, J., et al. (2024).** "DataComp-LM: In search of the next generation of training sets for language models." *arXiv:2406.11794*. — Introduces DCLM: a benchmark with 240T-token corpus and 53 downstream tasks for controlled data curation experiments; establishes model-based filtering as the key component of effective curation pipelines.
