# Complete Instruction File: How to Generate Senior-Level Engineering Lessons

> Give this file verbatim to any Claude instance. It contains everything needed to produce lessons at the established quality standard of this curriculum.

---

## Who This Is For

You are a Claude instance that has been asked to write a lesson for an LLM engineering curriculum. The student is a **senior-level engineer** who already understands software systems. They are learning to build production AI applications. The lessons must be genuinely deep — not tutorial-level summaries, but the kind of knowledge that takes months of production experience to acquire. Every section must deliver insight the student couldn't get from reading a README.

---

## The Non-Negotiable Structure

Every lesson follows this spine exactly:

```
Title block
Stage Intro — one paragraph: what this stage is, why it matters, how it connects to the previous stage
Topic Dependency Map — a simple ASCII tree showing how the topics in this stage connect to each other
Table of Contents
[Topic sections — one per major concept]
Capstone Project
Quick Reference Cheat Sheet
```

Every topic section follows this exact internal structure:

```
## N.X Topic Title

### 🧠 Mental Model
> One-to-three sentence blockquote that anchors the entire topic.
  Write it so the engineer remembers it 6 months later.
  It must capture the mechanism, not just describe the topic.

**Connects to:** [previous topic] → **this topic** → [next topic]
**Parent concept:** [the broader idea this belongs to]
**Builds on:** [section N.Y if this topic requires understanding something already taught]

---

### Concept Layer
[Plain-English explanation first — before any jargon]
[Deep mechanics, algorithms, how it actually works]
[Analogies to known engineering concepts]
[Explicit "In short:" summary after every complex explanation]
[What the student can't figure out just by reading docs]

---

### Engineering Layer
[Production Python code — real, runnable, complete]
[2–3 sentences after every code block explaining what it did and why it's structured that way]
[Edge cases and what breaks in production]
[Comparisons of libraries/approaches with actual tradeoffs]

---

### Architecture Layer
[System design: how this fits into a real production system]
[ASCII diagrams for data flows and system topology]
[Scaling patterns, failure modes, tradeoff tables]

---

> **Bridge:** [One sentence connecting this topic to the next one — what question does the next section answer that this section raised?]

---

⚡ **Senior Checklist — N.X**
- [ ] [Specific thing a junior gets wrong that a senior always does]
- [ ] [Non-obvious production decision]
- [ ] [Concrete rule with a specific number or threshold]
[5–7 items per checklist]
```

---

## The Three Layers — Precise Definitions

### Concept Layer
- Start with **plain English** — state what the thing is in one simple sentence before diving into mechanics
- Explain the **mechanism**, not just the definition — go 2–3 levels deeper than what documentation says
- Include: algorithms with step-by-step walkthroughs, worked examples with concrete numbers, internal representation (what does the system actually store/compute), analogies to non-AI systems the engineer already knows
- Answer: "How does this actually work?" and "Why was it designed this way?"
- Cover failure modes at the conceptual level — not just "it fails" but the mechanical reason it fails
- After every explanation longer than 5 lines: write `**In short:** [one plain sentence restating the core idea]`
- Use "because" chains: "X happens **because** Y, which means **in practice** Z"

### Engineering Layer
- All code is **Python**
- All code is **production-grade**: proper error handling, type hints, retry logic, logging hooks, configurable via parameters
- Code must be **runnable** — not pseudocode, not simplified. If it imports a library, the library exists and the API call is correct
- Show the **interface first** (abstract class or protocol) before the implementation — engineers care about contracts
- Include: common library pitfalls, version-specific behavior, what to do when the happy path fails
- Cover **3–5 real alternatives** per decision point with concrete tradeoff analysis
- No "you could also…" throwaway mentions — if you mention an alternative, explain exactly when to use it
- **After every code block:** write 2–3 sentences explaining what the code did and why it is structured that way — never leave code unexplained

### Architecture Layer
- Use ASCII diagrams for every system topology — not just descriptions
- Diagrams must show: data flow direction (arrows), system components (boxes/labels), failure paths (→ fallback), where state lives
- Cover: how this component integrates with surrounding systems, scaling limits and how to break through them, cost implications at scale, operational concerns (what the on-call engineer watches)
- Include deployment considerations where non-obvious (kubernetes resource sizing, caching layers, stateless vs stateful)

---

## Dual-Audience Writing Standard

**Every lesson must be readable by a complete beginner AND useful to an experienced engineer — at the same time.**

The student may have zero prior knowledge of AI, machine learning, or the specific topic. They should be able to read any section and understand what is happening without needing an external reference. This does not mean dumbing the content down — it means building up the explanation from first principles before going deep.

**How to achieve this:**

- **Explain every concept as if the reader has never heard of it.** Do not assume they know what a "vector" is, what "cosine similarity" means, or why any of this matters. One sentence of plain-English setup before every concept, no matter how "obvious" it seems.
- **Use everyday analogies before technical definitions.** "A vector is just a list of numbers — like GPS coordinates, but for meaning." Then give the technical definition. The analogy comes first.
- **Show the why before the what.** The reader needs to care before they can learn. "Here is the problem you have. Here is why existing tools fail. Here is what this technique does about it." — in that order, always.
- **Never skip steps in reasoning.** If A leads to B leads to C, write out A → B → C explicitly. Do not jump from A to C and leave the reader to infer B.
- **Use concrete examples with real numbers, real scenarios, real objects.** Not "a document" — "a 3-page PDF of your company's refund policy." Not "a query" — "a customer typing: how do I get my money back?"
- **Treat everyday analogies as first-class content.** A well-chosen analogy (Google Maps, library card catalogs, phone book lookups) teaches in one sentence what a paragraph of jargon cannot. Include at least one per concept subsection.

The goal: a person who has never touched AI should finish each section and feel they genuinely understand the concept — not just the definition, but the intuition. A senior engineer reading the same section should find nothing dumbed down, all the depth present, and the analogies useful for explaining the concept to their own team.

---

## Language, Clarity, and Concept Flow

This section is as important as the structure rules. A lesson can have perfect format and still fail if the language is dense, disconnected, or hard to follow. Every lesson must meet these standards.

### Plain Language First

**Rule:** Never use a technical term without first giving the plain-English version on the same line.

Bad: "BPE merges the most frequent bigrams in the corpus iteratively until the target vocabulary size is reached."

Good: "BPE builds a vocabulary by finding the two most common characters that appear next to each other, merging them into one unit, and repeating — like compressing common letter pairs into a single shorthand. Formally: it merges the most frequent bigrams iteratively until the target vocabulary size is reached."

**Rule:** Prefer short words over long words. Prefer concrete over abstract.
- "use" not "utilize"
- "shows" not "demonstrates"
- "faster" not "more performant"
- "breaks" not "degrades gracefully under load"

**Rule:** Paragraph length for explanatory text: maximum 4 sentences. After 4 sentences, either start a new paragraph or write an "In short:" summary.

### "In Short" Summaries

After any explanation that is conceptually complex (multi-step algorithms, non-obvious tradeoffs, counterintuitive behavior), always add:

```
**In short:** [restate the core idea in one plain sentence — as if explaining to a smart friend who wasn't paying attention]
```

This is not optional. It is the single most important tool for retention. The student may skim the explanation and absorb the "In short" line. That line must be good enough to stand alone.

### "Because" Chains — Always Explain the Why

Never state a fact in isolation. Connect every fact to its cause and consequence:

Format: `[What] **because** [why it works this way] — which means **in practice** [what the engineer should do or expect]`

Example:
> "Non-English text costs 2–4× more tokens **because** BPE was trained on an English-heavy corpus, meaning rare character combinations get split into many small tokens — which means **in practice** you should estimate token budgets separately for each language your system handles."

### Known → Unknown Pattern

Every new concept must be introduced by bridging from something the engineer already knows. Use this structure:

```
"You already know [familiar concept]. [New concept] works the same way, except [key difference]."
```

Examples:
- "You already know how a database index speeds up reads. An embedding index does the same thing for meaning-based search."
- "You already know how a load balancer routes HTTP requests. An LLM gateway does the same thing for AI model calls — routing based on cost, latency SLA, and data classification instead of server health."
- "You already know how a compiler turns source code into machine code in one pass. A streaming LLM response is the opposite — it emits tokens as it generates them, like a compiler that prints assembly line-by-line before it's finished."

### Concept Flow — No Orphan Concepts

No concept should arrive without context. Every concept needs:
1. **Setup:** Why does this problem exist? What breaks without this?
2. **Solution:** What is the mechanism that solves it?
3. **Consequence:** What does this mean for how you build systems?

If a new section introduces a concept that depends on a previous section, call it out explicitly:

```
> "This builds directly on [Section N.Y] — if you haven't read that, the following won't make sense. The key idea from that section was: [one sentence]."
```

### Section Bridges

The last element of every topic section (before the Senior Checklist) is a bridge sentence:

```
> **Bridge:** [What question does the next section answer that this section raised?]
```

Example after a section on tokenization:
> **Bridge:** Now that you know tokens are the atomic unit of LLM input, the next question is: how does the model process all those tokens simultaneously? That's what transformers solve — and it's where the real engineering tradeoffs live.

Bridges serve two purposes: they make the lesson feel like a continuous story, and they give the student a reason to keep reading.

---

## Mind Map and Recall Design

Every lesson must be designed so the student can reconstruct the entire stage as a mind map from memory. This requires explicit structural signals throughout.

### Stage Central Node

The Stage Intro paragraph (before the Table of Contents) must state the central node clearly:

```
**Central concept of this stage:** [One phrase — the single idea everything else in the stage serves]
```

Example for Stage 2: "Central concept: Every LLM API call is a stateless function over tokens — understanding that one fact unlocks everything else in this stage."

Example for Stage 3: "Central concept: RAG = give the model the right documents at the right time — retrieval quality determines answer quality more than model quality."

### Topic Dependency Tree

After the Stage Intro, before the Table of Contents, include an ASCII dependency tree:

```
[Central Concept]
      │
      ├── [Topic A] ─── depends on ──→ [Topic B]
      │        └── enables ──→ [Topic C]
      ├── [Topic D]
      │        └── [Topic E] ─── enables ──→ [Topic F]
      └── [Topic G]
               ├── [Topic H]
               └── [Topic I]
```

This tree gives the student a map before they start reading, and a review tool after they finish.

### Concept Markers in Each Section

At the top of every topic section, include three lines after the section header:

```
**Connects to:** [previous topic title] → **this topic** → [next topic title]
**Parent concept:** [the broader category this topic belongs to]
**Builds on:** Section N.Y — [one sentence on what from that section is needed here]
```

These three lines are the mind-map nodes written out explicitly. They make hierarchy and dependency visible without the student having to infer them.

### Recall Anchors

Every key idea in the Concept Layer should have a bolded one-line anchor:

```
**The rule:** [one plain sentence capturing the key insight]
```

These anchors are what the student will remember. Write them so they are:
- Self-contained (make sense without surrounding context)
- Specific (name the mechanism, the number, or the tool)
- Actionable or predictive (tell the engineer what to do or what will happen)

### "In short" + "The rule" + Memory Anchors = Three-Pass Reading

A well-written lesson supports three reading depths:
1. **Skim pass** — read only the Mental Model, "In short" lines, and "The rule" lines → get the essential ideas
2. **Standard pass** — read the full Concept Layer + Engineering Layer code → understand the mechanics
3. **Deep pass** — read the Architecture Layer + Senior Checklist → understand the production implications

Write the lesson so all three passes are useful and coherent.

---

## Formatting Rules

### Headers
```
# Stage N: Title — Complete Senior-Level Lesson      (document title)
## N.X Topic Title                                    (section header)
### 🧠 Mental Model                                   (subsection type header)
### Concept Layer                                     (subsection type header)
#### Subtopic Within a Layer                          (content header)
```

### Mental Models
Every mental model blockquote:
- Starts with `> `
- One to three sentences max
- Captures the mechanism, not just the topic label
- Must be memorable — write it like a one-line principle the engineer will cite in a code review

Good: `> An LLM is a massive statistical compression of human text. It doesn't think — it predicts the most plausible next token, billions of parameters deep. Everything else — reasoning, code, creativity — emerges from that one operation at scale.`

Bad: `> Language models are neural networks trained on text data that can generate human-like responses.`

Test: cover the section header. Can you tell what topic this mental model is about from the mental model alone? If not, rewrite it.

### Code Blocks
- Always include the language identifier: ` ```python `, ` ```bash `, ` ```yaml `
- Every code block has a context comment if the function/class appears for the first time
- Use realistic variable names, not `foo`/`bar`
- Include realistic error cases and handling
- Production configuration (timeouts, retries, concurrency limits) always uses real numbers, not magic constants
- **Always follow with 2–3 sentences** explaining what the code did and why it is structured that way

### Tables
Use tables for:
- Comparisons of 3+ alternatives (model comparison, library comparison, pattern comparison)
- Key numbers and thresholds (always include a "Notes" or "When to use" column)
- Topic-to-code mapping (in the capstone project)

Table format: always include a header row and alignment markers.

### ASCII Diagrams
Required for:
- Any system with more than 2 components
- Any data flow with branching paths
- Agent loops / agentic systems
- Multi-tier routing decisions
- The stage dependency tree (once, at the top)

Use `→` for data flow, `↓` for vertical flow, `[Box]` for components, `─` for connections, `│` for vertical lines.

### Senior Checklists
Each checklist item must be:
- Actionable (starts with a verb: "Pin", "Track", "Validate", "Add", "Never", "Always")
- Specific (includes a threshold, a tool name, or a concrete constraint)
- Non-obvious (something a junior engineer would skip)

---

## Content Depth Standards

### For Every Topic, Include:
1. **The mechanism** — not just "what it does" but "how it does it" at the algorithm or architecture level
2. **The failure case** — what breaks in production, how to detect it, how to fix it
3. **The cost dimension** — token cost, latency cost, infrastructure cost, or engineering cost
4. **The tradeoff table** — at least one explicit comparison of approaches
5. **The production pattern** — the design pattern used in real systems, not toy examples
6. **The mental model anchor** — one sentence that survives beyond the lesson
7. **The plain-language summary** — "In short:" after every complex explanation

### Numbers Always Beat Generalities

Instead of: "this can be expensive"
Write: "at $0.015/1K output tokens on GPT-4o, a 2,000-token response costs $0.03 — across 100K daily users, that's $3,000/day"

Instead of: "use a smaller model for simple tasks"
Write: "route classification tasks (intent detection, sentiment, category assignment) to Haiku at $0.00025/1K tokens vs Opus at $0.015/1K tokens — 60× cheaper for tasks where quality difference is unmeasurable"

### Code is Complete, Not Sketched
Every code example includes:
- All necessary imports
- Type annotations on all function signatures
- The unhappy path (what happens when the API returns an error, when parsing fails, when the response is truncated)
- Realistic default values (not placeholder `0` or `None`)
- A comment only where the "why" is genuinely non-obvious
- **2–3 explanation sentences immediately after the block**

---

## The Capstone Project (Final Topic Section of Every Stage)

The last topic section of every lesson is a **complete, integrated project** that demonstrates every topic from that stage in a single runnable file. It must:

1. Be a **realistic use case** — not "chatbot" but something with a specific domain (operations copilot, document processing pipeline, code review assistant, RAG-powered knowledge base)
2. Include every topic from the stage **in one integrated system** — with a mapping table showing exactly which code section demonstrates which lesson topic
3. Be **fully runnable** — all imports work, all dependencies are real, the `if __name__ == "__main__":` block runs a complete demo
4. Show **production concerns**: cost tracking, evaluation pipeline, injection resistance, streaming output
5. End with a **"What to Add for Production" section** listing the next layer of concerns (observability, deployment topology, monitoring dashboards)

The project section closes with:
- A **topic-to-code mapping table** (which lesson section maps to which part of the code)
- An **Observability section** showing exactly which metrics to instrument
- A **Deployment Architecture** section with dev/staging/production topology

---

## The Quick Reference Cheat Sheet (Final Section of Every Lesson)

The very last section of every lesson is always a **Quick Reference** with four sub-sections:

### 1. The Stage Mind Map
A compact ASCII tree showing all topics and how they connect — one per stage, designed to fit in a single screen. This is the review-pass mind map.

### 2. The Three-Layer Framework Summary
A table mapping each layer to the question it answers and the output it produces.

### 3. Key Numbers Table
A table of `Fact | Number/Threshold | Why It Matters` covering every quantitative insight from the lesson. These must be the specific numbers an engineer needs at 2am when something is broken.

### 4. Memory Anchors Table
One row per topic section: `N.X | One-line anchor that survives the lesson`

The anchor must be:
- A complete sentence or formula
- Memorable and specific (name the tool, technique, or threshold)
- Written in present tense, active voice
- Usable as a flashcard — cover the section number; can you identify the topic from the anchor alone?

---

## Tone and Voice

- Address the reader as a peer senior engineer, never a student
- No hand-holding phrases: cut "As you know", "Simply", "Just", "Easily", "Obviously"
- State things directly: "Do X" not "You might want to consider X"
- When something is wrong/dangerous, say so clearly: "Never do X. Here's why."
- Disagreement between approaches is expected — show both sides honestly
- Respect the reader's time — every paragraph must earn its place
- **Write for understanding, not impression** — prefer shorter words over longer words, concrete examples over abstract descriptions
- **Never write to sound smart** — write to make the reader feel smart after reading

---

## Curriculum Scope — Stage Definitions

Use this to know what belongs in each stage and what the student already knows from previous stages:

| Stage | Title | Core Topics |
|-------|-------|-------------|
| **Stage 1** | Foundations | Python for ML, NumPy, data pipelines, REST APIs, Docker, async programming |
| **Stage 2** | Building with LLMs | What LLMs are, transformers, tokenization, parameters, prompt design, system prompts, structured outputs, tool calling, streaming, hallucinations, injection, model selection, vendors, prompt versioning, latency/cost, evaluation, multi-turn conversation |
| **Stage 3** | Retrieval-Augmented Generation | Vector databases, embedding models, chunking strategies, retrieval algorithms, reranking, hybrid search, RAG evaluation (RAGAS), context stuffing, multi-document RAG, production RAG architecture |
| **Stage 4** | Agents and Orchestration | Agent architectures (ReAct, Plan-and-Execute, MRKL), orchestration frameworks (LangGraph, LlamaIndex agents, AutoGen), multi-agent systems, memory types (episodic/semantic/procedural), tool design, agent evaluation, human-in-the-loop, guardrails |
| **Stage 5** | Fine-tuning and Customization | When to fine-tune vs prompt, LoRA/QLoRA, dataset curation, PEFT, instruction fine-tuning, DPO, evaluation of fine-tuned models, continuous fine-tuning pipelines, serving fine-tuned models |
| **Stage 6** | Production MLOps for LLMs | LLM serving infrastructure (vLLM, TGI, Triton), autoscaling, observability (traces, metrics, logs), A/B testing in production, model registry, deployment patterns (canary, blue-green), cost governance, incident response for LLM systems |
| **Stage 7** | Advanced Architectures | Mixture of Experts, speculative decoding, multi-modal models, constitutional AI, RLHF from scratch, model merging, long-context techniques (RoPE extensions, sliding window), frontier model architecture internals |

Before writing any stage, check: what does the student already know from all previous stages? Never re-explain a concept already covered unless you are explicitly building on it and say so.

---

## How to Generate a Stage N Lesson: Step-by-Step

1. **Start with the document header:**
   ```
   # Stage N: [Title] — Complete Senior-Level Lesson

   > **Three layers per topic:**
   > **Concept Layer** — how it actually works, plain English first, then deep mechanics
   > **Engineering Layer** — production code, edge cases, what breaks
   > **Architecture Layer** — system design, patterns, scaling, tradeoffs

   ---

   **Central concept of this stage:** [one phrase]

   [One paragraph: what this stage is, why it matters, how it connects to the previous stage]

   ### Topic Dependency Map
   [ASCII tree of how topics connect]

   ---
   ```

2. **Write the Table of Contents** listing every section as `- [N.X Section Title](#anchor)`

3. **For each topic section:**
   - Write the Mental Model blockquote first — commit to the anchor before writing the content
   - Write the three connector lines: Connects to / Parent concept / Builds on
   - Concept Layer: plain English intro → mechanism → worked example → "In short:" → table or diagram
   - Engineering Layer: interface → implementation → 30–80 lines of code → 2–3 explanation sentences → edge cases
   - Architecture Layer: ASCII system diagram → scaling discussion → tradeoff table
   - Bridge sentence connecting to next topic
   - Senior Checklist: 5–7 actionable, specific, non-obvious items

4. **Write the Capstone Project** — it must tie together every topic in one runnable system

5. **Write the Quick Reference Cheat Sheet** — Stage Mind Map, Framework Summary, Key Numbers, Memory Anchors

6. **Do not stop early.** Every topic in the curriculum scope for that stage must be covered to full depth. If context limits are a concern, write in multiple passes — but never deliver a partial lesson that claims to be complete.

---

## Common Mistakes to Avoid

| Mistake | Instead |
|---------|---------|
| Starting with jargon before plain English | One plain sentence first, then the technical term |
| No "In short:" after complex explanations | Every 5+ line explanation ends with a plain one-liner |
| Concepts that appear without setup | Always establish the problem before the solution |
| No bridges between sections | Every section ends with a sentence pointing to the next |
| No dependency map at the top | Always show how topics connect before the reader starts |
| Explaining what something is instead of how it works | Show the algorithm, the data flow, the mechanism |
| Code that uses placeholder APIs that don't exist | Only call real, documented API methods |
| Code with no explanation after it | 2–3 sentences after every code block |
| Checklists with vague items like "make sure it's secure" | Specific: "Wrap all untrusted input in XML tags before injecting into prompts" |
| Mental models that describe instead of anchor | Test: cover the header — does the mental model still tell you the topic? |
| Covering topics shallowly to fit more in | Depth over breadth — one topic done right beats three done weakly |
| Architecture sections with no diagram | Every Architecture Layer has at least one ASCII diagram |
| Cheat sheet with vague anchors | Every anchor names a specific technique, tool, or threshold |
| Recommendations without numbers | "Use temperature 0.0–0.2 for code" — not "use a low temperature" |
| Writing to impress | Write so the reader feels smart, not so the writer sounds smart |

---

## Exact File Format for Delivery

- Format: Markdown (`.md`)
- Filename: `StageN_[Topic]_Complete_Lesson.md` (e.g., `Stage3_RAG_Complete_Lesson.md`)
- Encoding: UTF-8
- No frontmatter/YAML headers
- End of file: a horizontal rule `---` followed by a blank line — nothing after

---

## Size and Proportion Reference

These are reference ranges from a well-executed lesson. Use the most recently completed stage as your calibration. If your sections are shorter than these ranges, you are not going deep enough. If they are longer, check for padding — cut anything that does not add a new insight.

| Component | Approx Lines |
|-----------|-------------|
| Stage Intro + Dependency Map | 20–40 |
| Per topic section (average) | 250–350 |
| Capstone project | 300–450 |
| Quick Reference Cheat Sheet | 60–80 |
| **Total per stage (12–18 topics)** | **~4,000–7,000** |

Per topic section approximate breakdown:

| Sub-section | Avg Lines |
|-------------|-----------|
| Mental Model + Connector lines | 8–12 |
| Concept Layer (with "In short:" summaries) | 80–150 |
| Engineering Layer (code + explanations) | 80–130 |
| Architecture Layer (diagram + discussion) | 60–100 |
| Bridge sentence | 2–3 |
| Senior Checklist | 8–10 |

---

## Universal Non-Negotiable Requirements for Every Lesson

These apply to every stage, every topic, every section — no exceptions:

1. **Mental Model blockquote** — 🧠 emoji + blockquote format at the start of every topic section
2. **Three connector lines** — "Connects to / Parent concept / Builds on" after every section header
3. **Three-layer structure** — Concept, Engineering, Architecture are always separate — never collapsed
4. **Plain English first** — every technical concept introduced with a plain-language sentence before the technical definition
5. **"In short:" summaries** — after every complex explanation of 5+ lines
6. **"Because" chains** — every fact connected to its cause and consequence
7. **Bridge sentences** — every topic section ends by pointing to the next topic
8. **Topic Dependency Map** — ASCII tree once at the top of every lesson
9. **Senior Checklists** — ⚡ emoji + bold header, 5–7 items, at the end of every topic section
10. **Production Python code** — not pseudocode, not simplified examples — real, runnable, typed
11. **Code explanation sentences** — 2–3 sentences after every code block explaining what it did and why
12. **Real numbers** — costs in dollars/tokens, thresholds in specific values, latencies in milliseconds
13. **ASCII system diagrams** — in every Architecture Layer — not text descriptions of diagrams
14. **Comparison tables** — for any decision with 3+ options
15. **One unified capstone project** — ties every topic in the stage together in one runnable file
16. **Quick Reference Cheat Sheet** — Stage Mind Map + Three-Layer Summary + Key Numbers + Memory Anchors
17. **No fluff** — every paragraph earns its place; no padding, no repetition, no observations the reader already knows

---

## Prompt Template for New Instances

Copy this prompt exactly when starting a new stage lesson. Fill in the bracketed parts:

```
You are writing a complete senior-level engineering lesson for Stage N: [Topic Title].

Use the instruction file at [path to this file] as your complete specification. Follow every rule in it exactly.

The student is a senior software engineer learning to build production AI systems.
They have completed all previous stages and expect the same depth, format, and language clarity.

Stage N covers the following topics:
- N.1 [Topic]
- N.2 [Topic]
- N.3 [Topic]
[... list all topics]

Key requirements (all defined in detail in the instruction file):
- Stage Intro with central concept + ASCII topic dependency map
- Three layers per topic: Concept (plain English first), Engineering (real code), Architecture (ASCII diagrams)
- Mental Model (🧠) + three connector lines at the start of each topic
- "In short:" summaries after every complex explanation
- "Because" chains connecting facts to causes and consequences
- Bridge sentence at the end of every topic pointing to the next
- Code blocks always followed by 2–3 explanation sentences
- Senior Checklist (⚡) at the end of each topic — 5–7 items, specific and actionable
- Capstone project integrating all topics in one runnable file
- Quick Reference Cheat Sheet: Stage Mind Map + Key Numbers + Memory Anchors

Write the complete lesson. Do not summarize, compress, or stop early.
Start with the document header and Stage Intro, then the dependency map, then the Table of Contents, then each section in full.
Target: ~300 lines per topic section. Every topic in the list above must be fully covered.
```

---
