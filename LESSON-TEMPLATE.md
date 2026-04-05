# AI/ML Engineering Curriculum - Lesson Template

**Version:** 2.1 (Cognitive Order + Beginner-First Language)
**Last updated:** 2026-04-04
**Reference lesson:** Stage-1-Practical-ML-Foundations/1.1-What-ML-Is-in-Practical-Business-Terms.md

---

## DESIGN PRINCIPLES

This template is built on cognitive principles. Every section exists because of one of these - if a section does not serve a principle, it does not belong.

There are currently eight principles listed below.

| Principle | What it means for lesson design |
|---|---|
| **Meaning gates storage** | The brain files new information against existing schemas. Admin, references, and contracts placed before the concept have no schema to attach to — they slide off. Meta goes last. |
| **Attention is finite** | Extraneous detail competes with learning-critical detail. Every section has a job. If two sections do the same job, one is redundant. |
| **Repetition must layer** | A concept can appear multiple times, but each appearance must do different work: introduce → apply → retrieve. Restating is not repetition — it is noise. |
| **Code teaches or proves** | Code placed after a concept is explained proves it is real. Code placed at peak curiosity — before the question is resolved — is a teaching tool. These are not the same. |
| **Desirable difficulty encodes** | Forcing the learner to produce an answer before it is shown creates stronger memory than reading the answer directly. Small effort before revelation is a feature, not a burden. |
| **Symmetry reduces cognitive load** | When every lesson has the same structure and every concept block has the same shape, the learner stops spending effort navigating and can spend it learning. |
| **Mind-maps beat flat lists** | The brain remembers grouped families, contrasts, and output-shapes better than long undifferentiated inventories. Tables should cluster related methods into meaningful mental buckets. |
| **Meaning first, map second** | Concept blocks should teach the idea first. Broad method landscapes and tool surfaces belong later in one consolidated map, not repeated inside every concept. |
| **Simplify language, not knowledge** | Keep the full technical and senior-level content, but explain it in plain language first so a beginner can understand it before learning the expert framing. |

---

## GLOBAL WRITING STANDARD

Every lesson generated from this template must satisfy all of the following:

- Keep the full technical meaning. Simplify the explanation, not the knowledge.
- Write for a first-time learner who has no prior ML background unless the lesson explicitly says otherwise.
- Use simple words first, then introduce the precise technical term.
- Keep senior-level judgment, production framing, and interview-grade rigor in the content.
- Do not remove nuance or hard truths. Translate them into easier language.
- Prefer short sentences, explicit transitions, and concrete examples.
- Introduce one new mental move at a time inside each concept block.
- Examples must be complete enough that a beginner does not get stuck on missing setup.
- `Senior signal` is the destination. The explanation before it should still be beginner-readable.

**Core rule:** a beginner should be able to read the lesson, while an experienced engineer should still respect its correctness.

---

## LESSON HEADER BLOCK

Place at the very top of every lesson file.

```markdown
# Lesson [X.Y]: [Full Lesson Title]

**Stage:** [Stage Name]
**Evaluation:** D=[H/M/L] | I=[H/M/L] | W=[H/M/L] | Depth=[Wide/Strong] | Priority=[Now/Next/Later]
**Estimated Time:** [X–Y minutes]
**Roles:** [comma-separated list of target roles]
```

---

## FULL SECTION ORDER

These sections appear in this exact order in every lesson. Do not reorder. Do not add sections between them.

```
1.  HOOK
2.  ORIENTATION
3.  CONCEPT BLOCK 1          ← repeating unit (max 3 per lesson)
4.  CONCEPT BLOCK 2
5.  CONCEPT BLOCK 3
6.  DECISION RULES
7.  COMMON MISTAKES
8.  RETRIEVAL PRACTICE
9.  SUMMARY
10. FAST RECALL
11. PRACTICAL LANDSCAPE
12. REFERENCE AND ADMIN
```

---

## SECTION 1: HOOK

**Cognitive job:** Capture attention with a specific, role-relevant consequence before any content is introduced.

**Rules:**
- Maximum 3 lines of prose
- Must describe a specific failure, wasted effort, or missed opportunity — not a generic "this is important" statement
- The failure should be something the target learner could realistically walk into
- End with one sentence: "After this lesson you will be able to [specific capability]."
- Use plain language. The hook should be immediately understandable on first read.

**Template:**

```markdown
## HOOK

[One specific scenario — name the domain, describe what happened, name the consequence.]
[Second line: what was missing that caused this — the knowledge gap this lesson fills.]

**After this lesson you will be able to [specific verb phrase].**
```

**Example (Lesson 1.1):**
> A product team spent three months building an ML wiki categorizer. It scored 89% accuracy. Eight months later it needed a retrain nobody owned, and accuracy had silently fallen to 71%. A two-hour keyword classifier would have scored 91% on day one and never needed maintenance.
>
> **After this lesson you will be able to answer the question that prevents this: should this be ML at all — and if yes, what are you committing to beyond the model?**

---

## SECTION 2: ORIENTATION

**Cognitive job:** Give the learner just enough context to know where they are — not a full introduction, not a lesson contract.

**Rules:**
- Maximum 2 lines of prose
- Must include a pipeline ASCII diagram showing where this lesson sits in the larger system
- No career notes, no prerequisites, no forward links — those belong in the Admin section
- "YOU ARE HERE" marker on the diagram
- The two lines should orient a beginner, not assume insider vocabulary.

**Template:**

```markdown
## ORIENTATION

[One sentence: where this sits in the learning sequence.]
[One sentence: what it unlocks or depends on.]

```text
[ASCII pipeline diagram with YOU ARE HERE marker]
```
```

---

## CONCEPT BLOCKS (Sections 3–5)

**Every lesson has exactly 2 or 3 concept blocks. Never 1. Never more than 3.**

Each concept block is a self-contained, repeating unit with a fixed internal structure. The same structure appears in every block in every lesson. A reader can predict exactly what is coming before they scroll.

**Concept block internal structure (in order):**

```
A. Concept heading
B. 6-row fixed table
C. Mental model
D. Desirable difficulty prompt
E. First concrete artifact (code / scenario / contrast table)
F. Three-column code annotation (if code was used)
G. Faded practice prompt
H. Real-world scenario (blockquote)
I. Mini recall prompt
```

---

### A. Concept Heading

```markdown
## CONCEPT [N]: [Concept Name]
<!-- cognitive job: [state the specific job this concept does in this lesson] -->
```

---

### B. Six-Row Fixed Table

This table has the same six rows in every concept in every lesson. Never change the row order. Never rename the rows.

```markdown
| Dimension | Answer |
|---|---|
| **What it is** | [one-sentence definition — no jargon, no hedging] |
| **Why it matters** | [why this concept exists — what problem it solves] |
| **When to use it** | [specific conditions where this applies] |
| **When NOT to use it** | [the mistake case — when it looks right but is wrong] |
| **What juniors miss** | [the cognitive trap — the assumption that breaks in production] |
| **Senior signal** | [the exact language or framing that shows mastery] |
```

**Rules:**
- Every row must be filled. No "N/A" except in rows where the concept genuinely has no answer.
- "What juniors miss" is NOT the same as "When not to use it." It describes a belief that leads to misuse — the trap that feels correct but is not.
- "Senior signal" is language, not advice. Write it as if quoting a senior engineer speaking.
- The table should be readable by a beginner. Keep jargon out of the row text unless the term is explained immediately.

---

### C. Mental Model

**Cognitive job:** Build a schema the learner can attach the concept to. Use analogy and contrast, not explanation. Do not re-explain what the table already said.

**Rules:**
- Maximum 4 sentences
- Must use a concrete analogy (not a vague metaphor)
- For discrimination concepts: include a contrast between this concept and a confusable neighbor
- Do not repeat the table content in prose form
- The analogy should reduce confusion, not sound clever.

```markdown
### Mental Model

[Analogy: "[This concept] is like [familiar object/system] because [precise reason]."
Contrast (optional): "Unlike [confusable concept], this [key difference]."]
```

---

### D. Desirable Difficulty Prompt

**Cognitive job:** Force a small amount of thinking before the code or concrete artifact is shown. This is the mechanism that converts passive reading into active encoding.

**Rules:**
- One question or one prediction request
- Placed directly before the code block or first concrete artifact
- The learner should attempt an answer before reading further
- The prompt reveals its answer through the artifact that follows — do not give the answer in the prompt

```markdown
---

> **Before reading on:** [One question or prediction the learner should attempt.]

---
```

---

### E. First Concrete Artifact

**Cognitive job:** Anchor the abstraction to something visible and concrete before the explanation fully resolves it.

**Selection rule — pick the artifact type based on concept type:**

| Concept type | Best artifact | Why |
|---|---|---|
| Operational (something you call, run, or configure) | Code block | Shows the interface and behavior directly |
| Judgment / decision-based | Scenario or worked example | Shows how the concept applies to a real situation |
| Discrimination (A vs B) | Side-by-side contrast table or two code blocks | Forces direct comparison |
| Structural / architectural | ASCII diagram | Shows relationships that prose cannot express clearly |

**For code blocks:**
- All code must be runnable
- If the code produces output, show the output as inline comments or in a following block
- Code block immediately followed by the three-column annotation (Section F)
- Any variables used later in the example must be defined. Do not leave missing setup that a beginner has to infer.
- Prefer the smallest complete example that still preserves the full concept.

---

### F. Three-Column Code Annotation

**Cognitive job:** Map the important parts of the code to the concept. Teach the *why*, not just the *what*.

**Rules:**
- Only annotate **decision-bearing lines** — lines that carry a logical choice, are commonly misunderstood, or have a common mistake associated with them
- Do NOT annotate every line — over-annotation collapses attention to the level of no annotation
- Maximum 6 rows per annotation table
- The "Why it matters" column must add meaning beyond what the code already says
- Write the annotation so a beginner can understand both the mechanics and the reason.

```markdown
| Code line / block | What it does | Why it matters |
|---|---|---|
| `[code fragment]` | [mechanics — what this line does] | [meaning — why this matters, or what goes wrong without it] |
```

---

### G. Faded Practice Prompt

**Cognitive job:** Force the learner to apply the concept with partial or no guidance before the next concept begins.

**Rules:**
- One scenario-based prompt
- Learner must produce something (fill in values, make a decision, name a consequence) — not just read
- For Wide depth lessons: this can be one sentence. For Strong depth lessons: include a mini scenario with 2–3 decision points.
- The prompt must be answerable from what was just taught. Difficulty should come from thinking, not hidden assumptions.

```markdown
**Practice — apply the concept:**

> [Scenario or prompt that requires the learner to use the concept they just learned.]
> [For Strong depth: include 2–3 specific questions within the scenario.]
```

---

### H. Real-World Scenario

**Cognitive job:** Ground the concept in a named domain with a real outcome.

**Rules:**
- Always in a blockquote
- 2–4 sentences
- Must name a domain or company type (not "a company" — say "a fintech startup" or "a SaaS support team")
- Must show an outcome, not just describe a situation
- If the scenario is a failure: describe what the correct approach would have prevented

```markdown
> **Real-world scenario:** [Domain/company type] [what they did or tried to do] [what the outcome was and why it demonstrates the concept].
```

---

### I. Mini Recall Prompt

**Cognitive job:** Strengthen encoding of the concept just taught before moving to the next one.

**Rules:**
- One sentence. No more.
- The learner should be able to answer from memory — no scrolling back
- Answers should be checkable in a few seconds (definition, contrast, one-word concept name)

```markdown
---

> **Mini recall:** [One-sentence question answerable from memory without re-reading.]

---
```

---

## SECTION 6: DECISION RULES

**Cognitive job:** Support transfer — give the learner a practical instrument for applying the lesson's concepts in real decisions.

**Rules:**
- This section follows all concept blocks
- Must include a step-by-step decision framework (numbered or flowchart)
- Must include a worked example applying the framework to a real scenario
- Must include a table of red-flag phrases or conditions that indicate misapplication
- For Strong depth lessons: include a worked example followed by a faded version the learner completes
- Keep the framework operational and beginner-followable. A learner should be able to execute the steps, not just admire them.

**Template structure:**

```markdown
## DECISION RULES

### [Framework Name]

[Step-by-step decision sequence — numbered list or ASCII flowchart]

### Worked Example

[Table walking through the framework for a specific scenario]

### Red Flags

| What someone says / does | Why it should raise a flag |
|---|---|

### [When to Use / When Not to Use] (summary table)

| Use [concept] when | Do NOT use [concept] when |
|---|---|
```

---

## SECTION 7: COMMON MISTAKES

**Cognitive job:** Inoculate against predictable errors by teaching the error's structure, not just the correct answer.

**Rules:**
- 3–4 mistakes per lesson (Wide depth) or 4–5 (Strong depth)
- Every mistake must follow the four-part structure below — no exceptions
- "Why it feels right" is the most important part: it explains why the mistake is not stupidity but a predictable cognitive trap
- Mistakes should cover: one conceptual error, one tooling/process error, one measurement/metric error, one production/operational error
- Explain the mistake in plain language first, then name the deeper mechanism.

**Four-part mistake structure (mandatory):**

```markdown
### Mistake [N]: [Mistake Name]

**Mistaken instinct:** [What the learner will naturally do — stated as an action or belief]

**Why it feels right:** [The surface logic that makes this seem correct — be generous, not condescending]

**Why it fails:** [The deeper mechanism that breaks it in production — be specific]

**Better move:** [The concrete alternative in 1–2 sentences — actionable, not abstract]
```

---

## SECTION 8: RETRIEVAL PRACTICE

**Cognitive job:** Strengthen encoding through active retrieval at multiple cognitive levels.

**Rules:**
- 3 questions for Wide depth lessons, 5 questions for Strong depth lessons
- Must include one of each type: explain + classify + apply + error diagnosis
- For Strong depth: add a fifth question — "senior framing" (interview pressure simulation)
- Questions must require thinking, not recognition — scenario-based, not "define X"
- Include expected answer direction below each question (not a full answer, a check)
- Retrieval prompts must test understanding of the concept, not the learner's ability to decode complicated wording.

**Four question types:**

| Type | What it tests | Example format |
|---|---|---|
| **Explain** | Concept compression — can the learner reduce the concept to its essential meaning | "In two sentences, explain X to a software engineer who has never worked with ML." |
| **Classify** | Discrimination — can the learner apply a concept to unfamiliar cases | "For each of these scenarios, say whether you would use A or B, and name the deciding factor." |
| **Apply** | Transfer — can the learner use the framework in a new situation | "Walk through the decision framework for this scenario." |
| **Error diagnosis** | Production judgment — can the learner identify what went wrong and why | "Your model was 91% at launch and now feels random. Name three hypotheses before looking at the code." |

**Template:**

```markdown
## RETRIEVAL PRACTICE

Work through these without re-reading. The effort of recall is the mechanism that locks the concept in.

---

**1. Explain**
[Question]

*Check:* [Expected answer direction — not the full answer, just enough for self-assessment]

---

**2. Classify**
[Question with 3–4 labelled scenarios]

*Check:* [Expected classification for each scenario]

---

**3. Apply**
[Scenario requiring the learner to run through the decision framework]

*Check:* [Expected outcome and the key condition that drives it]

---

**4. Error diagnosis**
[A production situation where something degraded — what would you check?]

*Check:* [The most important hypotheses and what would confirm or rule them out]

---

**5. Senior framing** (Strong depth lessons only)
[Interview simulation prompt]

*Check:* [The first 2–3 questions a senior engineer would ask, and why]
```

---

## SECTION 9: SUMMARY

**Cognitive job:** Stabilize recall by reducing the lesson to its highest-density form.

**Rules:**
- Exactly 5 bullets — no more, no fewer
- Each bullet is one sentence
- Bullets must be the things the learner should be able to recall in an interview without notes
- One ASCII mental model diagram at the end — shows the main decision or structure of the lesson
- No new information here 
- Summary bullets should be simple enough to say out loud from memory.— everything in the summary was taught earlier in the lesson

**Template:**

```markdown
## SUMMARY

**Five things to carry out of this lesson:**

1. [Bullet — fact or principle]
2. [Bullet — fact or principle]
3. [Bullet — fact or principle]
4. [Bullet — fact or principle]
5. [Bullet — fact or principle]

```text
[ASCII mental model diagram — the main decision, framework, or structure of the lesson]
```
```

---

## SECTION 10: FAST RECALL

**Cognitive job:** Compress the full lesson into a revision-friendly memory board that can be scanned in under a minute.

**Rules:**
- This section appears after `SUMMARY` and before `PRACTICAL LANDSCAPE`
- It is for fast revision, not for introducing new material
- Keep it compact and high-signal
- Use chunks, contrasts, triggers, and output-shapes so it matches how human memory reconstructs knowledge

**Sub-sections (in order):**

```markdown
## FAST RECALL

### What This Lesson Is Really About
[One sentence that compresses the lesson into its deepest practical meaning.]

### 30-Second Recall
- [5 short bullets: whole lesson compressed]

### Core Mental Model
```text
[tiny ASCII map]
```

### Must Know Cold
- [3-6 bullets]

### Code Patterns to Remember
```python
# canonical shapes only
```

### Decision Triggers
- [if X, usually Y]

### Output Shapes to Remember
| Need | Typical output |
|---|---|

### Confusions to Avoid
| Easy confusion | Correct distinction |
|---|---|

### Top Traps
- [short warning bullets]

### One Good Example to Keep
[2-4 sentence memorable anchor example]

### Revision Path
1. [scan order]
2. [scan order]
3. [scan order]
```

**Design guidance:**
- `What This Lesson Is Really About` should be the single sentence the learner could say if forced to summarize the entire lesson in one breath
- Fast recall should use easy wording even when the lesson itself is technically deep.
- `30-Second Recall` should let the learner reconstruct the lesson quickly
- `Core Mental Model` should be smaller than the main summary diagram
- `Code Patterns to Remember` should show API shapes, not full code
- `Decision Triggers` should compress action rules
- `Output Shapes to Remember` should anchor methods to what they return
- `Confusions to Avoid` should list the most commonly mixed-up pairs
- `Top Traps` should be emotionally sticky warning cues
- `One Good Example to Keep` should be the single most memorable concrete case in the lesson

---

## SECTION 11: PRACTICAL LANDSCAPE

**Cognitive job:** Give the learner one consolidated practical map of methods, choices, tools, and outputs after the concepts are already understood.

**Rules:**
- This section appears after `FAST RECALL` and before `REFERENCE AND ADMIN`
- Use it to consolidate method families and coding surfaces in one place
- Do not repeat these tables inside each concept block
- Keep the tables grouped, compact, and easy to scan
- Prefer grouped family maps over exhaustive algorithm lists

**Sub-sections (in order):**

```markdown
## PRACTICAL LANDSCAPE

### Technique Landscape

| [concept family] | [main technique families] | [what they are for] |
|---|---|---|

### When to Use What

| If your real need is... | Usually start with... | Not with... |
|---|---|---|

### Coding Surface

| [concept family] | [common libraries / tools] | [canonical API shape] | [typical output] |
|---|---|---|---|

### Output Shapes to Remember

| Need | Typical output |
|---|---|
```

**Design guidance:**
- `Technique Landscape` should show the family-level map, not detailed tutorials
- `When to Use What` should compress routing decisions into memorable triggers
- `Coding Surface` should show only the canonical API shapes learners are most likely to touch
- `Output Shapes to Remember` should anchor methods to what they return, because outputs are strong memory hooks
- If a lesson has very little method surface, this section can be brief, but it should still exist

---

## SECTION 12: REFERENCE AND ADMIN

**Cognitive job:** Support later reuse — this section is for scanning on return, not for first reading.

**Rules:**
- This section is always last
- Contains all administrative and reference material that was removed from the opening
- Sub-sections appear in this fixed order
- Reference material may use more compact wording, but it must still be readable on a later revisit.

### Sub-sections (in order):

**12.1 Quick Reference table**

```markdown
### Quick Reference

| Concept | One-Line Meaning | Signal | Memorize or Look Up |
|---|---|---|---|
| [concept] | [one-line definition] | ⭐/⭐⭐/⭐⭐⭐ | 🧠 Memorize / �- Look up |
```

Signal key: ⭐ = useful to know, ⭐⭐ = important, ⭐⭐⭐ = essential, must own
Memorize key: 🧠 = hold in working memory, �- = look up when needed, 🧠 pattern �- syntax = memorize the concept, look up the exact API

---

**12.2 Signal Hierarchy**

```markdown
### Signal Hierarchy — What to Know at What Depth

**Must know cold (no prompting)**
- [3–5 items — things a senior engineer should say without hesitation]

**Should recognize when you encounter it**
- [2–4 items — concepts you should identify on sight but not define from memory]

**Can look up**
- [items — specific APIs, exact syntax, implementation details]

**Overkill for this stage**
- [items — things that are real but not relevant to the target role at this level]
```

---

**12.3 Vocabulary Engineers Use**

```markdown
### Vocabulary Engineers Use — Senior Signal

```text
"[Phrase a senior engineer uses in production]"
  → [What it signals: what this phrasing reveals about their understanding]

"[Phrase]"
  → [Signal]
```
```

---

**12.4 Job Market Signal**

```markdown
### Job Market Signal

[2–3 sentences: why employers care about this specific lesson, what capability it signals]

**Job description phrases that map to this lesson:**
- "[exact JD phrase]"
- "[exact JD phrase]"

**Weak vs strong answer:**

Q: "[Interview question]"

- **Weak:** "[Answer that shows surface knowledge only]"
- **Strong:** "[Answer that shows judgment, production experience, and calibrated conditions]"
```

---

**12.5 Interview Q&A**

```markdown
### Interview Q&A

**Q: [Question]**

> "[Full senior answer — 4–8 sentences, natural speech, specific conditions and examples]"

---

**Q: [Question]**

> "[Full senior answer]"

---

**Q: [Question]**

> "[Full senior answer]"
```

Rules for senior answers:
- Must name specific conditions, not just say "it depends"
- Must include at least one concrete example (a domain, a number, a specific case)
- Must sound like a person speaking, not a textbook definition

---

**12.6 Forward Links**

```markdown
### Forward Links — Where You Will Use This

| What you learn here | Where it reappears |
|---|---|
| [concept] | Lesson [X.Y] ([brief description]) |
```

---

**12.7 Lesson Contract**

```markdown
### Lesson Contract

**Prerequisites**
- [specific lesson or skill — one line each]

**After this lesson you can**
- [capability — written as "can [verb phrase]", not "understand X"]

**This lesson supports**
- [downstream lesson or project — one line each]

**Interview questions this lesson directly prepares**
- "[exact question]"

**Completion checklist**
- [ ] [specific demonstrable behavior — not "understand X", write "can explain X to a non-ML engineer"]
```

---

## DEPTH CALIBRATION RULES

### Wide (W) Depth

- 2-3 concepts
- Shorter mental models (2 sentences)
- 1 code block per concept
- 3 retrieval questions
- Faded practice is optional (1 per lesson minimum)
- Admin section can be lighter
- Keep wording especially simple because wide lessons often serve as first exposure

### Strong (S) Depth

- 3 concepts (always)
- Fuller mental models (3-4 sentences with explicit contrast)
- Worked example + faded practice per concept
- 5 retrieval questions
- Faded practice required for every concept block
- Keep full senior-level content, but explain it in plain language before using compressed expert wording
- Code annotation table required (not optional)

---

## REPETITION RULE

Every concept may appear **at most 3 times** in one lesson:

1. **Introduction** — the concept block (introduce and build schema)
2. **Application** — the Decision Rules or worked example (apply in context)
3. **Retrieval** — the Retrieval Practice section (retrieve under pressure)

If the same concept appears a fourth time, that appearance must:
- Add a new contrast that was not in the first three appearances, OR
- Add a new failure mode that was not covered, OR
- Be cut

---

## ANTI-PATTERNS TO AVOID

| Anti-pattern | Why it fails | Fix |
|---|---|---|
| Quick Reference before concepts | No schema to attach it to — slides off | Move to Admin section |
| Lesson Contract before teaching | Administrative overhead before meaning | Move to Admin section |
| Code in Part 3 (after full explanation) | Code proves, not teaches — curiosity is gone | Move code into concept block, after mental model |
| "Junior vs senior contrast" without error structure | Learner knows the answer but not why the mistake happens | Use four-part mistake format |
| Retrieval only at the end | Tests reading completion, not encoding | Add mini recalls inside each concept block |
| Every code line annotated | Attention collapses — all lines look equally important | Annotate decision-bearing lines only |
| Same concept restated in 4+ places | Familiarity without mastery — feels thorough, is noise | Apply repetition rule: max 3 appearances, each doing different work |
| "This is important" hook | Generic, not emotionally salient | Name a specific failure with a specific cost |
| Section structure varies between lessons | Learner spends cognitive effort navigating, not learning | Lock section order; lock concept block structure |

---

## TEMPLATE FILE VERSIONS

- **v1.0:** 12-section combined template (PART 1–6 structure + AI-ML What/Why callouts)
- **v2.0 (current):** Cognitive-order template with fixed concept blocks, cognitive jobs per section, four-part mistake format, mid-lesson retrieval, three-column code annotation




