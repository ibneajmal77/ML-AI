# Stage 2 — Lesson Template System

**Applies to:** All lessons in Stage 2 — LLM Foundations for Real Applications (2.1–2.18)
**Purpose:** Ensure cross-lesson coherence, eliminate repetition, maintain project continuity, and enforce cognitive clarity across all 18 lessons.

---

# THE TWO-DOCUMENT SEPARATION

The system is split into two concerns:

```
DOCUMENT A — STAGE DESIGN (read once, reference when needed)
  - Lesson type classification
  - Concept ownership map
  - Project build progression

DOCUMENT B — LESSON TEMPLATE (used every time a lesson is written)
  - Lean, cognitive-flow-first structure
  - Type-specific guidance embedded as notes
  - Flexibility clause
```

---

# DOCUMENT A — STAGE DESIGN REFERENCE

## Lesson Type Map

```
FOUNDATION        2.1, 2.2, 2.3          Build mental models that explain LLM behavior
CONFIGURATION     2.4                    Control model parameters for specific tasks
PROMPT ENG.       2.5, 2.6               Design prompts as a professional discipline
IMPLEMENTATION    2.7, 2.8, 2.9          Build real modules for the copilot
RELIABILITY       2.10, 2.11             Understand and defend against failure modes
STRATEGY          2.12, 2.13             Make architectural decisions with tradeoffs
OPERATIONS        2.14, 2.15, 2.16, 2.17 Run and optimize LLM systems in production
PROJECT           2.18                   Final integration and evaluation
```

---

## Concept Ownership Map

Each concept has one home lesson. All others cite it — never re-explain it.

```
CONCEPT                              OWNED IN    CITED IN
─────────────────────────────────────────────────────────────────
Tokens (concept)                      2.1         2.3, 2.5, 2.15
Next-token prediction                 2.1         2.4, 2.10
Pretraining and its limits            2.1         2.10, 2.12
Instruction tuning / RLHF             2.1         2.5, 2.6
Embeddings (intuition)                2.2         (Stage 3 deep dive)
Attention (intuition)                 2.2         2.3
Context window (concept)              2.2         2.3, 2.17
Tokenization mechanics                2.3         —
Token budget management               2.3         2.15, 2.17
Temperature                           2.4         2.5, 2.10
Top-p, max tokens, stop sequences     2.4         2.15
Prompt structure and clarity          2.5         2.6, 2.7
Few-shot prompting                    2.5         2.6
System prompts                        2.6         2.8, 2.11
JSON mode / structured output         2.7         2.8, 2.10
Pydantic schema validation            2.7         2.8
Function calling mechanics            2.8         2.11
SSE / streaming mechanics             2.9         —
Hallucination causes and types        2.10        2.5, 2.11, 2.12
Prompt injection                      2.11        2.8
Unsafe tool use                       2.11        2.8
Model size vs. quality tradeoff       2.12        2.13, 2.15
Vendor comparison                     2.13        2.15
Prompt versioning                     2.14        2.16
Batching, caching, routing            2.15        —
Evaluation frameworks                 2.16        2.14
Context compression                   2.17        2.15
```

**Reference syntax — use exactly this, no re-explaining:**

```
Already taught:   "Tokens (→ 2.1) determine your cost here."
Coming later:     "We'll cover caching in 2.15 — for now, know it exists."
Outside stage:    "Fine-tuning is covered in Stage 4, not here."
```

---

## Project Build Progression

The project is `ticket-ops-api` — a FastAPI-based Internal Operations Copilot.
Every lesson builds a specific piece. This is the project spine.

```
LESSON   COMPONENT BUILT                                    TYPE
──────────────────────────────────────────────────────────────────────────
2.1      capability-boundary.md                             Architecture decision
         (what LLM handles directly vs. needs tools)

2.2      architecture-notes.md                             Understanding artifact
         (transformer behavior and design implications)

2.3      app/utils/tokens.py                               Utility module
         (token counter + budget tracker)

2.4      app/config.py — LLMConfig class                   Configuration module
         (task-based parameter sets: temperature, top-p, max_tokens)

2.5      app/prompts/templates.py                          Prompt library
         (core templates: classify, summarize, extract, draft)

2.6      app/prompts/system.py                             System module
         (system prompt architecture + task injection pattern)

2.7      app/schemas/ + Pydantic validators                Parser module
         (JSON schema enforcement, field validators, retry wrapper)

2.8      app/tools/definitions.py + handlers.py            Tool module
         (tool definitions, /route endpoint, call handler)

2.9      app/services/summarize.py — SSE streaming         Streaming module
         (SSE reader + partial output handler + first-token latency)

2.10     app/services/extract.py — grounding check         Reliability module
         (grounding check, is_grounded flag on extraction response)

2.11     app/security/sanitizer.py                         Security module
         (injection detection, tool call validation, audit logging)

2.12     app/routing/model_map.py                          Routing module
         (task → model selection config)

2.13     infra setup + .env config                         Infrastructure config
         (vendor setup, primary + fallback provider wiring)

2.14     app/prompts/registry.py                           Versioning module
         (prompt version store + rollback logic)

2.15     app/cost/ + optimization layer                    Optimization module
         (caching + batching + token truncation)

2.16     app/eval/framework.py                             Eval module
         (golden dataset + rubric scoring + regression runner)

2.17     app/services/chat.py                              Memory module
         (sliding window + summarization policy + /chat endpoint)

2.18     Full integration + evaluation dashboard           Final project
         (all modules wired, tested, evaluated, README complete)
```

---

# DOCUMENT B — LESSON WRITING TEMPLATE

## THE GOVERNING PRINCIPLE

Before reading the sections, understand the one rule that overrides everything:

> **Every lesson must move through a cognitive arc: Anchor → Build → Apply → Reinforce.**
>
> Anchor: establish the mental model before introducing concepts.
> Build: explain concepts in the order they depend on each other — not alphabetically, not by importance.
> Apply: have the learner do something real before the lesson ends.
> Reinforce: close with principles, not a content repeat.
>
> **Template sections serve this arc. If strict compliance breaks the arc, break the template.**

---

## WRITING STYLE RULES — APPLY TO EVERY WORD IN EVERY LESSON

These rules apply everywhere — mental models, concept explanations, exercises, takeaways, the review card, everything. Technical depth must stay intact. The language must be simple.

**The core rule: write like you're explaining to a smart person learning this for the first time.**

---

### Sentence rules

- **One idea per sentence.** If a sentence has more than one clause connected by "—" or "which" or "and", split it.
- **Short sentences by default.** If you can say it in 10 words instead of 20, use 10.
- **Say "you" not "the learner" or "one".** Talk directly to the person reading.
- **Active voice.** "Your code catches this" not "this must be caught by your application."

**Before:**
> "The engineering consequence of this architectural reality is that budget enforcement must happen inside your application, prior to the API call, with a deliberate truncation or rejection strategy defined in advance."

**After:**
> "This means your code needs to catch oversized prompts before they reach the API — not after. Have a plan for what happens when content is too long."

---

### Word rules

Replace formal words with plain ones:

| Instead of | Use |
|---|---|
| denominated in | measured in |
| operationally | in practice / in production |
| enforcement | check / rule / guard |
| architectural | design / structural |
| instantiate | create / set up |
| allocate | give / assign |
| subsequently | after that / then |
| however | but |
| utilize | use |
| in order to | to |
| it is important to note that | just say the thing |
| this allows the learner to | you can now |

---

### Explanation rules

- **Concrete before abstract.** Show the situation first, then name the concept.
- **Say what to DO, not just what IS.** "Use `temperature=0` for classification" beats "temperature controls output variance."
- **No hedging.** Don't write "it is worth noting that" or "one might consider." Just say it.
- **No padding.** Don't open paragraphs with "In this section we will explore..." Just start.

---

### Structure rules

- Use bullet points and tables instead of long paragraphs wherever possible.
- Bold the rule or the key point — not random words.
- Code blocks for all code. Never inline multi-line code in prose.
- When listing steps, use a numbered list. When listing options, use bullets.

---

### Writing checklist — run this before finishing any lesson

```
Language
  □ No sentence longer than 25 words (split if over)
  □ No formal words from the replacement table above
  □ "You" language throughout — not passive, not third person
  □ Every paragraph starts with the point, not with setup

Explanations
  □ Every concept opens with a concrete situation, not a definition
  □ Every rule says what to DO, not just what IS
  □ No hedging phrases ("it is worth noting", "one might")
  □ No filler opening lines ("In this section...")

Structure
  □ Long prose replaced with bullets or tables where possible
  □ Steps are numbered, options are bulleted
  □ Code is in code blocks, never inline paragraphs
```

---

## LESSON HEADER

```
Lesson:     [number and title]
Type:       [from type map above]
Owns:       [concepts fully explained here]
References: [concepts cited but not re-explained — use reference syntax]
Builds on:  [previous lesson — one line on what it contributed to the project]
Enables:    [next lesson — one line on what this lesson unlocks]
Project:    [module/file being built this lesson]

Scope Firewall:
  Must know now:        [concepts the learner must fully own before leaving this lesson]
  Recognize but defer:  [concepts that will appear but are owned by a later lesson]
  Ignore for now:       [concepts that are out of scope for this stage entirely]
```

The scope firewall is mandatory. If a concept appears in the lesson but belongs to the "Ignore" or "Defer" buckets, it gets one sentence maximum and a citation — never a full explanation. This is what prevents lessons from collapsing into undisciplined surveys.

---

## SECTION 1 — OBJECTIVE

One sentence. A DO verb. Concrete and verifiable.

```
Allowed:   build, implement, design, configure, debug, evaluate,
           decide, apply, structure, detect, defend, optimize

Forbidden: understand, learn, explore, get familiar with

GOOD: "Build a structured output parser with Pydantic validation
       and fallback handling for the Internal Copilot."
BAD:  "Understand how structured outputs work."
```

---

## SECTION 2 — WHERE THIS FITS IN THE SYSTEM

Three parts. Three sentences maximum each. No motivational preamble.

```
STATE:    What does the project currently look like? What files exist? What works?
GAP:      What breaks or is impossible without this lesson?
UNLOCKS:  What becomes possible after this lesson — for the project and the next lesson?
```

This is not "why this topic matters." It is a specific description of where the learner stands in the project right now and what problem they are solving.

---

## SECTION 3 — CORE MENTAL MODEL + DECISION RULE

Two parts. Keep them short and sharp.

**Part A — Mental model:** The organizing insight that frames the entire lesson.

```
Rules:
  - One clear model, analogy, or framework — not a list of definitions
  - Must explain WHY the concept works the way it does, not just WHAT it is
  - Must reduce confusion, not just introduce vocabulary
  - Must be referenced again in Section 4 — not dropped after the intro

Weight by type:
  Foundation:       HEAVY — this is the centerpiece of the lesson
  Configuration:    LIGHT — one sentence framing is enough
  Implementation:   MINIMAL — one line to frame the build
  Strategy:         MEDIUM — a decision framework or tradeoff model
  Reliability:      MEDIUM — a mental model of the failure space
  Operations:       MEDIUM — a systems thinking frame
```

**Part B — Decision rule:** One explicit rule for when to use this lesson's pattern — and when NOT to.

```
Format:
  USE when:     [specific condition — not a vague "when appropriate"]
  AVOID when:   [specific condition where this pattern breaks or is overkill]
  SIGNAL:       [the observable sign that tells you which branch you're in]

Example (Lesson 2.4 — temperature):
  USE when:     output must be deterministic — classification, extraction, routing
  AVOID when:   output benefits from variation — drafting, brainstorming, summarization
  SIGNAL:       if you'd run the same input twice and expect the same output → temperature=0

Bad (too vague): "Use temperature when you need to control randomness."
Good: "If the task has a single correct answer, use temperature=0. If variety improves the output, use 0.7–1.0. Never leave it at the API default in production."
```

The decision rule is the thing a senior engineer would say in an interview when asked "when would you use this?" Write it as a rule you can defend, not as a description.

---

## SECTION 4 — MAIN CONCEPTS + PRODUCTION REALITY

Cover all roadmap-required concepts for this lesson.
Concepts flow into each other — order by cognitive dependency, not alphabetically.
Production reality is woven in, not separated into its own section.

**Per-concept approach by lesson type:**

```
Foundation:      concept → what it explains about LLM behavior → engineering consequence
Configuration:   what the parameter does → effect on output → when to use → failure risk
Prompt Eng.:     pattern name → when to apply → before/after example → common trap
Implementation:  problem it solves → implementation pattern → edge cases → production reality
Reliability:     failure mode → why it happens → how to detect → how to mitigate
Strategy:        option A vs B → tradeoffs → when each wins → decision rule
Operations:      process → how it breaks at scale → how to instrument → how to improve
```

**Cognitive load rule:** Do not introduce more than **3 new concepts** per lesson. If a concept has a subtopic that belongs to a later lesson, introduce it at the level needed now and note `(→ expanded in X.X)`. If you find yourself needing 5 or 6 concepts, you have two lessons — split or defer.

**Cross-lesson rule:** Concepts owned by another lesson get one sentence and a citation. Nothing more. Do not re-explain them.

---

## CONCEPT EXPLANATION FORMAT — applies inside every concept block in Section 4

Human working memory is limited. Every concept explanation must minimize extraneous load (wasted effort parsing the format) so the learner's cognitive budget goes toward forming the concept, not decoding the prose.

**The required sequence for every concept:**

```
1. SCENARIO — one sentence, concrete situation the learner can picture immediately
   No jargon. No concept name yet. Just the situation.
   Example: "You build a support bot. A user asks: 'Why is my payment failing?'"

2. WHAT HAPPENS — numbered steps or arrow chains, one line each
   Walk through the mechanism as a visible sequence.
   Use → to show flow, | to separate parallel things.
   Example:
     1. "payment failing" → mapped to meaning space
     2. clusters nearby: billing, error, transaction
     3. model focuses on "payment" + "failing", ignores filler
     4. generates response from that focused context

3. NAME — introduce the concept label after the learner has seen it work
   "This step-3 focusing process is called attention."
   The name lands on understanding, not in front of it.

4. ENGINEERING IMPLICATION — 2–3 bullets, not a paragraph
   What does this mean for how you build or design?
   Use a direct comparison where possible:
     50 policies in context  →  attention diluted  →  bad result
     5 relevant docs only    →  attention focused  →  strong result

5. COMMON TRAP — one line
   The single most frequent mistake with this concept.
   Example: "Assuming more context always helps — it doesn't if it's irrelevant."
```

**What this forbids:**
- Opening a concept with the concept name or a definition sentence before any scenario
- Paragraphs longer than 3 sentences without a structural break (bullets, steps, table, code)
- Abstract explanation before concrete example — always concrete first

**What this applies to:**
All concept explanations in Section 4, across all lesson types.
The scenario and steps adapt to the lesson type — a Foundation lesson shows a system behavior, an Implementation lesson shows a failure case, a Prompt Engineering lesson shows a before/after prompt — but the sequence is always: scenario → steps → name → implication → trap.

**What this does NOT apply to:**
Section 5 (Project Integration skeletons), Section 6 (exercise instructions), Section 8 (Key Takeaways).
Those sections have their own format rules.

---

## SECTION 5 — PROJECT INTEGRATION

Not the last section — the spine of the lesson.

```
COMPONENT:    Name and purpose of the module or file being built.
BUILD:        Exact description of what gets implemented.
SKELETON:     Starter code, prompt scaffold, or structure provided here.
              Not left for the learner to guess.
CONNECTS TO:  One sentence on what this builds on and what it enables next.
```

**Rules:**
- The skeleton must be production-relevant — variable names, error handling, and structure reflect how this would look in a real codebase.
- Every lesson must produce something that stays in the project. No throwaway code.
- Write Section 5 before writing Section 4. Let the project component anchor the lesson.

---

## SECTION 6 — PRACTICAL SESSION

3–4 exercises. Graduated in difficulty. Tied to the project.

```
Exercise format:
  Title → What to do → What to verify → Why this matters

Exercise types (pick 3–4 that fit the lesson):
  BUILD:      Implement the concept in the project context
  EXPERIMENT: Change one variable, observe the difference, explain it
  DEBUG:      Given broken output or code, diagnose and fix
  DECIDE:     Given a scenario, choose an approach and justify it
  EVALUATE:   Score output quality against a specific criterion
  COMPARE:    Run two approaches side by side, document the difference
```

**Mandatory:** At least one exercise involves a failure, edge case, or broken scenario. Production engineers debug — they don't only run happy paths.

**Mandatory:** Every exercise produces a verifiable output — a file, a printed result, a logged outcome, a documented decision. Not "think about this."

**Forbidden:** Pure reading exercises. Definition exercises. Exercises with no connection to the project.

---

## SECTION 7 — INTERVIEW DRILL

Three questions. No more. Designed to be answered out loud in under 2 minutes each.

```
Q1 — Concept question (what it is and why it works):
     "Explain [this lesson's core concept] to a senior engineer in 90 seconds."
     Write a model answer. 3–5 sentences. No filler.

Q2 — System design question (how you'd apply it at scale):
     A short scenario. The learner must choose an approach and defend it.
     Example: "You have three endpoints with different accuracy vs. cost requirements.
               How do you configure model parameters for each? What breaks if you get it wrong?"

Q3 — Debugging question (what goes wrong and how you find it):
     A broken output, a wrong metric, or a failed test.
     The learner must diagnose the cause and name the fix.
     Example: "Your /classify endpoint returns a different label on the same input 30% of the time.
               What is the most likely cause and what is the first thing you check?"
```

**Rules:**
- Q1 is always about the concept the lesson owns — not a referenced concept
- Q2 must name a real architectural tradeoff — not a trick question
- Q3 must be diagnosable from code the learner actually wrote in this lesson
- Write the model answer for all three — the learner checks their answer against it

**Why this is mandatory:** A learner who cannot answer these three questions after the lesson has not retained the material well enough to use it in an interview. Exercises build skill; interview drills build articulation.

---

## SECTION 8 — RETRIEVAL PACK

Five recall questions for today. A spaced review schedule for later.

```
RECALL QUESTIONS (answer from memory — no scrolling back):
  1. [Factual: name, definition, or value]
  2. [Causal: why does X happen / what causes Y]
  3. [Decision: in scenario Z, what would you do and why]
  4. [Trap: what is the most common mistake with this concept]
  5. [Transfer: how does this connect to [prior lesson concept]?]

PRIOR LESSON LINKS (three concepts this lesson depends on):
  → [concept name] (→ lesson X.X)
  → [concept name] (→ lesson X.X)
  → [concept name] (→ lesson X.X)

SPACED REVIEW SCHEDULE:
  D+1:  Re-answer questions 1 and 4 without looking at the lesson
  D+3:  Re-answer questions 2 and 3 without looking at the lesson
  D+7:  Answer all five from memory — if any fail, re-read only that concept block
```

**Rules:**
- Questions must be answerable from the lesson content — no trivia
- Question 5 must name a specific prior lesson (not "earlier concepts")
- The D+1/D+3/D+7 plan is for the learner to schedule themselves — it is not optional reading
- Do not write more than 5 questions. More questions = less retrieval = worse retention.

---

## SECTION 9 — DELIVERABLE

One artifact. Specific, checkable, project-integrated. Three exit criteria that must all pass.

```
Format: [filename or artifact] — [what it does] — [where it lives in the project]

Good:  "app/security/sanitizer.py — injection pattern detection and tool call
        validation with audit logging — wired into /route endpoint."

Bad:   "A module for security."
```

**Exit criteria — all three must be true before the lesson is done:**

```
FILE:    [specific file or artifact exists and is committed]
TEST:    [specific test passes — name it]
METRIC:  [one observable metric captured — what it is and where it appears]
```

Example (Lesson 2.4):
```
FILE:    app/config.py — LLMConfig class with task-based parameter sets
TEST:    tests/integration/test_classify.py::test_determinism passes
METRIC:  output_tokens logged per request — visible in ClassifyResponse
```

**One metric + one alert per lesson:**

Every lesson must name:
- **Metric:** one signal to measure that confirms the module is working correctly
  (schema compliance rate, token count, tool-call success rate, first-token latency, cost per request, etc.)
- **Alert condition:** the threshold or pattern that means something is wrong
  (e.g., "if classification returns anything outside the four labels → alert", "if output_tokens > 50 on /classify → alert, prompt may have drifted")

This is not optional. Observability is how senior engineers think — not a later concern.

**Deliverable by lesson type:**

```
Foundation:       Architecture decision document or capability map
Configuration:    Configuration module with environment-based switching
Prompt Eng.:      Prompt template file with versioning stub
Implementation:   Working module with error handling and at least one test
Reliability:      Defensive code integrated into existing module + tests
Strategy:         Architecture Decision Record (ADR) — written, not just decided
Operations:       Working operational component (versioning, caching, eval, etc.)
```

---

## SECTION 10 — KEY TAKEAWAYS

5–7 bullets. Engineering principles — not content summaries.

```
Rule: A takeaway that restates Section 4 content word-for-word is failing.
      It should crystallize the IMPLICATION, not repeat the explanation.

Bad:  "Temperature controls output randomness."
Good: "Temperature is a risk dial — low for reliability, high for creativity.
       Never leave it at default in production."
```

These are the insights the learner carries into the next lesson and into their work.

---

## SECTION 11 — QUICK REVIEW CARD

A 2–3 minute read that covers the whole lesson. Written for someone who finished the lesson a week ago and wants to refresh before an interview or before building the next module. It must work as a standalone reference — no flipping back through the lesson to understand it.

```
Format rules:
  - One line saying what the lesson is about
  - Each concept: name + one plain-English sentence + the decision rule in plain language
  - One table showing the key values, ranges, or options at a glance
  - The single most important thing to remember (the "if you forget everything else" line)
  - The exit criteria as a checklist
  - No jargon without immediate plain explanation
  - No paragraphs — only bullets, tables, and short lines
  - Must fit in one screen (roughly 30–40 lines)
```

**What makes a good card:**
- Someone can read it in 2–3 minutes and feel like they understand the lesson
- It answers: what is this, when do I use it, what goes wrong, what did I build
- It does NOT just copy the key takeaways — it restructures the whole lesson into a fast format

**Example structure:**

```
## Quick Review Card — Lesson X.X: [Topic]

**What this lesson is about in one line:**
[Plain English. No jargon.]

**Key concepts:**

| Concept | What it is (plain English) | When to use it |
|---|---|---|
| ... | ... | ... |

**The decision rules:**
- Use [X] when: ...
- Avoid [X] when: ...
- The signal that tells you which: ...

**What you built:**
- [filename] — [what it does in one sentence]
- Test: [test command] → [expected result]

**If you forget everything else, remember:**
[One sentence. The most important engineering principle from this lesson.]

**Exit criteria:**
- [ ] FILE: ...
- [ ] TEST: ...
- [ ] METRIC: ...
```

---

## SECTION 12 — OPTIONAL ADVANCED NOTES

Include only when ALL three conditions are true:
1. The topic has important advanced behavior the learner will actually hit in production.
2. Placing it in the main flow would derail the lesson.
3. It does NOT require a future lesson as a prerequisite.

Before including, ask:
- Does this belong in a later lesson instead?
- Does the main learner need this now?
- Is this genuinely hit in production, or is it research-level detail?

Keep it short. This is a footnote, not a second lesson.

---

## FLEXIBILITY CLAUSE

The template is scaffolding, not a constraint.

If two sections merge more cleanly than staying separate — merge them.
If the cognitive arc needs a different order for a specific lesson — reorder.
If strict compliance produces mechanical writing — break compliance.

The purpose of this template is:
1. Ensure every lesson is complete without being bloated.
2. Prevent cross-lesson repetition.
3. Keep the project as the spine.
4. Make the learner better at engineering, not better at reading.

If the template serves all four — follow it.
If part of it doesn't — drop that part and keep the goal.

---

## QUICK DECISION GUIDE — HOW TO WRITE A LESSON

```
Step 1: Look up the concept ownership map.
        Mark what this lesson OWNS vs. what it REFERENCES.

Step 2: Look up the project progression.
        Know exactly what module is being built and what state it starts in.

Step 3: Identify the lesson type.
        Know which sections get heavy treatment and which stay light.

Step 4: Write Section 5 (Project Integration) first.
        Let the project component anchor the lesson before explaining concepts.
        The best lessons flow TOWARD the build — not away from it.

Step 5.5: Write the Interview Drill (Section 7) before writing Section 4.
          Knowing what you need to answer forces you to teach the right things.
          If you can't write Q3 (the debug question), the lesson hasn't covered the failure mode.

Step 5: Write the rest of the lesson.
        Follow the cognitive arc: Anchor → Build → Apply → Reinforce.
```

---

## QUALITY CHECKLIST

Run this before finalizing any lesson:

```
Header
  □ Scope Firewall is filled in with specific concepts in each bucket
  □ "Must know now" concepts are the ONLY ones given full explanation

Objective
  □ Uses a DO verb — not "understand" or "learn"
  □ Describes something verifiable

System Position
  □ References actual project state, not generic motivation
  □ The GAP is specific — something breaks without this lesson

Mental Model + Decision Rule
  □ Mental model is specific, not a generic platitude
  □ Mental model is referenced again in the concept section
  □ Decision rule has USE / AVOID / SIGNAL — not just "use when appropriate"
  □ Decision rule is defensible in an interview

Main Concepts
  □ No more than 3 new concepts in the lesson
  □ No concept owned by another lesson is re-explained
  □ Every concept follows: scenario → steps → name → implication → trap
  □ No concept block opens with the concept name or a definition sentence
  □ Order follows cognitive dependency, not alphabet or importance
  □ Production reality is woven in, not appended

Project Integration
  □ Includes a concrete skeleton or scaffold
  □ The deliverable stays in the project (not throwaway code)
  □ Connects clearly to what was built before and what comes next

Practical Session
  □ At least one exercise involves a failure or edge case
  □ Every exercise produces a verifiable output
  □ Exercises are tied to the project, not isolated toy examples

Interview Drill
  □ Q1 is a concept question the learner can answer in 90 seconds
  □ Q2 is a system design scenario with a real tradeoff
  □ Q3 is a debugging scenario diagnosable from lesson code
  □ Model answers are written for all three

Retrieval Pack
  □ Exactly 5 recall questions — not more
  □ Question 5 names a specific prior lesson
  □ D+1 / D+3 / D+7 review schedule is included
  □ All questions are answerable from lesson content

Deliverable
  □ Specific (includes filename or artifact name)
  □ Checkable by someone else
  □ Project-integrated
  □ Exit criteria: FILE + TEST + METRIC all named
  □ One metric named with units or an observable output
  □ One alert condition named (what threshold means something is wrong)

Takeaways
  □ Written as engineering principles, not content summaries
  □ No takeaway simply restates a Section 4 explanation

Language and Style
  □ No sentence longer than ~25 words — split any that are
  □ No formal words (denominated, operationally, enforcement, instantiate, etc.)
  □ "You" language throughout — not passive voice, not third person
  □ Every concept opens with a concrete situation, not a definition
  □ Every rule says what to DO, not just what IS
  □ No hedging phrases ("it is worth noting", "one might consider")
  □ No filler opening lines ("In this section we will explore...")
  □ Long prose replaced with bullets or tables where possible

Overall
  □ No section repeats what another section already said
  □ Reading order makes each concept easier after the previous one
  □ Length is appropriate for the lesson type
```
