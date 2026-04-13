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

## LESSON HEADER

```
Lesson:     [number and title]
Type:       [from type map above]
Owns:       [concepts fully explained here]
References: [concepts cited but not re-explained — use reference syntax]
Builds on:  [previous lesson — one line on what it contributed to the project]
Enables:    [next lesson — one line on what this lesson unlocks]
Project:    [module/file being built this lesson]
```

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

## SECTION 3 — CORE MENTAL MODEL

The organizing insight that frames the entire lesson.

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

**Cognitive load rule:** Do not introduce more than 4–5 new ideas in a single concept block. If a concept has a subtopic that belongs to a later lesson, introduce it at the level needed now and note `(→ expanded in X.X)`.

**Cross-lesson rule:** Concepts owned by another lesson get one sentence and a citation. Nothing more. Do not re-explain them.

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

## SECTION 7 — DELIVERABLE

One artifact. Specific, checkable, project-integrated.

```
Format: [filename or artifact] — [what it does] — [where it lives in the project]

Good:  "app/security/sanitizer.py — injection pattern detection and tool call
        validation with audit logging — wired into /route endpoint."

Bad:   "A module for security."
```

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

## SECTION 8 — KEY TAKEAWAYS

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

## SECTION 9 — OPTIONAL ADVANCED NOTES

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

Step 5: Write the rest of the lesson.
        Follow the cognitive arc: Anchor → Build → Apply → Reinforce.
```

---

## QUALITY CHECKLIST

Run this before finalizing any lesson:

```
Objective
  □ Uses a DO verb — not "understand" or "learn"
  □ Describes something verifiable

System Position
  □ References actual project state, not generic motivation
  □ The GAP is specific — something breaks without this lesson

Mental Model
  □ Is specific, not a generic platitude
  □ Is referenced again in the concept section

Main Concepts
  □ All roadmap-required concepts are covered
  □ No concept owned by another lesson is re-explained
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

Deliverable
  □ Specific (includes filename or artifact name)
  □ Checkable by someone else
  □ Project-integrated

Takeaways
  □ Written as engineering principles, not content summaries
  □ No takeaway simply restates a Section 4 explanation

Overall
  □ No section repeats what another section already said
  □ Reading order makes each concept easier after the previous one
  □ Length is appropriate for the lesson type
```
