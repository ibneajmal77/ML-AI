# Stage 2 — LLM Foundations for Real Applications

## Before doing anything with lessons — read this

This project has a template that governs how every lesson is written, rewritten, or reviewed.

**Read the template first, every time:**
`Stage-2-Lesson-Template-System.md` (in this directory)

It contains: concept ownership map, project build progression, lesson type rules, section format, writing style rules, hard constraints, and the quality checklist. Do not write or edit a lesson without reading it first.

---

## Project

**Name:** `ticket-ops-api` — a FastAPI-based Internal Operations Copilot.

**How it works:** Each lesson builds one module. Every lesson produces a file that stays in the codebase. The project is the spine — lessons exist to build it, not the other way around.

---

## What has been built

| Lesson | File built | What it does |
|--------|-----------|--------------|
| 2.1 | `capability-boundary.md` | LLM capability boundary decisions |
| 2.2 | `architecture-notes.md` | Transformer behavior and design implications |
| 2.3 | `app/utils/tokens.py` | Token counter, budget tracker, truncation |
| 2.4 | `app/config.py` + `app/services/llm.py` | Task-based parameter config + API wrapper |
| 2.5 | `app/prompts/templates.py` | User message constructors for all four task types |

---

## Hard constraints — these cannot be waived

1. Every concept opens with a **scenario**, not a definition or concept name
2. No concept owned by another lesson is re-explained — one sentence + `(→ X.X)` citation only
3. Scope Firewall must be filled in with specific concepts in all three buckets
4. Exit criteria must include FILE + TEST + METRIC — all three, all specific
5. No sentence longer than 25 words
6. Concept names appear **after** the scenario demonstrates the concept — never before
7. Key Takeaways (§10) come before Quick Review Card (§11)

---

## Canonical section order

```
§1   Objective
§2   Where This Fits in the System
§3   Core Mental Model + Decision Rule
§4   Main Concepts + Production Reality
§5   Project Integration
§6   Practical Session
§7   Interview Drill
§8   Retrieval Pack
§9   Deliverable
§10  Key Takeaways
§11  Quick Review Card
§12  Optional Advanced Notes (only if all three inclusion conditions are met)
```

---

## Writing style — applied to every word

- Write for a programmer who can read code but has never used an LLM API
- Short sentences, max 25 words each
- Plain words — see the Word Rules tables in the template
- "You" throughout — no passive voice, no third person
- Concrete scenario first, concept name second — always
- Takeaways say what to DO, not what something IS
- No academic section titles or textbook names for mental models

---

## Lesson files

All lessons: `X.X-Title-NEW.md` in this directory
Template: `Stage-2-Lesson-Template-System.md` in this directory
