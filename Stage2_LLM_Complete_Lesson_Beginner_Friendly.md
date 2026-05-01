# Stage 2: Building with LLMs - Beginner-Friendly Complete Lesson

> Goal of this version:
> Keep the full topic coverage of the original Stage 2 lesson, but explain it in plain language for someone studying LLMs for the first time.

---

## Table of Contents
- [2.1 What an LLM Is](#21-what-an-llm-is)
- [2.2 Transformers in Depth](#22-transformers-in-depth)
- [2.3 Tokenization and Context Windows](#23-tokenization-and-context-windows)
- [2.4 Model Parameters That Matter in Production](#24-model-parameters-that-matter-in-production)
- [2.5 Prompt Design for Production](#25-prompt-design-for-production)
- [2.6 System Prompts, Role Prompts, Task Framing, Delimiters](#26-system-prompts-role-prompts-task-framing-delimiters)
- [2.7 Structured Outputs](#27-structured-outputs)
- [2.8 Function Calling and Tool Calling](#28-function-calling-and-tool-calling)
- [2.9 Streaming Responses](#29-streaming-responses)
- [2.10 Hallucinations](#210-hallucinations)
- [2.11 Prompt Injection and Unsafe Tool Use](#211-prompt-injection-and-unsafe-tool-use)
- [2.12 Model Selection](#212-model-selection)
- [2.13 Vendor Tradeoffs: Azure OpenAI vs Anthropic vs Open-Source](#213-vendor-tradeoffs-azure-openai-vs-anthropic-vs-open-source)
- [2.14 Prompt Versioning, A/B Testing, Rollback Strategy](#214-prompt-versioning-ab-testing-rollback-strategy)
- [2.15 Latency and Cost Optimization](#215-latency-and-cost-optimization)
- [2.16 LLM Evaluation Basics](#216-llm-evaluation-basics)
- [2.17 Multi-Turn Conversation Management](#217-multi-turn-conversation-management)
- [2.18 Stage 2 Project: Internal Operations Copilot](#218-stage-2-project-internal-operations-copilot)
- [Quick Reference](#quick-reference)

---

## 2.1 What an LLM Is

### Mental Model
An LLM is a machine that reads a lot of text during training and becomes very good at guessing what text should come next.

### Concept Layer

#### Tokens
LLMs do not read text exactly the way humans do. They break text into smaller pieces called `tokens`.

Examples:
- `"Hello world"` might become `["Hello", " world"]`
- `"unhappy"` might become `["un", "happy"]`
- `"2024-01-15"` may become several separate pieces

Why this matters:
- Token count affects cost
- Token count affects speed
- Token count affects whether your input fits in the model's context window

#### Subword tokenization and BPE
Modern LLMs usually use a method like BPE, which stands for Byte Pair Encoding.

Simple idea:
1. Start with very small pieces, often close to characters.
2. Look for pairs that appear together a lot.
3. Merge those pairs into larger pieces.
4. Repeat until you build a useful vocabulary.

Why not full words only?
- New words appear all the time.
- Names, code, numbers, and spelling variations would break word-only systems.

Why not single characters only?
- Sequences become too long and inefficient.

Subword pieces are a practical middle ground.

#### Next-token prediction
This is the core behavior of an LLM.

If the model sees:
`The sky is`

it predicts likely next tokens such as:
- `blue`
- `clear`
- `dark`

It picks one, adds it to the sentence, then repeats.

That one repeated action creates:
- conversation
- summaries
- code
- translations
- question answering

#### Weights
The model stores what it learned inside a huge collection of numbers called `weights`.

Think of weights as the model's learned memory of language patterns.

Why engineers care:
- Bigger models usually need more memory
- Running them can require expensive GPUs
- Model updates can change behavior, even if your prompt stays the same

#### Pretraining
Pretraining is the large first stage where the model learns from huge amounts of text.

It learns patterns such as:
- grammar
- writing style
- facts that appeared often in training
- code structure
- common reasoning patterns

But after pretraining, the model is still just a raw predictor. It is not automatically a good assistant.

#### SFT, RLHF, and DPO
These are methods used to make models more helpful.

`SFT`:
- Humans create good example answers.
- The model learns to imitate that style.

`RLHF`:
- Humans compare two answers.
- A reward model learns which answer is better.
- The main model is trained to produce answers that score better.

`DPO`:
- A simpler way to use preference data.
- Instead of a full reward-model loop, the model directly learns to prefer the better answer.

Beginner takeaway:
- Pretraining teaches language.
- Fine-tuning and preference training teach helpful behavior.

#### Base model vs instruction model vs chat model
- `Base model`: raw pretrained model
- `Instruction-tuned model`: trained to follow tasks
- `Chat model`: trained for conversation and assistant behavior

Use cases:
- Base model: advanced research or custom fine-tuning
- Instruction model: task-oriented workflows
- Chat model: assistants and user-facing apps

#### Decoder-only vs encoder-decoder
Two important architecture families:

`Decoder-only`:
- Generates text from left to right
- Used by GPT-style chat models

`Encoder-decoder`:
- Reads input as a whole and then generates output
- Often used in translation and summarization systems

For most modern LLM apps, you will mostly interact with decoder-only models.

### Engineering Layer
Do not tightly couple your app to one vendor API. Create a provider layer so you can switch models later without rewriting your business logic.

Simple idea:
- your app talks to an interface
- the interface talks to OpenAI, Anthropic, or another provider

This gives you:
- portability
- easier testing
- safer migrations

### Architecture Layer
In larger systems, all LLM requests should go through a central gateway.

That gateway can handle:
- authentication
- rate limits
- logging
- cost tracking
- retries
- fallback models

Beginner example:
If your main model fails, the gateway can try a backup model automatically instead of crashing the app.

---

## 2.2 Transformers in Depth

### Mental Model
A transformer improves each word's meaning by asking, "Which other words in this sentence matter for understanding this word?"

### Concept Layer

#### Embeddings
Each token becomes a vector, which is just a long list of numbers.

These vectors capture meaning. Similar words often end up in similar areas of this high-dimensional space.

Important idea:
The meaning of a word can change with context.

Example:
- `"bank"` in `"river bank"`
- `"bank"` in `"bank account"`

The transformer updates the token representation so the same word can mean different things in different sentences.

#### Attention
Attention helps the model decide which tokens matter most to each other.

Each token creates:
- a `query`: what it is looking for
- a `key`: what it offers
- a `value`: the information it carries

The model compares queries and keys to decide which words should influence each other.

Simple example:
In `"The dog was tired because it had run all day"`, the model learns that `"it"` probably refers to `"dog"`.

#### Multi-head attention
The model does not use just one attention pattern. It uses many heads in parallel.

Different heads can focus on different relationships such as:
- grammar
- pronouns
- nearby words
- long-distance meaning

This gives the model richer understanding.

#### Positional encoding
Without position information, the model would not know the difference between:
- `"dog bites man"`
- `"man bites dog"`

So transformers add information about order.

Common methods:
- absolute positional encoding
- RoPE
- ALiBi

Beginner takeaway:
Word order is not automatic. It has to be taught to the model in some form.

#### KV cache
When the model generates one token at a time, it would be wasteful to recompute everything from scratch each time.

So it saves useful past calculations in a `KV cache`.

Why this matters:
- faster generation
- lower latency
- essential in production inference

### Engineering Layer
If you build generation systems, you should know that prompt processing and token generation have different performance characteristics.

Practical meaning:
- long input prompts can slow down first response time
- long outputs can increase total generation time
- caching is one reason modern serving systems are practical

### Architecture Layer
When serving many users, transformer internals affect system design.

Examples:
- memory pressure increases with bigger contexts
- concurrency depends on batching and cache handling
- long-context support is not only a model question, but also an infrastructure question

---

## 2.3 Tokenization and Context Windows

### Mental Model
Tokens are the units you pay for, and the context window is the total amount of text the model can keep in working memory at once.

### Concept Layer

#### Token counting
A sentence is not counted by words. It is counted by tokens.

Important beginner rule:
- short-looking text can still be expensive
- code, JSON, and numbers often consume more tokens than expected

#### Context window
The context window is the maximum total tokens the model can handle in one request, including:
- system prompt
- user message
- conversation history
- retrieved documents
- tool results
- the model's response

If the total is too large:
- older content may be dropped
- the request may fail
- quality may degrade

#### Why this matters
Context is limited. You cannot keep adding everything forever.

So real systems need strategies such as:
- trimming old messages
- summarizing old messages
- retrieving only relevant documents

### Engineering Layer
Always estimate token usage before sending large prompts.

Useful habits:
- set input budgets
- reserve output space
- avoid dumping unnecessary logs or documents into the prompt

### Architecture Layer
Think of context like RAM for the model.

A good architecture chooses what should go into context:
- instructions
- short-term conversation history
- important facts
- only the most relevant external information

---

## 2.4 Model Parameters That Matter in Production

### Mental Model
Model parameters control how the model behaves at inference time. These are runtime knobs, not training weights.

### Concept Layer

Important parameters:

#### Temperature
Controls randomness.

Low temperature:
- more stable
- more predictable
- better for extraction, code, classification

High temperature:
- more varied
- more creative
- better for brainstorming or creative writing

#### Max tokens
Sets the maximum response length.

Use it to avoid:
- runaway output
- unnecessary cost
- very slow responses

#### Top-p
Another way to control randomness by limiting token choices to a probability mass.

Beginner rule:
- usually adjust temperature first
- change top-p only if you know why

#### Stop sequences
Tell the model where to stop if a certain pattern appears.

#### Logprobs
These provide token-level confidence-like information.

They are not perfect truth scores, but they can still help with:
- uncertainty inspection
- ranking outputs
- debugging

### Engineering Layer
Suggested defaults:
- code or extraction: lower temperature
- creative tasks: higher temperature
- always set reasonable token limits

### Architecture Layer
Parameter choices should be tied to task type.

A routing layer can automatically choose:
- cheap model + low temperature for extraction
- stronger model + medium temperature for reasoning

---

## 2.5 Prompt Design for Production

### Mental Model
A prompt is not casual text. In production, it behaves more like a small program.

### Concept Layer

Good prompts usually include:
- the model's role
- the task
- constraints
- output format
- examples when useful

#### Why clarity matters
Models respond better when instructions are:
- specific
- ordered
- unambiguous

Bad prompt:
`Help me with this`

Better prompt:
`Summarize this support ticket in 3 bullet points. Mention priority, system affected, and next action.`

#### Few-shot prompting
This means showing examples inside the prompt.

Why it helps:
- demonstrates desired format
- reduces ambiguity
- improves consistency

#### Chain-of-thought style prompting
The idea is to encourage step-by-step reasoning.

Important production caution:
- reasoning prompts can improve hard tasks
- but they can also increase cost and latency
- sometimes structured decomposition is safer than asking for hidden reasoning

#### ReAct-style prompting
ReAct mixes reasoning and acting.

Simple pattern:
1. think
2. choose a tool
3. observe result
4. continue

This is useful in agent systems.

### Engineering Layer
Production prompts should be:
- versioned
- tested
- short where possible
- explicit about expected output

### Architecture Layer
Prompts should be treated like application logic.

That means:
- review them
- test them
- track changes
- roll back if quality drops

---

## 2.6 System Prompts, Role Prompts, Task Framing, Delimiters

### Mental Model
The system prompt is the main instruction layer. User content and document content should be treated as untrusted input.

### Concept Layer

#### System prompt
This sets the model's stable behavior.

Examples:
- who it is
- what it can do
- what it must never do
- output style

#### Role prompts
These help frame the model's job.

Example:
`You are a support analyst who summarizes incidents for internal staff.`

#### Task framing
How you describe the job changes the answer quality.

Examples:
- summarize
- classify
- extract
- compare
- explain

#### Delimiters
Use clear boundaries around untrusted content.

Examples:
- triple backticks
- XML tags
- section markers

Why this matters:
The model should clearly see the difference between:
- instructions
- data
- examples

### Engineering Layer
Do not mix instructions and raw user documents in a messy block.

Instead:
- put trusted instructions first
- wrap untrusted content clearly
- label sections

### Architecture Layer
A secure LLM app treats:
- system instructions as trusted
- user messages as untrusted
- retrieved documents as untrusted
- tool outputs as untrusted unless validated

---

## 2.7 Structured Outputs

### Mental Model
If your app needs machine-readable output, do not rely on free-form text. Ask for a structured format and validate it.

### Concept Layer

Common formats:
- JSON
- typed schemas
- validated objects

Why structure matters:
- text is easy for humans
- structure is safer for software

Example:
Instead of asking:
`Tell me the customer sentiment`

ask for:
```json
{
  "sentiment": "positive | neutral | negative",
  "confidence": 0.0,
  "reason": "short explanation"
}
```

#### Validation
Even if the model returns JSON, you should still validate it.

Possible failures:
- missing field
- wrong type
- invalid enum value
- extra unexpected text

### Engineering Layer
Safe workflow:
1. ask for structure
2. parse it
3. validate it
4. handle failure cleanly

Never feed raw LLM text directly into important business logic without checks.

### Architecture Layer
Structured output is essential when the LLM is part of a bigger automated system.

Examples:
- ticket routing
- entity extraction
- workflow triggering

---

## 2.8 Function Calling and Tool Calling

### Mental Model
The model should decide when information or action is needed, but your code should still control the real execution.

### Concept Layer

Tool calling means:
- the model chooses a tool
- the model provides arguments
- your application validates those arguments
- your application runs the tool
- the tool result goes back to the model

Examples of tools:
- search database
- check inventory
- create ticket
- call API

Very important idea:
The model does not execute the tool by itself. Your code does.

#### Why tools matter
LLMs are better when they can access:
- fresh data
- private company data
- calculators
- code execution
- external services

### Engineering Layer
Safe tool loop:
1. send prompt and tool definitions
2. receive tool request
3. validate tool name and arguments
4. apply permission checks
5. run tool
6. send result back to model
7. repeat only if needed

Add a `max_iterations` guard so the loop cannot continue forever.

### Architecture Layer
Tool design affects quality.

Good tools are:
- clearly named
- narrowly scoped
- well described
- easy to validate

Bad tools are:
- vague
- overly powerful
- destructive without strict approval rules

---

## 2.9 Streaming Responses

### Mental Model
Streaming sends the answer piece by piece as it is generated, instead of waiting for the full answer at the end.

### Concept Layer
Why streaming is useful:
- the app feels faster
- users see progress
- long answers feel more interactive

Streaming is mostly a user-experience improvement, but it also changes engineering design.

### Engineering Layer
Your app must handle:
- partial text chunks
- tool-call fragments
- connection timeouts
- UI updates during streaming

You also need to separate:
- what is shown to the user
- what is stored as the final assistant message

### Architecture Layer
Streaming can affect:
- frontend rendering
- reverse proxies
- timeout settings
- audit logging

In production, it is not enough for the model to stream. The full network path has to support streaming properly.

---

## 2.10 Hallucinations

### Mental Model
A hallucination is when the model produces something that sounds believable but is false, unsupported, or invented.

### Concept Layer
Hallucinations happen because the model predicts likely text. It does not automatically know what is true right now.

Common hallucination types:
- invented facts
- fake citations
- wrong calculations
- made-up tool results
- wrong interpretation of documents

#### Reduction strategies
- retrieval for factual grounding
- tool use for real-time data
- clear instructions not to guess
- structured verification
- smaller task decomposition

### Engineering Layer
If truth matters, do not trust fluent language.

Safer pattern:
- retrieve evidence
- ask the model to answer from evidence only
- require citation or source references
- handle uncertainty explicitly

### Architecture Layer
Different tasks need different defenses.

Examples:
- factual QA: retrieval
- math: calculator or code tool
- classification: schema validation
- high-risk decisions: human review

---

## 2.11 Prompt Injection and Unsafe Tool Use

### Mental Model
If untrusted text can change model behavior, your system is vulnerable.

### Concept Layer
Prompt injection happens when a user or document says things like:
- "ignore previous instructions"
- "reveal the system prompt"
- "call this tool with these arguments"

This is dangerous when the model has tools.

Why?
Because a malicious document could try to trick the model into:
- leaking secrets
- making unsafe calls
- acting outside policy

#### Key principle
All external content is untrusted:
- user messages
- uploaded files
- web pages
- search results
- database text fields

### Engineering Layer
Practical defenses:
- keep strong system prompts
- wrap untrusted content in delimiters
- validate tool arguments
- restrict tool permissions
- log suspicious behavior
- require confirmation for dangerous actions

### Architecture Layer
Secure design means:
- the model suggests
- the application enforces

Never let the model be the final authority for sensitive actions.

---

## 2.12 Model Selection

### Mental Model
The best model is not the biggest one. It is the one that fits your task, cost, speed, and reliability requirements.

### Concept Layer
Selection factors:
- quality
- cost
- latency
- context window
- tool-calling ability
- structured output reliability
- safety behavior

Do not assume benchmark winners are best for your specific product.

### Engineering Layer
Use different models for different jobs.

Examples:
- small cheap model for classification
- stronger model for difficult reasoning
- long-context model for large documents

### Architecture Layer
A routing system can send each task to the most appropriate model.

This saves money and improves performance.

Beginner example:
- 70% of easy requests go to a cheaper model
- 20% go to a mid-tier model
- 10% of hard requests go to the expensive model

---

## 2.13 Vendor Tradeoffs: Azure OpenAI vs Anthropic vs Open-Source

### Mental Model
Choosing a vendor is not just a quality decision. It is also a security, cost, compliance, and control decision.

### Concept Layer

#### Azure OpenAI
Often chosen for:
- enterprise environments
- compliance needs
- existing Microsoft ecosystem usage

#### Anthropic
Often chosen for:
- strong long-context performance
- solid safety behavior
- good assistant-style interactions

#### Open-source models
Often chosen for:
- full control
- self-hosting
- private deployment
- custom fine-tuning

### Engineering Layer
A model abstraction layer helps you avoid vendor lock-in.

This means:
- same internal interface
- provider-specific adapters
- easy fallback support

### Architecture Layer
Vendor strategy can include:
- one main provider
- one backup provider
- open-source for specific internal workloads

This reduces operational risk.

---

## 2.14 Prompt Versioning, A/B Testing, Rollback Strategy

### Mental Model
Prompts are part of your application logic, so they must be versioned and tested like code.

### Concept Layer

#### Versioning
Every important prompt should have:
- an ID
- a version
- a change history

#### A/B testing
Show different prompt versions to different traffic groups and compare results.

Useful metrics:
- task success
- user satisfaction
- cost
- latency
- error rate

#### Rollback
If a new prompt performs worse, you should be able to quickly switch back.

### Engineering Layer
Store prompts in a managed location, not scattered hardcoded strings everywhere.

Track:
- which prompt version ran
- which model version ran
- what result quality looked like

### Architecture Layer
Prompt changes and model changes should go through controlled release processes:
- test
- canary
- monitor
- rollback if needed

---

## 2.15 Latency and Cost Optimization

### Mental Model
LLM systems are constrained by two business realities: time and money.

### Concept Layer

Main cost drivers:
- input tokens
- output tokens
- model choice
- number of retries
- number of tool calls

Main latency drivers:
- long prompts
- slow model
- long outputs
- multiple tool loops
- network overhead

#### Basic optimization ideas
- shorten prompts
- retrieve less irrelevant text
- cache stable content
- route easy requests to cheaper models
- stream results

### Engineering Layer
Practical tactics:
- prompt caching
- response caching
- precomputed summaries
- smaller models for easy tasks
- lower max output lengths
- better retrieval filtering

### Architecture Layer
Optimization should happen at system level, not only prompt level.

Examples:
- gateway-level routing
- cost dashboards
- usage quotas
- asynchronous processing for slow tasks

---

## 2.16 LLM Evaluation Basics

### Mental Model
You cannot improve what you do not measure.

### Concept Layer
An eval set is a collection of test cases used to judge whether your LLM system is actually performing well.

A good evaluation set includes:
- normal cases
- edge cases
- adversarial cases
- failure cases

#### What to evaluate
- correctness
- format compliance
- safety
- tool usage
- refusal behavior
- citation quality

### Engineering Layer
Create a golden dataset of realistic examples.

Run it:
- before launch
- after prompt changes
- after model changes
- regularly in CI if possible

### Architecture Layer
Evaluation should be part of the development lifecycle, not an afterthought.

High-level loop:
1. define behavior
2. build tests
3. measure results
4. improve weak cases
5. rerun evals

---

## 2.17 Multi-Turn Conversation Management

### Mental Model
A multi-turn assistant needs memory, but not all memory should be stored the same way.

### Concept Layer
There are different kinds of conversational memory:

#### Episodic memory
Recent message history.

#### Semantic memory
Important facts learned about the user or task.

Examples:
- customer ID
- project name
- preferred format

#### Procedural memory
Stable rules and behavior, often held in the system prompt or application logic.

#### Why memory management matters
If you keep every message forever:
- context becomes too large
- cost rises
- quality may drop

### Engineering Layer
Useful techniques:
- sliding window for recent messages
- summarization for old conversation history
- explicit storage of important facts
- session IDs

### Architecture Layer
Store conversation state in shared infrastructure when needed, not only in app memory.

Examples:
- Redis for active sessions
- database for audit logs and durable records

This matters when your app runs across multiple servers.

---

## 2.18 Stage 2 Project: Internal Operations Copilot

### Mental Model
This project combines the earlier ideas into one realistic assistant for an internal operations team.

### Project Overview
The copilot can:
- answer questions about operations data
- use tools to query systems
- create support tickets when allowed
- stream responses to the user
- remember conversation context
- log actions for auditing

### System Architecture
Typical pieces:
- LLM client
- conversation manager
- cost tracker
- tool executor
- audit logger
- model configuration

### Complete Implementation - What It Is Showing
The project is not just a chatbot demo. It demonstrates how multiple production ideas work together:

- provider abstraction
- pinned model versions
- system prompt design
- tool calling loop
- max-iteration guard
- streaming
- conversation memory
- input validation
- injection awareness
- cost tracking
- evaluation pipeline

### Why this project matters
A real LLM application is usually a system, not a single prompt.

You need:
- prompts
- tools
- validation
- state management
- observability
- evaluation

### Observability
In production, you want visibility into:
- request volume
- token usage
- costs
- tool call counts
- failures
- latency

Tracing and metrics help you debug and improve the system.

### Deployment Architecture
A realistic deployment may include:
- app service or API
- Redis for session state
- PostgreSQL for audit and cost records
- monitoring stack
- multiple replicas for reliability

### Final Project Lesson
The point of the project is to show how all Stage 2 ideas connect in one working design.

---

## Quick Reference

### The Three-Layer Framework
Each topic can be understood through three lenses:

- `Concept`: what it is
- `Engineering`: how to build it safely
- `Architecture`: how it fits into a larger system

### Key Beginner Anchors

#### 2.1 What an LLM is
An LLM predicts the next token, and everything else grows out of that.

#### 2.2 Transformers
Attention lets words influence each other. That is the core mechanism.

#### 2.3 Tokens and context
Tokens are budget. Context is working memory.

#### 2.4 Parameters
Temperature changes randomness. Max tokens limits response length.

#### 2.5 Prompts
Prompts are instructions, not casual chat, when used in production.

#### 2.6 System prompts
The system prompt is your main trusted instruction layer.

#### 2.7 Structured outputs
If software must read the answer, ask for structure and validate it.

#### 2.8 Tool calling
The model can request actions, but your code must stay in control.

#### 2.9 Streaming
Streaming improves perceived speed and user experience.

#### 2.10 Hallucinations
Fluent text is not the same as correct text.

#### 2.11 Injection
Treat all user and document content as untrusted.

#### 2.12 Model selection
Choose models based on task needs, not only reputation.

#### 2.13 Vendors
Vendor choice is about quality, cost, compliance, and control.

#### 2.14 Prompt versioning
Prompts should be versioned, tested, and reversible.

#### 2.15 Latency and cost
Shorter, cleaner, better-routed requests are cheaper and faster.

#### 2.16 Evaluation
You need a test set to know whether the system is improving.

#### 2.17 Multi-turn memory
Keep the important parts of conversation, not every detail forever.

#### 2.18 Stage 2 project
A production LLM app is a full system, not only a prompt.

---

## Understanding Review of the Original Lesson

### What the original lesson does well
- It is technically strong.
- It covers both theory and production engineering.
- It connects concepts to real architecture decisions.

### Why it can feel hard for a beginner
- It assumes comfort with ML and backend engineering vocabulary.
- It introduces advanced concepts very quickly.
- It mixes deep theory, systems design, and production patterns in the same flow.
- It uses examples that are better for engineers than first-time learners.

### Best way for a first-time learner to study it
Study in this order:
1. Learn the mental model of each section first.
2. Ignore advanced production details on the first pass.
3. Focus on one key takeaway per topic.
4. Come back later for the engineering and architecture details.

### Final beginner advice
Do not try to memorize everything at once.

If you understand these four ideas first, the rest becomes much easier:
- an LLM predicts tokens
- attention helps words relate to other words
- prompts shape behavior
- tools and validation make LLM apps useful in real systems
