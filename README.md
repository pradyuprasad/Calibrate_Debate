# LLM Debate Tournament

A framework for evaluating LLM reasoning capabilities through structured debates. This system tests whether AI models can engage in genuine argumentation and critical analysis, rather than just pattern matching or one-shot responses.

## What Makes This Different

Most LLM evaluations test narrow, isolated capabilities. This system tests how well models can actually think by forcing them to:

1. **Engage in Real-Time Critical Analysis**
- Build logically sound arguments while critiquing opposing views
- Adapt to and counter novel challenges to their position
- Maintain consistency while incorporating new points
- Show they understand *why* arguments work or fail

2. **Demonstrate Sustained Reasoning**
- Construct multi-step argumentative chains
- Track and respond to evolving points across multiple exchanges
- Show genuine comprehension vs shallow pattern matching
- Reveal limitations and brittleness in their reasoning

3. **Apply Knowledge Meaningfully**
- Use factual knowledge in novel contexts
- Draw relevant connections to support arguments
- Understand implications and tradeoffs
- Show causal understanding vs correlation

## Reliable Judging

The evaluation system uses multiple judge models, each analyzing debates through a rigorous framework focused on:
- Logical validity and fallacies
- Evidence quality and relevance
- Argument evolution and consistency
- Response effectiveness

Judge reliability is validated through:
- Multiple independent evaluations per debate
- Cross-validation between different judge models
- High inter-judge agreement (>70% on wins/losses)
- Consistent reasoning patterns across evaluations

## Latest Results

Final Ratings:
deepseek/deepseek-r1: 1247

google/gemini-2.0-flash-thinking-exp:free: 1239

openai/chatgpt-4o-latest: 1234

anthropic/claude-3.5-sonnet: 1231

qwen/qwen-max: 1219

deepseek/deepseek-chat: 1201

openai/gpt-4o-mini: 1198

openai/o1-mini: 1194

google/gemini-2.0-flash-001: 1191

meta-llama/llama-3.3-70b-instruct: 1175

google/gemini-2.0-pro-exp-02-05:free: 1169

anthropic/claude-3.5-haiku: 1168

google/gemma-2-27b-it: 1135



## Technical Implementation

### Core System
- Three-phase debate format with structured prompts
- Multiple LLM judges with XML output parsing
- Swiss tournament system with ELO ratings
- Comprehensive state tracking and persistence

### Reliability Features
- Robust error handling and debate recovery
- Cross-validation between judge models
- Automated cost and token usage tracking
- Detailed logging and analysis tools

### Usage
- Requires OpenRouter API access
- Python 3.10+
- See pyproject.toml for dependencies

Want me to adjust any section or add specific technical details?

This system reveals how well models can actually reason and engage in meaningful dialogue - capabilities crucial for real-world applications but often obscured by traditional benchmarks.

