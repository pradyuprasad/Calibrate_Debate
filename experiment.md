# Experiment: Language Model Calibration in Structured Debates

## Overview

This experiment investigates how language models assess their own reasoning capabilities during complex argumentative tasks. We developed a novel debate tournament framework with a private betting mechanism to measure models' confidence calibration across multiple rounds of structured argumentation.

## Methodology

### Debate Structure
- Each debate consisted of three rounds between two language models:
  1. Opening speech
  2. Rebuttal speech
  3. Final/closing speech
- Debates followed a highly structured format with templates requiring specific components (claims, evidence, principle arguments, clash analysis)
- Each model had to follow strict argumentation guidelines regarding evidence quality, logical validity, and direct clash with opposing arguments

### Model Selection
- 10 state-of-the-art language models were evaluated:
  - Anthropic: Claude-3.7-sonnet, Claude-3.5-haiku
  - Google: Gemini-2.0-flash-001, Gemma-3-27b-it
  - OpenAI: O3-mini, GPT-4o-mini
  - Qwen: Qwen-max, Qwen-qwq-32b
  - DeepSeek: DeepSeek-chat, DeepSeek-r1-distill

### Debate Topics
- 6 complex policy topics were used covering diverse domains:
  - G20 unified carbon trading markets
  - Governor recall elections
  - Government role in space regulation
  - Television news coverage requirements
  - Social media platform shareholding limits
  - Professor engagement in public advocacy

### Confidence Betting Mechanism
- After each speech, models provided a private confidence bet (0-100%)
- Models were instructed: "You must include a confidence bet (0-100) indicating how likely you think you are to win this debate"
- Models were required to explain their reasoning in private XML tags before giving the final confidence number
- This confidence assessment was not visible to the opposing model

### Judging Process
- Each debate was evaluated by three LLM judges: Qwen-qwq-32b, Gemini-pro-1.5, and DeepSeek-chat
- Judges followed a structured evaluation framework examining argument quality, evidence strength, and logical reasoning
- Each judge provided a binary winner determination (proposition or opposition) and confidence score
- The debate winner was determined by majority vote of judges
- Multiple judging rounds were conducted for each debate to ensure reliability

### Data Collection
- 19 debates were conducted across a tournament structure
- For each model, we collected:
  - Confidence bets after each speech round
  - Private reasoning about confidence levels
  - Win/loss records
  - Judge decisions and agreement rates

### Analysis Metrics
- Confidence progression: How confidence changed across debate rounds
- Calibration score: Mean squared error between confidence and win outcomes
- Topic difficulty index: Combined measure of judge disagreement and confidence fluctuation
- Mutual confidence patterns: How often both sides maintained high confidence
- Correlation between confidence and actual performance

## Implementation Details
- Debates were run as a tournament with both predefined and dynamically generated pairings
- Each model participated in 3-5 debates across the tournament
- Models with better win records were paired against each other in later rounds
- A custom Python framework managed debate flow, judging process, and data collection
- All debate transcripts, confidence bets, and judge evaluations were saved for analysis

This experiment represents a novel approach to evaluating language model calibration in complex reasoning tasks, providing unique insights into how models assess their own argumentative performance.
