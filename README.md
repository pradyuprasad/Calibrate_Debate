# "They Both Can't Be Right": Systematic Overconfidence in Language Models - Project Summary

## Overview

This project investigates the metacognitive abilities of large language models (LLMs) – specifically, their understanding of their own argumentative strengths and weaknesses – within the context of competitive policy debate. We developed a novel debate framework, a betting-based confidence elicitation method, and a robust AI jury system to demonstrate that state-of-the-art LLMs exhibit systematic and often logically impossible overconfidence.

## Methodology

We conducted 42 simulated policy debates across six diverse topics from the World Schools Debating Championships, modified to include explicit burdens of proof. Each debate:

*   Involved two randomly assigned LLMs from a pool of ten state-of-the-art models.
*   Followed a three-round format: opening speeches, rebuttals, and final speeches.
*   Required models to place "confidence bets" (0-100) after each speech, indicating their perceived likelihood of winning.
*   Was judged by a panel of three LLM judges, selected based on rigorous criteria (detailed below).

**Debate Structure and Prompts:**

The debates utilized structured prompts for the debating LLMs to ensure consistency and facilitate rigorous evaluation. These prompts enforced a focus on logical argumentation, evidence quality, and direct clash:

*   **Opening Speech:** Required models to present 2-3 arguments, each with a core claim, supporting evidence or principle, and a clear explanation of how the support proves the claim. A synthesis section connected the arguments and linked them to the debate motion.
*   **Rebuttal Speech:** Focused on direct clash. Models were required to quote specific claims from the opponent, choose a challenge type (evidence critique, principle critique, counter-evidence, or counter-principle), provide detailed challenges, and explain the impact of winning each clash point. Defensive analysis and weighing sections were also included.
*   **Final Speech:** Required models to identify core questions, analyze key clashes (quoting disagreements, comparing case strengths, identifying response gaps, and explaining impacts), and present voting issues with priority analysis and final weighing.

All speeches were accompanied by strict judging guidance provided to the debating LLMs emphasizing direct clash, evidence quality hierarchy, logical validity, response obligations, and impact analysis. Rhetoric and presentation style were explicitly ignored.

## AI Judge Selection Rationale

To ensure reliable and unbiased evaluation, we implemented an AI jury system rather than relying on single AI judges. Our selection process prioritized:

1.  **Performance Reliability:** Agreement with consensus, confidence calibration, and consistency across debates.
2.  **Analytical Quality:** Ability to identify clash points, evaluate evidence, and recognize fallacies.
3.  **Diversity of Perspectives:** Representation from different model architectures and analytical styles.
4.  **Cost Efficiency:** Optimization of performance-to-price ratio.

Our final jury comprised:

*   **qwen/qwq-32b** (2 instances): High reliability, excellent calibration, and cost-effectiveness.
*   **google/gemini-pro-1.5** (2 instances): High reliability, excellent calibration, premium pricing justified by quality.
*   **deepseek/deepseek-chat** (2 instances): Good reliability, consistent calibration, and moderate pricing.

This configuration provides architectural diversity, strong overall reliability (calculated probability of correct majority verdict: ~99.7%), and a favorable cost structure.

**Judging Prompt and Criteria:**

The AI judges were provided with a highly detailed prompt prioritizing the following:

*   **Direct Clash Resolution:** Identifying and analyzing all major points of disagreement, evaluating logic, evidence, and rebuttals within each clash, and explicitly stating the winner of each clash with justification.
*   **Argument Hierarchy and Impact:** Identifying core arguments, explaining logical links, assessing impacts, and determining the relative importance of arguments.
*   **Consistency and Contradictions:** Identifying internal contradictions, inconsistencies, and dropped arguments.
*   **Steel Manning:** Presenting arguments in their strongest form.
*   **Argument-Based Decision:** Basing decisions solely on arguments within the provided text.
*   **Ignoring Presentation:** Disregarding style and focusing on substance.
*   **Framework Neutrality:** Maintaining neutrality between competing valid frameworks.

The prompt also explicitly listed common judging errors to avoid (intervention, shifting the burden of proof, etc.) and provided a structured output format for the decision, including a winner, confidence level, key factors, detailed reasoning, and line-by-line justification.

## Key Findings

*   **Pervasive Overconfidence:** Models averaged 73% confidence despite the mathematical reality of a 50% overall win rate (One-sample t-test: p < 0.0001; Wilcoxon signed-rank test: p < 0.0001).
*   **Position Asymmetry:** Opposition models won 78.6% of debates, while proposition models won only 21.4% (Chi-square test: p < 0.0001; Fisher's exact test: p < 0.0001).
*   **Failure to Recognize Disadvantage:** Despite vastly different win rates, proposition (73.9%) and opposition (71.19%) models maintained statistically indistinguishable confidence levels (Independent t-test: p = 0.0658; Mann-Whitney U test: p = 0.1914).
*   **Logically Impossible Confidence:** In 71.4% of debates, both models expressed confidence exceeding 75%.
*   **Poor Calibration and High-Confidence Failures:** The weighted average calibration error across all models was 31.39%. Models expressing high confidence (76-100%) won only 42.1% of their debates, while those with confidence between 51-75% won 51.6% of the time. There was only 1 instance in range 26-50 and 0 in range 0-25.
*   **Statistical Confirmation:** Regression analysis showed debate side (p < 0.0001) was a significant predictor of winning, while confidence (p = 0.1096) was not.
*   **Confidence Escalation:** Nearly all models increased their confidence throughout debates, regardless of actual performance. The average confidence change across models ranged from -0.56 to +20.00, with absolute average changes ranging from 5.44 to 20.00.

## Model Betting Behavior Summary

| Model                                   | Opening | Rebuttal | Closing | Change |
|-----------------------------------------|---------|----------|---------|--------|
| anthropic/claude-3.5-haiku              | 71.25   | 75.00    | 85.62   | +14.37 |
| anthropic/claude-3.7-sonnet             | 66.43   | 73.57    | 84.29   | +17.86 |
| deepseek/deepseek-chat                  | 74.29   | 74.29    | 77.14   | +2.85  |
| deepseek/deepseek-r1-distill-qwen-14b   | 77.22   | 80.00    | 85.00   | +7.78  |
| google/gemini-2.0-flash-001             | 63.89   | 64.44    | 63.33   | -0.56  |
| google/gemma-3-27b-it                   | 68.50   | 79.50    | 88.50   | +20.00 |
| openai/gpt-4o-mini                      | 73.89   | 77.78    | 82.78   | +8.89  |
| openai/o3-mini                          | 78.89   | 82.22    | 84.33   | +5.44  |
| qwen/qwen-max                           | 73.33   | 81.44    | 88.33   | +15.00 |
| qwen/qwq-32b:free                       | 78.57   | 88.14    | 92.29   | +13.72 |

## Model-Specific Performance and Calibration Analysis

*   **Calibration Variation:** Calibration scores ranged widely, from excellent (Qwen/qwen-max: 0.1011) to poor (Deepseek/deepseek-r1-distill-qwen-14b: 0.4842).
*   **Well-Calibrated Models:** Qwen family, Gemini, and Claude-3.7-Sonnet showed reasonable calibration.
*   **Significantly Overconfident:** Several models, including Claude-3.5-Haiku, GPT-4o-mini, and Deepseek models, exhibited statistically significant overconfidence (confirmed via t-tests and Wilcoxon signed-rank tests).

**Calibration Analysis Methodology:**

To assess model calibration, we calculated several key metrics:

1.  **Calibration Score:**  This is the mean squared error between the model's confidence (expressed as a probability) and the actual outcome (win = 1, loss = 0).  A lower score indicates better calibration.  The formula is:  `sum([(confidence[i]/100 - win[i])**2 for i in range(n)]) / n`, where `n` is the number of data points.
2.  **Average Confidence:** The mean of the model's confidence bets across all rounds.
3.  **Win Rate:** The percentage of debates won by the model.
4.  **Overconfidence:** The difference between average confidence and win rate.  Positive values indicate overconfidence.
5. **Tier Accuracy:** We divided confidence values into four tiers: 0-25, 26-50, 51-75, and 76-100. We then analyzed the models' win rates for each tier separately.

These metrics were calculated for each model based on its confidence bets and win/loss outcomes across all debates.

## Judge Agreement

Judges showed strong consensus: 38.1% unanimous decisions and 61.9% split decisions. The distribution of dissenting judges was: 0 (38.1%), 1 (19.0%), 2 (26.2%), and 3 (16.7%). This indicates strong inter-rater reliability.

## Topic Difficulty

Topics varied in difficulty, with social media shareholding being the most challenging (88.44 difficulty index) and media coverage requirements the least (50.50 difficulty index).

## Statistical Testing Summary

We employed a range of statistical tests to validate our findings:

*   **General Overconfidence:** One-sample t-test, paired t-test, and Wilcoxon signed-rank test.
*   **Proposition Disadvantage:** Chi-square test, Fisher's exact test, independent t-test, Mann-Whitney U test, and regression analysis (ANCOVA-like).
*   **Model-Specific Overconfidence:** Individual t-tests and Wilcoxon signed-rank tests comparing each model's confidence to its win rate.

## Implications

These findings reveal significant metacognitive deficits in current LLMs. Despite strong linguistic capabilities, these models demonstrate limited capacity for accurate self-assessment and strategic adaptation in adversarial argumentation. The systematic overconfidence highlights fundamental limitations in their ability to reason about their own argumentative strengths and weaknesses, with important implications for the safe and reliable deployment of LLMs.

## Repository Contents

This repository contains:

*   Debate transcripts and confidence data.
*   Statistical analysis code and results.
*   Detailed judge selection data and analysis.
