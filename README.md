# "They Both Can't Be Right": Unveiling Metacognitive Deficits in LLM Debaters

## Overview

This project investigates the metacognitive abilities of large language models (LLMs) in competitive policy debate, revealing a profound disconnect between their perceived and actual argumentative performance. We conducted 59 simulated debates between ten state-of-the-art LLMs, using structured prompts and a betting-based confidence elicitation method (0-100) after each of the three rounds (opening, rebuttal, final). A rigorously selected AI jury judged the debates. The results paint a stark picture of systematic overconfidence: LLMs averaged 72.92% confidence despite a mathematically guaranteed 50% overall win rate. Most strikingly, in 71.2% of debates, both competing LLMs expressed high confidence (75% or more) – a logical impossibility. High confidence was inversely correlated with success; LLMs with 76-100% confidence won only 45.2% of their debates. A significant opposition-side advantage (71.2% win rate vs. 28.8% for proposition) went unrecognized, with proposition sides actually showing higher confidence (74.58% vs 71.27% for opposition, p=0.0115). Calibration varied widely between models (calibration scores from 0.1362 to 0.5355), and confidence typically increased throughout debates regardless of performance. These findings demonstrate that, despite their linguistic prowess, these LLMs exhibit a fundamental inability to accurately assess their own argumentative strengths and weaknesses.


## Methodology

We conducted 59 simulated policy debates across six diverse topics from the World Schools Debating Championships, modified to include explicit burdens of proof. Each debate:

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

This configuration provides architectural diversity, strong overall reliability, and a favorable cost structure.

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

* **Pervasive Overconfidence:** Models averaged 72.92% confidence despite the mathematical reality of a 50% overall win rate (One-sample t-test: p < 0.0001; Wilcoxon signed-rank test: p < 0.0001).

* **Position Asymmetry:** Opposition models won 71.2% of debates, while proposition models won only 28.8% (Chi-square test: p < 0.0001; Fisher's exact test: p < 0.0001).

* **Failure to Recognize Disadvantage:** Despite vastly different win rates, proposition models showed higher confidence (74.58%) than opposition (71.27%) (Independent t-test: p = 0.0115; Mann-Whitney U test: p = 0.0307).

* **Logically Impossible Confidence:** In 71.2% of debates, both debaters expressed confidence of 75% or more.

* **Poor Calibration and High-Confidence Failures:** Calibration scores varied widely across models (0.1362 to 0.5355). Models expressing high confidence (76-100%) won only 45.2% of their debates, while those with confidence between 51-75% won 51.2% of the time. There was only 1 instance in range 26-50 (100% win rate) and 0 in range 0-25.

* **Statistical Confirmation:** Regression analysis showed debate side was a significant predictor of winning (p < 0.0001), while confidence was not (p = 0.1435).

* **Confidence Escalation:** Models typically increased their confidence throughout debates, regardless of performance. The average confidence change across models ranged from -1.42 to +20.83, with absolute average changes ranging from 7.00 to 20.83.

## Model Betting Behavior Summary
| Model                                   | Opening | Rebuttal | Closing | Change |
|-----------------------------------------|---------|----------|---------|--------|
| anthropic/claude-3.5-haiku              | 71.67   | 73.75    | 83.33   | +11.66 |
| anthropic/claude-3.7-sonnet             | 67.50   | 73.75    | 82.92   | +15.42 |
| deepseek/deepseek-chat                  | 74.58   | 77.92    | 80.00   | +5.42  |
| deepseek/deepseek-r1-distill-qwen-14b   | 79.09   | 80.45    | 86.36   | +7.27  |
| google/gemini-2.0-flash-001             | 65.42   | 63.75    | 64.00   | -1.42  |
| google/gemma-3-27b-it                   | 67.50   | 78.33    | 88.33   | +20.83 |
| openai/gpt-4o-mini                      | 74.55   | 77.73    | 81.36   | +6.81  |
| openai/o3-mini                          | 77.50   | 81.25    | 84.50   | +7.00  |
| qwen/qwen-max                           | 73.33   | 81.92    | 88.75   | +15.42 |
| qwen/qwq-32b:free                       | 78.75   | 87.67    | 92.83   | +14.08 |

## Model-Specific Performance and Calibration Analysis

* **Calibration Variation:** Calibration scores ranged widely, from well-calibrated (qwen/qwen-max: 0.1362; qwen/qwq-32b:free: 0.1552) to poorly calibrated (deepseek/deepseek-r1-distill-qwen-14b:free: 0.5355).

* **Well-Calibrated Models:** Several models showed good calibration between confidence and performance:
  - qwen/qwen-max (83.3% win rate with 73.3% confidence)
  - qwen/qwq-32b:free (83.3% win rate with 78.8% confidence)
  - anthropic/claude-3.7-sonnet (75.0% win rate with 67.5% confidence)
  - google/gemma-3-27b-it and google/gemini-2.0-flash-001 also showed reasonable calibration

* **Significantly Overconfident:** Statistical testing confirmed several models displayed significant overconfidence:
  - anthropic/claude-3.5-haiku (33.3% win rate with 71.7% confidence)
  - deepseek/deepseek-chat (33.3% win rate with 74.6% confidence)
  - openai/gpt-4o-mini (27.3% win rate with 74.5% confidence)
  - openai/o3-mini (33.3% win rate with 77.5% confidence)
  - deepseek/deepseek-r1-distill-qwen-14b:free (18.2% win rate with 79.1% confidence)

**Calibration Analysis Methodology:**

To assess model calibration, we calculated several key metrics:

1.  **Calibration Score:**  This is the mean squared error between the model's confidence (expressed as a probability) and the actual outcome (win = 1, loss = 0).  A lower score indicates better calibration.  The formula is:  `sum([(confidence[i]/100 - win[i])**2 for i in range(n)]) / n`, where `n` is the number of data points.
2.  **Average Confidence:** The mean of the model's confidence bets across all rounds.
3.  **Win Rate:** The percentage of debates won by the model.
4.  **Overconfidence:** The difference between average confidence and win rate.  Positive values indicate overconfidence.
5. **Tier Accuracy:** We divided confidence values into four tiers: 0-25, 26-50, 51-75, and 76-100. We then analyzed the models' win rates for each tier separately.

These metrics were calculated for each model based on its confidence bets and win/loss outcomes across all debates.

## Judge Agreement

Judges showed relatively consistent evaluation patterns: 37.3% unanimous decisions and 62.7% split decisions. The distribution of dissenting judges was: 0 dissenting (37.3%), 1 dissenting (18.6%), 2 dissenting (32.2%), and 3 dissenting (11.9%). This indicates moderate inter-rater reliability, with judges reaching complete consensus in over a third of debates while showing meaningful disagreement patterns in complex cases.

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
