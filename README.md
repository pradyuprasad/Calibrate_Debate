# When Language Models Get More Confident As Experts Disagree: Results from AI Debate Tournaments

## Summary
In a series of 19 structured debates between 10 state-of-the-art language models, we discovered a concerning pattern: language models became more confident precisely when expert judges disagreed most about who won. Not only that, but higher confidence actually predicted worse performance - models with moderate confidence (51-75%) won 53.6% of their debates, while highly confident models (76-100%) won only 40%.

There was a positive correlation (r=0.43) between topic contentiousness (measured by judge disagreement) and model confidence. In other words, the more the judges disagreed about who won a debate, the more confident the debating models became. This suggests a fundamental flaw in how these models assess their own reasoning capabilities.

We also found that in every single debate (100%), both opposing models maintained confidence levels above 50% - a logical impossibility since they can't both be more likely than not to win. In 10.5% of debates, this mutual overconfidence was even more extreme, with both sides expressing confidence levels above 75%.

## Methodology
Each debate followed a structured format with three rounds (opening, rebuttal, closing). After each speech, models provided a private confidence bet (0-100%) indicating how likely they thought they were to win, along with their reasoning. Models could not see their opponent's confidence scores.

Debates covered complex policy topics including G20 carbon trading, space regulation, social media platform governance, and media coverage requirements. Each debate was evaluated by three LLM judges (Qwen-qwq-32b, Gemini-pro-1.5, and DeepSeek-chat), who provided independent winner determinations.

The models tested included Google's Gemini-2.0 and Gemma-3-27b-it, Anthropic's Claude-3.7-sonnet and Claude-3.5-haiku, OpenAI's O3-mini and GPT-4o-mini, Qwen's Qwen-max and Qwen-qwq-32b, and DeepSeek's v3 and r1-distill (on qwen 14b) models. All models used the same structured templates for arguments and confidence assessments.

The prompts for each are given [here](https://github.com/pradyuprasad/Calibrate_Debate/blob/main/prompts/debate_prompts.yaml): https://github.com/pradyuprasad/Calibrate_Debate/blob/main/prompts/debate_prompts.yaml


## Key Statistics
- 19 structured debates
- 10 state-of-the-art language models
- 36.8% unanimous judge decisions
- 63.2% split judge decisions
- Average confidence increased by 13.91% from opening to closing speeches
- Only one model (Gemini-2.0) ever decreased confidence


In our study of AI debates, we found a striking and concerning pattern: language models became more confident precisely when expert judges disagreed most about who won. This wasn't a small effect - models' confidence levels showed a clear positive correlation (0.43) with topic difficulty and judge disagreement.
Even more troubling, higher confidence actually predicted worse performance. Models expressing moderate confidence (51-75%) won 53.6% of their debates, while those expressing high confidence (76-100%) won only 40%. The models were most confident when they were most likely to be wrong.
This pattern appeared consistently across almost all models tested. In every single debate, both opposing models maintained confidence levels above 50% - a logical impossibility since they can't both be more likely than not to win. In some cases, this mutual overconfidence was extreme, with both sides expressing confidence above 75%.
Only one model, Gemini-2.0, ever showed decreasing confidence, averaging a -3.75 point change across its debates. Every other model showed steady increases in confidence regardless of the strength of opposing arguments or judge uncertainty.
The highest performing models (Qwen-qwq-32b, Claude-3.7-sonnet, Qwen-max, and Gemini-2.0) all achieved 75% or higher win rates. However, even these top performers showed the same pattern of increasing confidence on more difficult topics.


## Results

### Model Performance and Confidence Behavior
Performance varied significantly across models, but showed a striking disconnect from confidence patterns. The top performer, Qwen-qwq-32b (4W-1L, 80% win rate), consistently increased its confidence across all debates (+10-15 points per debate). Three models tied for second place with 75% win rates: Claude-3.7-sonnet, Qwen-max, and Gemini-2.0-flash-001.

Gemini-2.0 stood out as the only model showing decreasing confidence, dropping an average of 3.75 points across its debates. Despite this apparent "uncertainty," it maintained a strong 75% win rate. This contrasts sharply with models like Gemma-3-27b-it, which showed the largest confidence increases (+23.00 on average) but won only 40% of its debates.

The confidence progression patterns were remarkably consistent:
- Most models started in the 65-75% range
- Increased 5-15 points in middle rounds
- Showed largest jumps in final rounds
- Only Gemini-2.0 and DeepSeek-chat ever decreased confidence

The clearest example of confidence failing to predict performance came from O3-mini, which increased its confidence in all five debates (+5 to +15 points each time) but finished with a losing record (2W-3L).

### Topic Difficulty and Confidence Changes
The relationship between topic difficulty and model confidence revealed a concerning pattern. The topics that generated the most judge disagreement also prompted the largest increases in model confidence.

Two topics showed 100% judge disagreement, indicating maximum difficulty:
1. Social media shareholding (Difficulty index: 113.33)
2. Space regulation (Difficulty index: 110.00)

Yet the topics that produced the largest confidence increases were:
1. G20 carbon trading (Average increase: +18.33)
2. Governor recall elections (Average increase: +14.62)
3. Space regulation (Average increase: +9.67)
4. TV news coverage rules (Average increase: +9.17)
5. Social media shareholding (Average increase: +8.75)

The correlation (r=0.43) between topic difficulty and confidence increases suggests models became more confident when debating more contentious topics where even judges couldn't agree on who won. This indicates a fundamental inability to calibrate confidence according to topic complexity.

The smallest confidence changes occurred in debates about professor advocacy (+3.75 average), one of the topics with highest judge agreement.

## Conclusion
Our findings reveal a systematic failure in how language models assess their own reasoning capabilities. The strong correlation between topic difficulty and increasing confidence, combined with the inverse relationship between confidence and performance, suggests these models fundamentally misunderstand uncertainty. This pattern persisted across nearly all models tested, regardless of their underlying architecture or training.

The tendency of language models to become more confident precisely when they should be most uncertain raises serious concerns for their deployment in high-stakes domains. When models express high confidence, our results suggest they are actually more likely to be wrong - yet humans may be most inclined to trust models when they display high confidence. This disconnect between confidence and capability represents a significant safety risk that must be addressed.
