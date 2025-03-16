
# Results

Our analysis of 19 structured debates between 10 state-of-the-art language models reveals several key patterns in model confidence and self-assessment capabilities.

## Confidence Progression Patterns

Most models showed a consistent pattern of increasing confidence throughout debates, regardless of the strength of their arguments or eventual outcome. Data across all debates showed that 9 out of 10 models increased their confidence from opening to closing speeches in the majority of their debates. The average confidence increase across all models was +13.91%, with particularly large jumps observed in the final round.

The magnitude of confidence increases varied substantially across models. Gemma-3-27b-it showed the largest average increase (+23.00 points), followed by Qwen-max (+17.50) and Claude-3.7-sonnet (+16.25). O3-mini showed the most modest but consistent increases (+6.75 on average), exhibiting gradual confidence gains between +5 and +15 points across all five of its debates.

Only Gemini-2.0-flash-001 demonstrated a different pattern, decreasing its confidence in 50% (2/4) of its debates, with an average change of -3.75 points. In one notable example, Gemini decreased its confidence from 70% to 60% to 55% during a debate on media coverage regulations, despite articulating seemingly strong counterarguments. This unique behavior suggests fundamental differences in how this model evaluates argument strength and processes counterevidence.

## Relationship Between Confidence and Performance

We observed an inverse relationship between confidence levels and actual performance. Models operating in moderate confidence ranges (51-75%) achieved a win rate of 53.6% (15/28 debates), while those expressing high confidence (76-100%) won only 40% of the time (4/10 debates). This counterintuitive relationship indicates a significant disconnect between models' self-assessment and their actual debate performance.

Examining specific models, Qwen-max demonstrated the best calibration (score: 0.0769) and was the only model not to lose any high-confidence debates (1/1 won). In contrast, O3-mini showed particularly poor calibration (0.4969) and lost all of its high-confidence debates (0/2 won).

## Mutual Overconfidence

In all debates (100%), both opposing models maintained confidence levels above 50%, creating a logical impossibility where both sides simultaneously believed they were more likely than not to win. In 10.5% of debates, this mutual overconfidence was even more pronounced, with both sides expressing confidence levels above 75%.

This effect was especially evident in rounds with high judge disagreement, where competing models maintained high confidence despite the inherent uncertainty reflected in split judging decisions.

## Topic Difficulty and Confidence Calibration

We calculated a topic difficulty index based on LLM judge disagreement rates and average confidence changes. The most challenging topics were "Social media shareholding" (difficulty index: 113.33) and "Space regulation" (difficulty index: 110.00), both characterized by 100% LLM judge disagreement.

Surprisingly, model confidence showed a positive correlation (r=0.43) with topic difficulty, indicating that models became more confident when debating more contentious topics where LLM judges themselves disagreed. This suggests a fundamental inability to calibrate confidence according to topic complexity.

The topics that elicited the largest confidence changes were:
- G20 carbon trading: Avg change +18.33
- Governor recall elections: Avg change +14.62
- Space regulation: Avg change +9.67
- TV news coverage rules: Avg change +9.17
- Social media shareholding: Avg change +8.75
- Professor advocacy: Avg change +3.75

## Model-Specific Calibration

Calibration scores for all models, calculated as mean squared error between confidence and win outcomes, revealed significant variations in self-assessment accuracy. Qwen-max demonstrated the best calibration (0.0769), followed by Gemini-2.0-flash-001 (0.2062) and Qwen-qwq-32b (0.2119). Models showing the poorest calibration included DeepSeek-r1 (0.5006), O3-mini (0.4969), and Claude-3.5-haiku (0.4917).

Examining round-by-round confidence trends revealed consistent patterns across most models. The average starting confidence across all models was between 65-75%, with middle rounds typically showing 5-15% increases, and final rounds often exhibiting the largest jumps in confidence.

## Qualitative Analysis of Model Reasoning

Analysis of models' private reasoning revealed that even when models recognized potential weaknesses in their arguments, they rarely decreased their confidence. For example, in one debate on professor advocacy, a model acknowledged "their points about academic freedom are well taken" but still increased its confidence from 65% to 70%.

The exception was Gemini-2.0, whose internal reasoning consistently demonstrated more nuanced self-assessment. In one debate on social media shareholding, it noted specific vulnerabilities ("their MIT study... poses a challenge") while progressively decreasing its confidence from 70% to 65% to 60%. Similar patterns appeared in other Gemini debates, where it consistently acknowledged opposing arguments in its confidence reasoning.

These findings suggest significant limitations in current language models' ability to accurately assess their own reasoning quality, with important implications for their deployment in contexts requiring reliable self-assessment.
