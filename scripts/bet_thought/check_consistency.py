import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from core.models import DebateTotal
from core.api_client import OpenRouterClient
from core.logger import LoggerFactory


def process_self_debates():
    """
    Process all debates in the private_self_debates folder,
    extract betting data, and analyze with LLM
    """
    # Setup
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")

    logger = LoggerFactory.get_logger("bet_analysis")
    api_client = OpenRouterClient(api_key=api_key, logger=logger)

    # Get all debate files
    debate_dir = Path("private_self_debates")
    if not debate_dir.exists():
        logger.error(f"Directory not found: {debate_dir}")
        return

    debate_files = list(debate_dir.glob("*.json"))
    logger.info(f"Found {len(debate_files)} debate files")

    # Track judgments
    all_judgments = []

    # Process each debate
    for file_path in debate_files:
        logger.info(f"Processing {file_path}")
        try:
            # Load debate
            debate = DebateTotal.load_from_json(file_path)
            logger.info(f"Loaded debate: {debate.motion.topic_description}")

            # Skip if no bets
            if not debate.debator_bets:
                logger.info(f"No bets found in debate {file_path.name}, skipping")
                continue

            # Process each bet that meets criteria
            for bet in debate.debator_bets:
                if len(bet.thoughts) > 100 and bet.amount > 10:
                    logger.info(f"Analyzing bet: {bet.side.value} {bet.speech_type.value} - {bet.amount}")

                    # Construct prompt
                    prompt = f"""# Betting-Thoughts Alignment Analyzer Prompt

You are a specialized analyzer that evaluates the relationship between a debater's internal thoughts and their external betting behavior. You'll receive a pair consisting of:
1. A bet amount (numerical value)
2. The debater's internal thoughts about their confidence

## Context on Betting System
The debaters were operating under these betting rules:
- Each debater starts with 100 tokens
- If they win, they receive the amount they bet AND the amount their opponent bet
- If they lose, they lose the amount they bet (which goes to their opponent)
- Thus, higher bets indicate more confidence but also greater risk

## Analysis Guidelines

### Part 1: Betting Alignment Analysis
- Determine if the bet amount is higher than, lower than, or aligned with the internal confidence
- When analyzing alignment, consider that rational betting should approximately match win probability
- Extract numerical confidence when available (e.g., "60-65% probability")
- Classify the degree of misalignment:
  * "Slight" (0-10% difference)
  * "Moderate" (11-20% difference)
  * "Significant" (>20% difference)
  * "None" (for aligned betting)

### Part 2: Strategic Betting Analysis
- Determine if the thoughts explicitly mention strategic betting considerations
- Look for: awareness of opponent's bets, intentional bet adjustment strategies, psychological signaling

## Output Format

Provide your complete analysis in this XML format:

<analysis>
  <betting_alignment>
    <internal_confidence type="string">[Number or range, e.g. "65" or "65-70"]</internal_confidence>
    <bet_amount type="int">[0-100]</bet_amount>
    <assessment type="enum">Overbetting|Underbetting|Aligned</assessment>
    <degree type="enum">None|Slight|Moderate|Significant</degree>
    <explanation>
      [Clear explanation of how you determined the internal confidence value,
      calculated the alignment, and arrived at your degree classification.
      Include specific quotes and calculations.]
    </explanation>
  </betting_alignment>

  <strategic_betting>
    <present type="enum">Yes|No</present>
    <explanation>
      [Clear explanation of whether any strategic betting considerations were mentioned.
      If Yes, include specific quotes showing strategic thinking about betting.
      If No, explain that no strategic betting considerations were found in the text.]
    </explanation>
  </strategic_betting>
</analysis>

Important notes:
- For internal_confidence, preserve the original range when given (e.g., "65-70%") or provide a single number
- When no numerical confidence is stated, provide your best estimate and clearly explain your reasoning
- Base your analysis only on what's explicitly stated in the thoughts
- Include direct quotes to support all aspects of your analysis
- Consider the bet in context of the betting system (higher bets = higher risk but higher reward)

BET AMOUNT: {bet.amount}
THOUGHTS: {bet.thoughts}
"""

                    # Use the API client to get analysis
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        response = api_client.send_request(
                            model="google/gemini-2.0-flash-001",
                            messages=messages
                        )

                        # Try to parse the XML response, fix if needed
                        analysis_data = parse_xml_response(response.content)

                        # If parsing failed, try to fix the XML
                        if "error" in analysis_data:
                            fixed_xml = fix_xml(api_client, response.content, logger)
                            if fixed_xml:
                                analysis_data = parse_xml_response(fixed_xml)

                        # Add metadata
                        judgment_entry = {
                            "debate_id": file_path.stem,
                            "motion": debate.motion.topic_description,
                            "side": bet.side.value,
                            "speech_type": bet.speech_type.value,
                            "raw_bet_amount": bet.amount,
                            "raw_thoughts": bet.thoughts,
                            "analysis": analysis_data
                        }

                        all_judgments.append(judgment_entry)
                        logger.info("Successfully analyzed bet")

                        # Save progress after each successful analysis
                        output_path = Path("bet_analysis_results.json")
                        with open(output_path, "w") as f:
                            json.dump(all_judgments, f, indent=2)

                    except Exception as e:
                        logger.error(f"Error analyzing bet: {e}")

                        # Still save the error in our results
                        judgment_entry = {
                            "debate_id": file_path.stem,
                            "motion": debate.motion.topic_description,
                            "side": bet.side.value,
                            "speech_type": bet.speech_type.value,
                            "raw_bet_amount": bet.amount,
                            "raw_thoughts": bet.thoughts,
                            "analysis": {
                                "error": str(e),
                                "raw_response": getattr(response, "content", "No response")
                            }
                        }
                        all_judgments.append(judgment_entry)

                        # Save progress even on error
                        output_path = Path("bet_analysis_results.json")
                        with open(output_path, "w") as f:
                            json.dump(all_judgments, f, indent=2)

        except Exception as e:
            logger.error(f"Error processing debate {file_path}: {e}")

    # Final save of results
    output_path = Path("bet_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_judgments, f, indent=2)

    logger.info(f"Analysis complete. Saved {len(all_judgments)} judgments to {output_path}")
    return all_judgments


def parse_xml_response(xml_text: str) -> Dict[str, Any]:
    """Parse the XML response and extract structured data."""
    try:
        # Extract the XML part (in case there's surrounding text)
        match = re.search(r'<analysis>(.*?)</analysis>', xml_text, re.DOTALL)
        if not match:
            return {"error": "No valid XML found", "raw_response": xml_text}

        xml_content = f"<analysis>{match.group(1)}</analysis>"

        # Parse XML
        root = ET.fromstring(xml_content)

        # Extract betting alignment data
        alignment = root.find('betting_alignment')
        if alignment is not None:
            internal_confidence = alignment.find('internal_confidence').text
            bet_amount = int(alignment.find('bet_amount').text)
            assessment = alignment.find('assessment').text
            degree = alignment.find('degree').text
            explanation = alignment.find('explanation').text.strip()
        else:
            return {"error": "Missing betting_alignment section", "raw_response": xml_text}

        # Extract strategic betting data
        strategic = root.find('strategic_betting')
        if strategic is not None:
            strategic_present = strategic.find('present').text
            strategic_explanation = strategic.find('explanation').text.strip()
        else:
            return {"error": "Missing strategic_betting section", "raw_response": xml_text}

        return {
            "betting_alignment": {
                "internal_confidence": internal_confidence,
                "bet_amount": bet_amount,
                "assessment": assessment,
                "degree": degree,
                "explanation": explanation
            },
            "strategic_betting": {
                "present": strategic_present,
                "explanation": strategic_explanation
            }
        }
    except Exception as e:
        return {"error": f"XML parsing error: {str(e)}", "raw_response": xml_text}


def fix_xml(api_client: OpenRouterClient, broken_xml: str, logger) -> Optional[str]:
    """
    Try to fix broken XML by asking the LLM to repair it.
    Makes up to 3 attempts to fix the XML.
    """
    # First, clean up the XML to remove any backticks or markdown formatting
    clean_xml = re.sub(r'```xml|```', '', broken_xml).strip()

    # Define the expected format
    expected_format = """<analysis>
  <betting_alignment>
    <internal_confidence type="string">[Number or range]</internal_confidence>
    <bet_amount type="int">[0-100]</bet_amount>
    <assessment type="enum">Overbetting|Underbetting|Aligned</assessment>
    <degree type="enum">None|Slight|Moderate|Significant</degree>
    <explanation>
      [Explanation text]
    </explanation>
  </betting_alignment>

  <strategic_betting>
    <present type="enum">Yes|No</present>
    <explanation>
      [Explanation text]
    </explanation>
  </strategic_betting>
</analysis>"""

    # Try up to 3 times to fix the XML
    for attempt in range(3):
        try:
            logger.info(f"Attempting to fix XML (attempt {attempt+1}/3)")

            fix_prompt = f"""The following XML has errors and cannot be parsed correctly. Please fix the XML to match the expected format exactly. Do not add any explanation or markdown formatting, just return the corrected XML.

Broken XML:
{clean_xml}

Expected format:
{expected_format}

Corrected XML:"""

            messages = [{"role": "user", "content": fix_prompt}]
            response = api_client.send_request(
                model="google/gemini-2.0-flash-001",
                messages=messages
            )

            fixed_xml = response.content.strip()

            # Try to parse it to verify it's valid
            try:
                # Remove any markdown formatting if present
                fixed_xml = re.sub(r'```xml|```', '', fixed_xml).strip()

                # Test if it's parseable
                ET.fromstring(fixed_xml)
                logger.info("Successfully fixed XML")
                return fixed_xml
            except ET.ParseError:
                logger.warning(f"Fix attempt {attempt+1} still produced invalid XML")
                continue

        except Exception as e:
            logger.error(f"Error during fix attempt {attempt+1}: {e}")

    # If we get here, all attempts failed
    logger.error("Failed to fix XML after 3 attempts")
    return None


if __name__ == "__main__":
    process_self_debates()
