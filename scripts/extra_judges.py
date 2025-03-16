from config import Config
import logging
from core.models import DebateTotal

judges_list = ["google/gemini-2.0-flash-lite-001"]

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
config = Config()


def main():
    sample_debate_list = list(config.sample_debates_dir.glob("*.json"))

    for debate_path in sample_debate_list:
        logger.info(f"Processing debate: {debate_path}")
        debate = DebateTotal.load_from_json(debate_path)

        for model in judges_list:
            for i in range(3):
                logger.info(f"Processing judgment: Model={model}, Iteration={i}")
                config.judgement_processor.process_judgment(debate, model)

            logger.info(f"Completed all iterations for model {model}")

        logger.info(f"Completed processing debate: {debate_path}")


if __name__ == "__main__":
    main()
