# alignment_tax

This project investigates the "alignment tax" in language models, specifically focusing on how Supervised Fine-Tuning (SFT) and preference tuning (DPO) affect generative diversity.

## Hypothesis
The largest drop in output diversity (mode collapse) occurs during the SFT stage, not the DPO stage, while the SFT stage also provides the primary gain in safety compliance.

## Methodology
1. A generation script (`run_generation.py`) is used to generate responses from three models in the OLMo lineage (base, SFT, DPO).
2. Two probes are used: a diversity probe (creative prompts) and a safety probe (harmful prompts).
3. An analysis script (*to be created*) will calculate Self-BLEU for diversity and use an LLM-as-judge for safety refusal rates.

## How to Run
1. Clone the repository: `git clone <url>`
2. Set up a Python environment and install dependencies: `pip install -r requirements.txt`
3. Run the generation script: `python run_generation.py`