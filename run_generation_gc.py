# run generation for google cloud storage

import os
import json
import torch
import dataclasses
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
from google.cloud import storage

# --- Configuration Block ---

@dataclasses.dataclass
class GenerationConfig:
    # Model and environment settings
    models_to_test: list[str] = dataclasses.field(default_factory=lambda: [
        "allenai/OLMo-2-1124-7B",
        "allenai/OLMo-2-1124-7B-SFT",
        "allenai/OLMo-2-1124-7B-DPO",
    ])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    # Dataset and sampling settings
    num_prompts_per_probe: int = 100
    random_seed: int = 42

    # Generation parameters
    num_responses_per_prompt: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Output and logging settings
    output_file: str = f"generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_interval: int = 10
    gcs_bucket: str = "aligntment_tax_generations"  # Your actual bucket name


# --- Utility Functions ---

def setup_logging():
    """Sets up basic logging."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('generation.log'),
            logging.StreamHandler()
        ]
    )


def upload_to_gcs(local_path: str, bucket_name: str, model_name: str, max_retries: int = 3):
    """Uploads a local file to Google Cloud Storage with retry logic."""
    for attempt in range(max_retries):
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            safe_model_name = model_name.replace("/", "_")
            blob_name = f"{safe_model_name}/{os.path.basename(local_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            logging.info(f"✅ Uploaded {local_path} to gs://{bucket_name}/{blob_name}")
            return True
        except Exception as e:
            logging.error(f"❌ Upload attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(5)  # Wait 5 seconds before retry
            else:
                logging.error(f"❌ Failed to upload {local_path} after {max_retries} attempts")
                return False


def parse_hh_prompt(example):
    """Extracts the initial human prompt from the Anthropic HH-RLHF dataset."""
    text = example['chosen']
    human_turn_start = text.find("\n\nHuman:")
    if human_turn_start == -1:
        return None
    assistant_turn_start = text.find("\n\nAssistant:", human_turn_start)
    if assistant_turn_start == -1:
        prompt = text[human_turn_start + len("\n\nHuman:"):].strip()
    else:
        prompt = text[human_turn_start + len("\n\nHuman:"):assistant_turn_start].strip()
    return {"prompt": prompt}


def prepare_datasets(config: GenerationConfig):
    """Loads, samples, and prepares the evaluation datasets."""
    logging.info("Preparing datasets...")

    # 1. Diversity Probe
    diversity_dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    diversity_prompts = diversity_dataset.shuffle(seed=config.random_seed).select(range(config.num_prompts_per_probe))

    # 2. Safety Probe
    safety_dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    safety_dataset_parsed = safety_dataset.map(parse_hh_prompt, num_proc=4)
    safety_dataset_filtered = safety_dataset_parsed.filter(lambda x: x['prompt'] is not None)
    safety_prompts = safety_dataset_filtered.shuffle(seed=config.random_seed).select(range(config.num_prompts_per_probe))

    probes = {
        "diversity": [item['prompt'] for item in diversity_prompts],
        "safety": [item['prompt'] for item in safety_prompts]
    }

    logging.info(f"Prepared {len(probes['diversity'])} diversity prompts and {len(probes['safety'])} safety prompts.")
    return probes


# --- Core Generation Logic ---

def generate_responses(model, tokenizer, prompt, config: GenerationConfig):
    """Generates N responses for a single prompt using the specified parameters."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(config.device)
    responses = []
    for _ in range(config.num_responses_per_prompt):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response_only = full_output[len(prompt):].strip()
        responses.append(response_only)
    return responses


# --- Main Execution Block ---

def main():
    setup_logging()
    config = GenerationConfig()
    logging.info(f"Starting generation with config: {config}")
    logging.info(f"Device: {config.device}")
    logging.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    probes = prepare_datasets(config)

    all_results = []
    if os.path.exists(config.output_file):
        logging.info(f"Resuming from existing file: {config.output_file}")
        with open(config.output_file, 'r') as f:
            all_results = json.load(f)

    for model_name in config.models_to_test:
        logging.info(f"--- Loading model: {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config.torch_dtype,
            device_map="auto"  # Fixed: was config.device
        )
        model.eval()

        for probe_name, prompts in probes.items():
            logging.info(f"--- Starting probe: {probe_name} for model {model_name} ---")

            for i, prompt in enumerate(tqdm(prompts, desc=f"Generating for {probe_name}")):
                if any(d['model_name'] == model_name and d['probe_name'] == probe_name and d['prompt'] == prompt for d in all_results):
                    logging.info(f"Skipping already generated prompt {i+1}/{len(prompts)}")
                    continue

                try:
                    responses = generate_responses(model, tokenizer, prompt, config)
                    all_results.append({
                        "model_name": model_name,
                        "probe_name": probe_name,
                        "prompt": prompt,
                        "responses": responses
                    })
                except Exception as e:
                    logging.error(f"Error generating for prompt: '{prompt[:50]}...'. Error: {e}")
                    all_results.append({
                        "model_name": model_name,
                        "probe_name": probe_name,
                        "prompt": prompt,
                        "responses": ["ERROR: Generation failed."],
                        "error": str(e)
                    })

                # Periodically save and upload
                if (i + 1) % config.save_interval == 0:
                    logging.info(f"Saving intermediate results to {config.output_file}...")
                    with open(config.output_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    upload_to_gcs(config.output_file, config.gcs_bucket, model_name)

        # Save after each model completes
        logging.info(f"Saving results after completing model: {model_name}")
        with open(config.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        upload_to_gcs(config.output_file, config.gcs_bucket, model_name)

        # Clean up before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logging.info(f"--- Unloaded model: {model_name} ---")

    # Final save and upload
    logging.info(f"Generation complete. Saving final results to {config.output_file}.")
    with open(config.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    upload_to_gcs(config.output_file, config.gcs_bucket, "final_results")  # Fixed: generic name
    logging.info("🎉 All done!")


if __name__ == "__main__":
    main()