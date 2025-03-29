import ollama
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple
import logging
import sys
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ollama_queries.log'),
        logging.StreamHandler(sys.stdout)]
)

# Configuration
MODELS = ["mistral", "llama3.2", "phi", "gemma3", "aya", "qwen", "deepseek-r1"]
MAX_WORKERS = 5
CSV_HEADER = ["question_id", "category", "model", "prompt_en",
              "response_en", "prompt_gu", "response_gu", "opinion", 
              "validation_status"]

def warm_up_model(model: str, max_retries: int = 3, timeout: int = 300) -> bool:
    """Warm up a model with a simple query to ensure it's loaded."""
    logging.info(f"Warming up model: {model}")
    test_prompt = "Hi"
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = ollama.generate(
                model=model, 
                prompt=test_prompt,
                options={"num_predict": 1}  # Minimize response length for warm-up
            )
            if response and "response" in response:
                elapsed = time.time() - start_time
                logging.info(f"Model {model} warmed up successfully in {elapsed:.2f} seconds")
                return True
        except Exception as e:
            logging.warning(f"Warm-up attempt {attempt + 1} failed for {model}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    logging.error(f"Failed to warm up model {model} after {max_retries} attempts")
    return False

def load_questions(file_path: str) -> List[Dict]:
    """Load questions from JSON file with validation."""
    try:
        with open(file_path) as f:
            data = json.load(f)
            
        questions = []
        seen_ids = set()
        
        for category, cat_questions in data["categorized_questions"].items():
            for q in cat_questions:
                if q["id"] in seen_ids:
                    logging.warning(f"Duplicate ID: {q['id']}")
                    continue
                if not all(k in q for k in ["id", "text_en", "text_gu", "opinion"]):
                    logging.error(f"Invalid question: {q}")
                    continue
                
                questions.append({
                    "id": q["id"],
                    "category": category,
                    "text_en": q["text_en"],
                    "text_gu": q["text_gu"],
                    "opinion": q["opinion"]
                })
                seen_ids.add(q["id"])
                
        return sorted(questions, key=lambda x: x["id"])
    
    except Exception as e:
        logging.error(f"Failed to load questions: {str(e)}")
        sys.exit(1)

def validate_response(response: str) -> bool:
    """Check if response meets quality standards."""
    return response and len(response.strip()) > 25 and any(c.isalpha() for c in response)

def query_model(model: str, prompt: str) -> str:
    """Get response from LLM with basic error handling."""
    try:
        result = ollama.generate(model=model, prompt=prompt)
        return result.get("response", "")[:1000]  # Limit response length
    except Exception as e:
        logging.warning(f"{model} query failed: {str(e)}")
        return ""

def process_questions(questions: List[Dict], output_file: str):
    """Process all questions with proper dependency handling."""
    # Load existing progress
    completed: Set[Tuple[int, str]] = set()
    try:
        with open(output_file) as f:
            reader = csv.DictReader(f)
            completed = {(int(row["question_id"]), row["model"]) for row in reader}
    except FileNotFoundError:
        pass

    with open(output_file, "a", newline="") as f, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(CSV_HEADER)

        # Submit all language queries first
        futures = []
        for q in questions:
            for model in MODELS:
                if (q["id"], model) in completed:
                    continue
                
                # Capture prompts and submit both language queries
                text_en = q["text_en"]
                text_gu = q["text_gu"]
                futures.append((
                    executor.submit(query_model, model, text_en),
                    executor.submit(query_model, model, text_gu),
                    q["id"], q["category"], model, q["opinion"],
                    text_en, text_gu  # Store prompts with future
                ))

        # Process results and submit opinion queries
        opinion_futures = []
        for future_tuple in futures:
            en_future, gu_future, q_id, category, model, opinion_prompt, text_en, text_gu = future_tuple
            try:
                en_response = en_future.result()
                gu_response = gu_future.result()
                
                # Submit opinion query after languages complete
                opinion_future = executor.submit(query_model, model, opinion_prompt)
                opinion_futures.append((
                    opinion_future, q_id, category, model,
                    en_response, gu_response, text_en, text_gu
                ))
            except Exception as e:
                logging.error(f"Failed {q_id}-{model}: {str(e)}")

        # Write results to CSV
        for future in tqdm(as_completed([f[0] for f in opinion_futures]), 
                           total=len(opinion_futures),
                           desc="Finalizing"):
            # Find matching future data
            for f in opinion_futures:
                if f[0] == future:
                    data = f
                    break
            
            opinion_future, q_id, category, model, en_resp, gu_resp, text_en, text_gu = data
            opinion = opinion_future.result()
            
            # Validate responses
            en_valid = validate_response(en_resp)
            gu_valid = validate_response(gu_resp)
            
            # Write full record with prompts and responses
            writer.writerow([
                q_id,
                category,
                model,
                text_en,  # Original English prompt
                en_resp if en_valid else "",  # English response
                text_gu,  # Original Gujarati prompt
                gu_resp if gu_valid else "",  # Gujarati response
                opinion.strip(),
                "both_valid" if en_valid and gu_valid else
                "partial_valid" if en_valid or gu_valid else
                "invalid"
            ])

if __name__ == "__main__":
    try:
        questions = load_questions("questions.json")  # Update filename
        # Warm up models and filter out unsuccessful ones
        successful_models = []
        for model in MODELS:
            if warm_up_model(model):
                successful_models.append(model)
        MODELS[:] = successful_models  # Update models list
        
        if not MODELS:
            logging.error("No models available after warm-up. Exiting.")
            sys.exit(1)
            
        process_questions(questions, "llm_responses.csv")  # Update filename
        logging.info("Processing completed successfully!")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        sys.exit(1)