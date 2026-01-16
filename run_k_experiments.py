import os
import json
from pathlib import Path

import dotenv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# Load story data
with open("data/story.txt", "r", encoding="utf-8") as f:
    story = f.read()


def strip_newlines(data):
    """Remove all newline characters from text."""
    return data.replace('\n', ' ')


def make_splits(data, num_chars=600):
    """Split text into chunks of approximately num_chars, respecting sentence boundaries."""
    result = []
    sentences = data.split('.')
    
    current_chunk_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():  # Skip empty sentences
            continue
        
        # Try adding this sentence to current chunk
        test_chunk = current_chunk_sentences + [sentence]
        test_string = '.'.join(test_chunk) + '.'
        
        # If we have sentences and adding this one would exceed the limit
        if current_chunk_sentences and len(test_string) > num_chars:
            # Compare distances: is it closer with or without this sentence?
            current_string = '.'.join(current_chunk_sentences) + '.'
            distance_without = abs(num_chars - len(current_string))
            distance_with = abs(num_chars - len(test_string))
            
            if distance_without <= distance_with:
                # Closer without - save current chunk and start new one
                result.append(current_string)
                current_chunk_sentences = [sentence]
            else:
                # Closer with - include this sentence and start fresh
                result.append(test_string)
                current_chunk_sentences = []
        else:
            # Add to current chunk
            current_chunk_sentences.append(sentence)
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        result.append('.'.join(current_chunk_sentences) + '.')
    
    return result


def add_newlines(text, k=40):
    """Add newlines to text to keep lines under k characters."""
    result = []
    current_line = ""
    words = text.split()

    for word in words:
        # Check if adding this word would exceed k characters
        if len(current_line) + len(word) + (1 if current_line else 0) >= k:
            # Add current line to result and start new line
            if current_line:
                result.append(current_line)
                current_line = word
            else:
                # Word itself is longer than k, add it anyway
                result.append(word)
        else:
            # Add word to current line
            if current_line:
                current_line += " " + word
            else:
                current_line = word

    # Add the last line if it exists
    if current_line:
        result.append(current_line)

    indv_lines = [line + '\n' for line in result[:-1]] + [result[-1]]
    return '\n'.join(result), indv_lines


def load_model_and_tokenizer(model_name):
    """Load tokenizer and model."""
    print(f"Loading model: {model_name}")
    path = '/home/kartik/all_keys/.env'
    dotenv.load_dotenv(path)
    cache = "/mnt/SSD4/kartik/hf_cache"

    # Set HuggingFace cache directory globally
    os.environ['HF_HOME'] = cache
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache
    os.environ['TRANSFORMERS_CACHE'] = cache
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=cache,
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate_chat_completion(
    model,
    tokenizer,
    messages,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False
) -> str:
    """Generate chat completion from messages."""
    # Format messages into a prompt using the tokenizer's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    if not greedy:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Extract only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return completion


def create_chat_completion_prompt(tokenizer, user_question: str):
    """Create a chat completion prompt with system and user messages."""
    combined_user_message = f"""{user_question}"""
    system_prompt = 'You must continue the story with the same style and tone as the user message.'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_user_message}
    ]

    return messages


def run_experiment_for_k(model, tokenizer, data_split, k, output_dir, greedy=False):
    """Run experiment for a specific k value."""
    means = []
    stds = []
    outputs = []
    all_lens = []
    
    # Create output file for this k
    output_file = output_dir / f"output_k{k}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(data_split, desc=f"k={k}", leave=False)):
            text, indv_lines = add_newlines(line, k=k)
            
            # Write input text
            f.write(f"=== Chunk {i+1} ===\n")
            f.write(f"Input text (k={k}):\n")
            f.write(text + "\n")
            f.write('-' * 100 + "\n\n")
            
            # Generate completion
            mess = create_chat_completion_prompt(tokenizer, text)
            completion = generate_chat_completion(model, tokenizer, mess, greedy=greedy)
            
            # Store output
            outputs.append({
                'chunk_id': i,
                'input_text': text,
                'completion': completion
            })
            
            # Analyze completion
            f.write(f"Generated completion:\n")
            lengths = []
            for j, line_text in enumerate(completion.split("\n")):
                if j > 1:  # Skip the first two lines
                    lengths.append(len(line_text))
                f.write(f"{line_text:<{k}} | {len(line_text):>3}\n")
            
            # Calculate statistics
            lengths = [length for length in lengths if length != 0]
            all_lens.extend(lengths)
            if lengths:
                mean_length = np.mean(lengths)
                std_length = np.std(lengths) if len(lengths) > 1 else 0
                means.append(mean_length)
                stds.append(std_length)
                
                f.write(f"\nMean length: {mean_length:.2f}, Std: {std_length:.2f}\n")
            else:
                f.write("\nNo valid lines to measure.\n")
            
            f.write('=' * 100 + "\n\n")
    
    return means, stds, outputs, all_lens


def main():
    """Main execution function."""
    # Configuration
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    model_name = 'Qwen/Qwen2.5-7B-Instruct'
    num_chars = 600
    k_values = range(30, 130, 10)  # 30, 40, 50, ..., 120
    greedy = True
    
    # Create output directory structure based on model name
    model_folder = model_name.replace('/', '-')  # Sanitize model name for filesystem
    if greedy:
        output_base = Path(model_folder) / "outputs_greedy"
    else:
        output_base = Path(model_folder) / "outputs"
    output_base.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("Starting K-Value Experiments")
    print("=" * 80)
    print(f"num_chars: {num_chars}")
    print(f"k_values: {list(k_values)}")
    print(f"greedy: {greedy}")
    print(f"Output directory: {output_base}")
    print("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Process story data once (same chunks for all k values)
    data = strip_newlines(story)
    data_split = make_splits(data, num_chars=num_chars)
    print(f"\nTotal chunks created: {len(data_split)}")
    print("=" * 80)
    
    # Separate dictionaries for outputs and statistics
    outputs_dict = {}
    stats_dict = {}
    
    # Run experiments for each k value
    for k in tqdm(k_values, desc="Overall Progress"):
        # Create subfolder for this k
        k_output_dir = output_base / f"k_{k}"
        k_output_dir.mkdir(exist_ok=True)
        
        # Run experiment
        means, stds, outputs, all_lens = run_experiment_for_k(model, tokenizer, data_split, k, k_output_dir, greedy=greedy)
        
        # Store outputs separately
        outputs_dict[str(k)] = outputs
        
        # Store statistics separately
        stats_dict[str(k)] = {
            'means': means,
            'stds': stds,
            'all_lens': all_lens
        }
        
        print(f"\nk={k} completed: {len(means)} chunks processed")
        print(f"  Average mean: {np.mean(means):.2f}")
        print(f"  Average std: {np.mean(stds):.2f}")
        print(f"  Total lengths collected: {len(all_lens)}")
    
    # Save outputs dictionary
    outputs_file = output_base / "outputs.json"
    with open(outputs_file, 'w') as f:
        json.dump(outputs_dict, f, indent=2)
    
    # Save statistics dictionary
    stats_file = output_base / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"All experiments completed!")
    print(f"Outputs saved to: {outputs_file}")
    print(f"Statistics saved to: {stats_file}")
    print("=" * 80)
    
    # Print summary
    print("\nSummary:")
    for k in k_values:
        means = stats_dict[str(k)]['means']
        stds = stats_dict[str(k)]['stds']
        all_lens = stats_dict[str(k)]['all_lens']
        print(f"k={k:3d}: Avg Mean={np.mean(means):6.2f}, Avg Std={np.mean(stds):5.2f}, Chunks={len(means)}, Total Lengths={len(all_lens)}")


if __name__ == "__main__":
    main()

