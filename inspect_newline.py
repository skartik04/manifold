import os

import dotenv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import statistics


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    greedy: bool = False,
    use_chat: bool = True
) -> str:
    """Generate chat completion from messages or raw text."""
    # Format prompt based on use_chat flag
    if use_chat:
        # Format messages into a prompt using the tokenizer's chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Use raw text directly (messages should be a string in this case)
        formatted_prompt = messages
    print(formatted_prompt)

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


# Main execution
if __name__ == "__main__":
    # Load model and tokenizer
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    model, tokenizer = load_model_and_tokenizer(model_name)

    num_chars = 600
    k = 40
    use_chat = False   
    
    # Process story data
    data = strip_newlines(story)
    data_split = make_splits(data, num_chars=num_chars)

    for i, line in enumerate(data_split):
        text, indv_lines = add_newlines(line, k=k)
        print(text)
        print('-' * 100)

        # Prepare input based on use_chat flag
        if use_chat:
            prompt = create_chat_completion_prompt(tokenizer, text)
        else:
            prompt = text

        # print(prompt)

        a = generate_chat_completion(model, tokenizer, prompt, use_chat=use_chat)

        # Count chars before each \n
        lengths = []
        for j, line in enumerate(a.split("\n")):
            if j > 1:  # Skip the first line
                lengths.append(len(line))
            print(f"{line:<{k}} | {len(line):>3}")
        
        lengths = [length for length in lengths if length != 0]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths) if len(lengths) > 1 else 0
        print(f"Mean length: {mean_length:.2f}, Std: {std_length:.2f}")
        
        print('=' * 100)
        break


