import os

import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def add_newlines(text, k=40, insert="\n"):
    """Add insert symbol to text to keep lines under k characters."""
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

    indv_lines = [line + insert for line in result[:-1]] + [result[-1]]
    return insert.join(result), indv_lines


def load_model_and_tokenizer(model_name):
    """Load tokenizer and model."""
    print(f"Loading model: {model_name}")
    path = '/home/kartik/all_keys/.env'
    dotenv.load_dotenv(path)
    cache = "/mnt/SSD7/kartik/cache"

    # Set HuggingFace cache directory globally
    os.environ['HF_HOME'] = cache
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache
    os.environ['TRANSFORMERS_CACHE'] = cache
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    use_chat: bool = True,
    print_prompt: bool = True
) -> str:
    """Generate chat completion from messages or raw text."""
    # Format prompt based on use_chat flag
    if use_chat:
        # Format messages into a prompt using the tokenizer's chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Use raw text directly (messages should be a string in this case)
        formatted_prompt = messages
    if print_prompt:
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


def create_chat_completion_prompt(user_question: str):
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

    # Configuration
    k = 80
    use_chat = False
    insert = ">>"  # Symbol to insert as line break (try: "\n", " 🔴 ", " ||| ", " <BREAK> ")

    # Mode: "chars" = use fixed num_chars, "newlines" = compute num_chars from k * num_newlines
    mode = "newlines"  # "chars" or "newlines"
    num_chars = 600  # used when mode="chars"
    num_newlines = 12  # used when mode="newlines"

    # System prompt (same one used in create_chat_completion_prompt)
    system_prompt = 'You must continue the story with the same style and tone as the user message.\n\n'

    # Compute effective num_chars based on mode
    if mode == "newlines":
        effective_num_chars = k * num_newlines
    else:  # mode == "chars"
        effective_num_chars = num_chars

    # Process story data
    data = strip_newlines(story)
    data_split = make_splits(data, num_chars=effective_num_chars)

    print(f"Tokenization of insert symbol: {tokenizer.encode(insert, add_special_tokens=False)}")

    for i, chunk in enumerate(data_split):
        text, indv_lines = add_newlines(chunk, k=k, insert=insert)

        # Print formatted text (with newlines after insert for visibility + char/token counts)
        print("INPUT TEXT:")
        input_lines = text.split(insert)
        max_len = max(len(line) for line in input_lines) + len(insert)
        for line in input_lines:
            char_count = len(line)
            token_count = len(tokenizer.encode(line, add_special_tokens=False))
            if insert == "\n":
                print(f"{line:<{max_len}} | {char_count:>3} ({token_count})")
            else:
                display_line = line + insert
                print(f"{display_line:<{max_len}} | {char_count:>3} ({token_count})")
        print('-' * 100)

        # Prepare input based on use_chat flag
        if use_chat:
            prompt = create_chat_completion_prompt(text)
        else:
            prompt = system_prompt + text

        # Print actual prompt to model (AS-IS, exactly what's sent)
        print("ACTUAL PROMPT TO MODEL:")
        prompt_text = prompt if isinstance(prompt, str) else tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        print(prompt_text)
        print(f"Total chars: {len(prompt_text)}, Total tokens: {len(tokenizer.encode(prompt_text, add_special_tokens=False))}")
        print('-' * 100)

        a = generate_chat_completion(model, tokenizer, prompt, use_chat=use_chat, print_prompt=False)

        # Print output (AS-IS)
        print("OUTPUT (AS-IS):")
        print(a)
        print(f"Total chars: {len(a)}, Total tokens: {len(tokenizer.encode(a, add_special_tokens=False))}")
        print('-' * 100)

        # Print output (split by insert symbol with char/token counts)
        print("OUTPUT (SPLIT BY INSERT):")
        output_lines = a.split(insert)
        if len(output_lines) > 1:
            for line in output_lines:
                char_count = len(line)
                token_count = len(tokenizer.encode(line, add_special_tokens=False))
                if insert == "\n":
                    print(f"{line:<{k}} | {char_count:>3} ({token_count})")
                else:
                    display_line = line + insert
                    print(f"{display_line:<{k}} | {char_count:>3} ({token_count})")
        else:
            print("(No insert symbols found in output)")

        print('=' * 100)
        if i > 1:
            break


