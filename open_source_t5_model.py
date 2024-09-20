from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

def open_source_modal(inp):
# Define the input text and prepend a prompt
    input_text = (inp)

    # Prepend prompt
    prompt = "summarize: "
    # input_text_with_prompt = prompt + input_text
    input_text_with_prompt = inp

    # Tokenize the input text
    input_ids = tokenizer(input_text_with_prompt, return_tensors="pt").input_ids

    # Generate the summary
    outputs = model.generate(input_ids, max_length=250, min_length=100, length_penalty=2.0, num_beams=4)

    # Decode and print the summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
#print(summary)
