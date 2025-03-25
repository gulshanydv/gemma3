from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

device = 'cpu'

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    ckpt, device_map=device, torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(ckpt)

app = FastAPI()

@app.post("/generate_summary/")
async def generate_summary(
    system_role: str = Form(...),  
    user_prompt: str = Form(...),  
    text_to_summarize: str = Form(...), 
):
    print("Received system_role:", system_role)
    print("Received user_prompt:", user_prompt)
    print("Received text_to_summarize:", text_to_summarize)

    combined_input = f"{system_role}\n{user_prompt}\n\n{text_to_summarize}"

    print("Combined Input:", combined_input)

    inputs = processor.tokenizer(
        combined_input, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(model.device)

    # Check processed inputs type
    print("Processed Inputs type:", type(inputs))
    print(inputs)

            # Ensure inputs contain 'input_ids'
    if "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[-1]
    else:
        raise ValueError("Processed inputs do not contain 'input_ids'")


    generation = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

    # Decode the generated tokens into text
    decoded = processor.decode(generation, skip_special_tokens=True)
    # Return the generated summary
    return JSONResponse(content={"summary": decoded})

