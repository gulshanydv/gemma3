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
    system_role: str = Form(...),  # System role provided by the user
    user_prompt: str = Form(...), # Prompt provided by the user
    text_to_summarize: str = Form(...)  # Text to summarize
):
    try:
        # Structure the messages for the model
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_prompt + "\n\n" + text_to_summarize}
        ]

        # Process the input for the model
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate the summary
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

        # Decode the generated summary
        decoded = processor.decode(generation, skip_special_tokens=True)
        
        return JSONResponse(content={"summary": decoded})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)