To set up and use Gemma3 (a model variant from Google) in your environment,  follow these general steps:
Prerequisites:
Ensure that you have the following dependencies and versions installed in your environment. You can either manually install the dependencies or use the requirements.txt file you've provided.
    • Python version: >=3.8 
    • Pip version: >=20.3 
    • CUDA for GPU acceleration (optional, if using a GPU) 
Step 1: Install the required dependencies
You have already listed some dependencies, so you can install them via pip. You can either use the requirements.txt file you mentioned or install them manually.
Create a virtual environment (optional but recommended):
python3 -m venv gemma_env
source gemma_env/bin/activate 
Install dependencies:
To install the dependencies from the requirements.txt file, run:
pip install -r requirements.txt
Or you can manually install the required libraries:
pip install torch transformers pillow fastapi
Step 2: Setup the FastAPI application
Model and Processor
You are loading the Gemma3ForConditionalGeneration model and processor from Hugging Face using the AutoProcessor API.
ckpt = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    ckpt, device_map=device, torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(ckpt)
This loads the model weights from the Hugging Face Hub (google/gemma-3-4b-it). Make sure you have access to the Hugging Face model and the transformers library is updated to the latest version that supports Gemma3.
Step 3: API Endpoint for Image-Text Pair Generation
The FastAPI route /generate_response/ processes text and image input and sends them to the Gemma3 model for generation. You already have the logic set up correctly to handle the POST request, receive the image and text, and generate a response.
Step 4: Running the FastAPI server
You can run the FastAPI app by using uvicorn. Run the following command:
uvicorn main:app --reload
Make sure to replace main with the name of the Python script if it's different. This will start a development server that you can access at http://127.0.0.1:8000 by default.
Step 5: Test the API
Once the FastAPI app is running, you can test the /generate_response/ endpoint with the POST method. You can use tools like Postman or curl to test the API.
For example, with curl:
curl -X 'POST' \
  'http://127.0.0.1:8000/generate_response/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'text="Your sample text"' \
  -F 'image=@path_to_your_image_file.jpg'
Step 6: Error Handling and Debugging
If you run into any issues with loading the model or processing the data, ensure the following:
    • The model checkpoint "google/gemma-3-4b-it" is accessible (this can require authentication on Hugging Face if it's a private model). 
    • Check the device_map and torch_dtype arguments, ensuring that they are compatible with your setup. 
    • Make sure you have the necessary libraries and dependencies installed. 

       Out of memory errors: The Gemma-3 4B model is quite large. If you're using a GPU, ensure you have sufficient VRAM (at least 24GB for a 4B model). 



# For summary


CURL Request: 
curl -X 'POST' \
  'http://localhost:8000/generate_summary/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'system_role=You are an assistant that summarizes text.' \
  -F 'user_prompt=Please summarize the following text.' \
  -F "text_to_summarize=In a world filled with vast technological advancements, human creativity remains a driving force for innovation. From the simplest tools t



Original Text to Summarize:
In a world filled with vast technological advancements, human creativity remains a driving force for innovation. From the simplest tools to the most complex artificial intelligence systems, humanity has always sought to improve its way of life through ingenuity. The development of the internet, for example, has revolutionized communication, education, and commerce. Likewise, breakthroughs in healthcare have saved millions of lives and improved the quality of life for people worldwide. As we continue to push the boundaries of possibility, it's essential to recognize that while technology shapes our future, it is human creativity and collaboration that will lead us to a brighter tomorrow.

Examples
1. System Role: A Scientific Research Assistant
User Prompt: Summarize the following text in a concise and technical manner, highlighting the key technological advances mentioned.
Generated Summary:
{
"summary": " \n\nThe recent advancements in quantum computing are particularly noteworthy. Quantum computers leverage the principles of quantum mechanics, such as superposition and entanglement, to perform calculations far beyond the capabilities of classical computers. This has the potential to revolutionize fields like drug discovery, materials science, and cryptography. Furthermore, the development of advanced robotics, particularly those incorporating artificial intelligence, is transforming industries across the board. These robots are becoming increasingly autonomous, capable of learning and adapting to new situations, and performing complex tasks with greater precision and"
}
