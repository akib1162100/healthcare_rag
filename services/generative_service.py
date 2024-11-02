from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from langchain.llms import HuggingFacePipeline
import torch


class GenerationService:
    def __init__(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Hugging Face pipeline for generation
        hf_pipe = hf_pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_length=512
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    def generate_answer(self, retrieved_docs, question):
        context = "\n".join(retrieved_docs)

        prompt = f"Answer the following question based on these documents:\n\n{context}\n\nQuestion: {question}\nAnswer:"

        answer = self.llm(prompt)
        return answer
