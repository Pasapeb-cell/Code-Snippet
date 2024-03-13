from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLM:
    def __init__(self, modelID = "bigscience/bloom-1b7", max_length=300, top_k=1, temperature=0.9, repetition_penalty = 2.0):
        self.modelID = modelID
        self.max_length = max_length
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.model = AutoModelForCausalLM.from_pretrained(self.modelID, use_cache=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelID)

    def generateResponse(self,prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(0)
        sample = self.model.generate(**input_ids, max_length=300,
                        top_k=1, temperature=0.9,
                        repetition_penalty = 2.0)
        generated_story = self.tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        return generated_story