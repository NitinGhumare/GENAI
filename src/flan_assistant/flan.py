import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FlanAssistant:
    def __init__(self, model_name="google/flan-t5-small", device=None):
        self.model_name = model_name
        # device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FlanAssistant] using model {model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _generate(self, prompt, max_new_tokens=128):
        if not prompt or not prompt.strip():
            print("[_generate] empty prompt!")
            return ""
        # tokenization
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # generation (deterministic beam search)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
            no_repeat_ngram_size=2
        )
        # debug: shapes
        try:
            tok_out = outputs[0]
            decoded = self.tokenizer.decode(tok_out, skip_special_tokens=True).strip()
        except Exception as e:
            print("[_generate] decode error:", e)
            decoded = ""
        print("[_generate] decoded output preview:", repr(decoded[:200]))
        return decoded

    def summarize(self, text):
    # few-shot + strict instruction forcing bullet lines starting with '- '
        prompt = (
        "Example:\n"
        "Text: The team will deliver module A next Friday. Weekly syncs are planned.\n"
        "Bullets:\n- Team will deliver module A next Friday.\n- Weekly syncs are planned.\n\n"
        "Now, summarize the following text into 4–6 concise bullet points. "
        "Only use facts present in the text. Do NOT add or invent details. "
        "Each bullet must start with '- ' and be on its own line.\n\n"
        f"Text:\n{text}\n\nBullets:\n- "
    )
        return self._generate(prompt, max_new_tokens=200)



    def answer_from_context(self, question, context):
        if not context or not context.strip():
            return "Context file not found or empty. Create 'context.txt' first."
        prompt = (
            "You are an assistant that answers ONLY from the context below. "
            "If the answer is not present in the context, reply exactly: Not found.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return self._generate(prompt, max_new_tokens=120)
