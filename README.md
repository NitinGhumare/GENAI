# GENAI 🚀  
**Collection of Generative AI projects and experiments**  

The first project in this repository is **FLAN-T5 Summarizer & QnA Assistant**,  
a CPU-friendly CLI app that performs abstractive text summarization and context-aware question answering.  
Built using Hugging Face Transformers and PyTorch, this tool is ideal for **classroom demos**, **learning projects**, and **quick offline workflows**.  

---

## ✨ Features
- **Summarization** → Convert long text into **4–6 bullet points**.  
- **QnA Assistant** → Ask questions from a local context file (`context.txt`).  
- **Strict fallback** → Returns `"Not found"` if the answer isn’t in context.  
- **Runs fully offline** on CPU (Torch), no external APIs required.  
- Uses **prompt engineering controls**: beam search, top-p sampling, temperature, etc.  
- Built with **modular CLI code** (easy for students & demos).  

---

## 🛠 Tech Stack
- Python  
- PyTorch  
- Hugging Face Transformers  
- FLAN-T5  
- SentencePiece  

---

## ⚡ Installation
Clone the repo:
```bash
git clone https://github.com/NitinGhumare/GENAI.git
cd GENAI
```

Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Summarization
```bash
python -m src.flan_assistant.cli summarize --text "The project meeting discussed deadlines and milestones. Team agreed to deliver module A by next Friday. Risks about testing were raised and mitigations assigned. Weekly syncs will be scheduled."
```
✅ Output (example):
```
- Team will deliver module A next Friday  
- Weekly syncs are planned  
- Risks about testing were raised and mitigations assigned  
```

### 2. QnA Assistant
Edit `context.txt` and add your notes, e.g.:
```
FLAN-T5 is a language model family by Google.  
This project uses google/flan-t5-small for summarization and QnA.  
```

Run:
```bash
python -m src.flan_assistant.cli qa --question "Which model does the project use?"
```

✅ Output:
```
google/flan-t5-small
```

---

## 📂 Project Structure
```
GENAI/
│── src/
│   └── flan_assistant/
│       ├── __init__.py
│       ├── flan.py       # Core model code
│       └── cli.py        # Command-line interface
│── context.txt            # Local notes for QnA
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 📌 Next Steps
- Add more **Generative AI projects** (text-to-image, chatbots, advanced summarizers).  
- Improve evaluation with larger models (`flan-t5-base`, `flan-t5-large`).  
- Add unit tests + CI/CD pipeline.  

---

## 👤 Author
**Nitin Ghumare**  
🔗 [GitHub](https://github.com/NitinGhumare)  
