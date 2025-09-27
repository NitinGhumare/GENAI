# src/flan_assistant/cli.py
import argparse
from .flan import FlanAssistant
from pathlib import Path
import sys

def _ensure_text_from_stdin():
    """Read multi-line text from stdin until an empty line (for interactive use)."""
    print("Paste text (end with an empty line):")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()

def _postprocess_bullets(output: str, max_bullets: int = 6) -> str:
    """
    If model didn't return bullet-style output, convert simple sentence output
    into a bullet list heuristically. Avoid adding '- ' twice.
    """
    if not output:
        return output
    out = output.strip()
    # If already a bullet list, normalize and return
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if all(ln.startswith(("-", "*")) for ln in lines):
        # normalize each line to "- <text>" (single leading hyphen)
        norm = []
        for ln in lines[:max_bullets]:
            # remove leading bullets and whitespace
            text = ln.lstrip("-* ").strip()
            norm.append(f"- {text}")
        return "\n".join(norm)

    # Otherwise split into sentences and prefix with "- "
    parts = []
    for part in out.replace("\n", " ").split(". "):
        p = part.strip()
        if not p:
            continue
        # remove any leading bullet markers if model accidentally included them
        p = p.lstrip("-* ").rstrip(".").strip()
        parts.append(p)
        if len(parts) >= max_bullets:
            break

    if not parts:
        return out
    return "\n".join(f"- {p}" for p in parts[:max_bullets])


def run_summarize(args):
    fa = FlanAssistant(model_name=args.model, device=args.device)
    if args.text:
        text = args.text
    else:
        text = _ensure_text_from_stdin()
    if not text:
        print("No text provided. Use --text or paste text into stdin.")
        return
    out = fa.summarize(text)
    out = _postprocess_bullets(out, max_bullets=args.max_bullets)
    print("\nSummary:\n")
    print(out)

def run_qa(args):
    fa = FlanAssistant(model_name=args.model, device=args.device)
    ctx_path = Path(args.context_file)
    if not ctx_path.exists():
        print(f"Context file '{args.context_file}' not found. Create it and try again.")
        return
    context = ctx_path.read_text(encoding="utf-8")
    question = args.question or input("Question: ").strip()
    if not question:
        print("No question provided.")
        return
    out = fa.answer_from_context(question, context)
    print("\nAnswer:\n")
    print(out)

def main(argv=None):
    parser = argparse.ArgumentParser(prog="flan-assistant", description="FLAN-T5 Summarizer & QnA CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_sum = sub.add_parser("summarize", help="Summarize text into bullet points")
    p_sum.add_argument("--text", type=str, help="Text to summarize (if omitted, reads stdin)")
    p_sum.add_argument("--model", default="google/flan-t5-small", help="Model name or path")
    p_sum.add_argument("--device", default=None, help="Device (e.g., cpu or cuda). Auto detected if omitted")
    p_sum.add_argument("--max-bullets", type=int, default=6, help="Max bullets to output (cli postprocess)")

    p_qa = sub.add_parser("qa", help="Answer question using local context file")
    p_qa.add_argument("--question", type=str, help="Question to ask (if omitted, will prompt)")
    p_qa.add_argument("--context-file", default="context.txt", help="Path to local context file")
    p_qa.add_argument("--model", default="google/flan-t5-small", help="Model name or path")
    p_qa.add_argument("--device", default=None, help="Device (e.g., cpu or cuda). Auto detected if omitted")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "summarize":
            run_summarize(args)
        elif args.cmd == "qa":
            run_qa(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

if __name__ == "__main__":
    main()
