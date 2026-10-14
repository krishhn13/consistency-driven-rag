# Consistency-Gated Iterative RAG

Code and results for *Consistency-Driven Iterative Retrieval-Augmented
Generation: When Does Agreement Across Evidence Subsets Signal a
Reliable Answer?* (Krishan Kumar Sah, Adish Jain; Chitkara University,
course 22CS411, 2026).

## What it tests

An iterative RAG controller that stops retrieving when the generator's
answer survives removal of each group of evidence passages. The paper's
finding is negative: on 300 HotpotQA distractor questions, agreement
under evidence removal is no better a stopping signal than ordinary
self-consistency at a matched threshold, and it is a stricter test that
fires on four times fewer answers.

## Setup

- HotpotQA distractor validation split, closed pool of 10 paragraphs per
  question; BM25 retrieval (rank-bm25 0.2.2)
- Llama 3.1 8B Instruct, Q4_K_M, served by Ollama 0.40.3 on one T4 GPU
  (Kaggle free tier)
- Seed 13; questions 0–99 for design, 100–399 for evaluation
- K=4 initial passages, M=4 evidence groups, T=3 rounds, agreement
  threshold 0.6 F1

## Results (300 evaluation questions)

| Method            | EM   | F1   | Gen. calls |
|-------------------|------|------|------------|
| Single-pass RAG   | 36.7 | 48.2 | 1.00       |
| Fixed-depth (T=3) | 47.3 | 61.2 | 3.00       |
| Self-consistency  | 46.3 | 59.2 | 6.46       |
| Proposed          | 48.0 | 61.3 | 9.09       |

F1 of the proposed method vs fixed-depth: +0.1, 95% CI [−2.8, +3.2].

## Files

- `notebook.ipynb` — the full experiment, run top to bottom on Kaggle
- `results/final_300.jsonl` — one line per (question, method) with the
  answer, scores, cost, and per-round agreement log
- `results/final_report.txt`, `results/extras_report.txt` — printed
  summaries, including the pilot study and the printed source
- `results/wrong_50.csv` — the 50 wrong answers inspected by hand

## Reproducing

Open the notebook on Kaggle with a T4 GPU, run the boxes in order. The
main run takes about 15 minutes of generation time after setup. The run
appends to `final_300.jsonl` and resumes if the session restarts.

## License

Code: MIT. HotpotQA is released under CC BY-SA 4.0; Llama 3.1 under the
Llama 3.1 Community License.
