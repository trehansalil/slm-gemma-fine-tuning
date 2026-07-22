---
license: other
language: en
tags: [legal, sft, closed-book-qa, llama, small-language-model]
---

# slm-125m-qa

125M legal SLM, **closed-book QA fine-tune** of
[thesreedath/slm-125m-base](https://huggingface.co/thesreedath/slm-125m-base).
Answers questions from its own weights (no context passage), and refuses when it
does not know. Trained on ~7.1k self-contained, faithfulness-judged,
decontaminated Q&A pairs (10% refusals). Loss on answer tokens only.

Chat template: `<|bos|><|system|>\n{SYSTEM}<|eos|>\n<|user|>\n{QUESTION}<|eos|>\n<|assistant|>\n`
