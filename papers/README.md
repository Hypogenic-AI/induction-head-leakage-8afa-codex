# Downloaded Papers

1. [Deduplicating Training Data Makes Language Models Better](2107.06499_deduplicating_training_data_makes_lms_better.pdf)
   - Authors: Katherine Lee et al.
   - Year: 2021
   - arXiv: 2107.06499
   - Why relevant: establishes that duplicated text materially increases memorization and verbatim copying, which is an important confound when testing whether induction heads are a major leakage source.

2. [In-context Learning and Induction Heads](2209.11895_in_context_learning_and_induction_heads.pdf)
   - Authors: Catherine Olsson et al.
   - Year: 2022
   - arXiv: 2209.11895
   - Why relevant: foundational mechanistic paper connecting induction heads to pattern completion and in-context learning.

3. [Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy](2210.17546_preventing_verbatim_memorization_false_privacy.pdf)
   - Authors: Daphne Ippolito et al.
   - Year: 2022
   - arXiv: 2210.17546
   - Why relevant: shows exact-copy prevention is not enough to stop leakage, which matters if induction heads are only part of the mechanism.

4. [Bag of Tricks for Training Data Extraction from Language Models](2302.04460_bag_of_tricks_training_data_extraction.pdf)
   - Authors: Weichen Yu et al.
   - Year: 2023
   - arXiv: 2302.04460
   - Why relevant: strong extraction baseline and attack methodology for evaluating leakage.

5. [Training Data Extraction From Pre-trained Language Models: A Survey](2305.16157_training_data_extraction_survey.pdf)
   - Authors: Shotaro Ishihara
   - Year: 2023
   - arXiv: 2305.16157
   - Why relevant: consolidates attack/defense terminology and evaluation choices for memorization studies.

6. [Identifying Semantic Induction Heads to Understand In-Context Learning](2402.13055_identifying_semantic_induction_heads.pdf)
   - Authors: Jie Ren et al.
   - Year: 2024
   - arXiv: 2402.13055
   - Why relevant: extends induction-head analysis from literal copying to syntactic and knowledge-relation retrieval.

7. [Forcing Diffuse Distributions out of Language Models](2404.10859_forcing_diffuse_distributions.pdf)
   - Authors: Yiming Zhang et al.
   - Year: 2024
   - arXiv: 2404.10859
   - Why relevant: directly addresses the “models struggle to produce random outputs” side of the hypothesis.

8. [Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning](2407.07011_induction_heads_pattern_matching_icl.pdf)
   - Authors: Joy Crosbie and Ekaterina Shutova
   - Year: 2024
   - arXiv: 2407.07011
   - Why relevant: causal ablations in Llama-3-8B and InternLM2-20B show induction heads matter for few-shot pattern use.

9. [Demystifying Verbatim Memorization in Large Language Models](2407.17817_demystifying_verbatim_memorization.pdf)
   - Authors: Jing Huang et al.
   - Year: 2024
   - arXiv: 2407.17817
   - Why relevant: controlled memorization study showing leakage is intertwined with broader LM capabilities, not obviously one circuit.

10. [Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models](2505.13514_induction_head_toxicity_repetition_curse.pdf)
   - Authors: Shuxun Wang et al.
   - Year: 2025
   - arXiv: 2505.13514
   - Why relevant: closest direct evidence for the hypothesis; links induction-head dominance to runaway repetition.
   - Note: the current arXiv version is withdrawn and has no PDF; this workspace stores the available `v1` PDF.
