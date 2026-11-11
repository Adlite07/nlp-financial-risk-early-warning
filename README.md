# An NLP Early-Warning System for Financial Risk

Comparing classical (TF-IDF + logistic regression) and transformer (FinBERT) models on financial news, plus year-over-year change detection in 10-K risk factors.

**Status:** Stage 1 and Stage 2 completed. Stage 3 (retrieval-augmented question answering) and the Streamlit demo were **not completed** and are listed as future work.

**Headline metric:** recall on the negative class (the share of truly negative sentences the model catches). Missing bad news costs far more than a false alarm, so accuracy alone is never reported.

## Stage 1: News sentiment

**Data:** Financial PhraseBank, 50%-agreement version (4,846 sentences). Eight duplicated sentences were found. Six had consistent labels (one copy kept) and two had conflicting labels (both copies removed), leaving 4,836 sentences. These were split 70/15/15 into train/validation/test (3,385 / 725 / 726) with stratified sampling and seed 42. The test set was used once, after all settings were frozen. No sentence appears in more than one split (checked programmatically).

**Models:**
- Majority-class baseline (always "neutral").
- TF-IDF (unigrams) + logistic regression, C=10, balanced class weights, chosen on validation macro-F1.
- FinBERT (ProsusAI/finbert) fine-tuned for 3 epochs (learning rate 2e-5, batch size 16, max length 128, mixed precision, class-weighted loss). The epoch with the best validation macro-F1 was kept.
- Original FinBERT without our fine-tuning, as a leakage check.

**Validation results (used for model selection):**

| Model | Accuracy | Negative recall | Macro-F1 |
|---|---|---|---|
| Majority baseline | 0.594 | 0.000 | 0.249 |
| TF-IDF + LogReg | 0.760 | 0.589 | 0.701 |
| FinBERT fine-tuned | not recorded | 0.933 | 0.868 |

**Test results and cross-dataset generalisation:**

| Model | Neg. recall, PhraseBank test (95% bootstrap CI) | Neg. recall, tweets (95% CI) | Macro-F1, PhraseBank / tweets |
|---|---|---|---|
| TF-IDF + LogReg | 0.736 (0.636 to 0.826) | 0.219 (0.177 to 0.260) | 0.736 / 0.469 |
| FinBERT fine-tuned | 0.978 (0.943 to 1.000) | 0.697 (0.652 to 0.747) | 0.871 / 0.671 |
| FinBERT original | 0.967 (0.926 to 1.000) | 0.761 (0.720 to 0.805) | 0.894 / 0.668 |

**Second dataset:** Twitter Financial News Sentiment, validation split, 2,388 tweets as loaded (347 bearish), mapped bearish to negative, bullish to positive. No training or tuning was done on it.

**Findings:**
- FinBERT caught far more negative sentences than TF-IDF on both datasets.
- Fine-tuning on PhraseBank gave no clear gain over the original FinBERT, so the two are treated as comparable.
- Every model got worse on tweets. The drop between datasets was larger than the difference between models, so scores on one source should not be trusted on another without re-checking.
- On PhraseBank, fine-tuned FinBERT missed 2 of 91 negatives, and TF-IDF missed 24. TF-IDF failed mostly on numeric comparisons and implicit bad news with no negative keyword.

## Stage 2: Risk factors in 10-K filings

**Data:** EDGAR-CORPUS, Item 1A (Risk Factors), annual reports 2018 to 2020, for three companies (identified from their SEC CIK numbers):
- First Solar, Inc. (CIK 1274494), solar manufacturing
- Gladstone Investment Corporation (CIK 1321741), business development company
- MFA Financial, Inc. (CIK 1055160), mortgage REIT

**Method:** Item 1A text was split into paragraphs of at least 200 characters (891 paragraphs across company-years in the comparison). Each paragraph was embedded with all-MiniLM-L6-v2. For each paragraph, the maximum cosine similarity to any paragraph in the previous year was computed. Paragraphs were labelled **new** (similarity below 0.60), **changed** (0.60 to 0.85) or **unchanged** (0.85 or above).

**Results:**
- Most text is reused: median similarity 1.00, mean 0.947.
- 112 of 891 paragraphs were flagged as new or changed (26 new, 86 changed): 33 in 2019 and 79 in 2020.
- Flags rose in 2020 for all three companies, and included a COVID-19 paragraph with no match in the prior year (similarity 0.42).
- Flagged paragraphs were assigned to one of seven risk types by zero-shot classification (facebook/bart-large-mnli): market 45, legal 25, operational 20, credit 10, regulatory 9, cybersecurity 2, climate 1. Low-confidence labels were often doubtful in the examples read.

**Not completed:** manual validation of 30 flagged paragraphs (so flag precision is unmeasured), and validation of risk categories on a 100-paragraph hand-labelled sample.

## Not completed (future work)
- Stage 3: RAG question answering (chunking strategies, FAISS or Chroma, cited answers) evaluated on FinanceBench against the same model without retrieval.
- Streamlit demo.
- Scaling Stage 2 to 5 to 10 companies across sectors, and validating thresholds and categories by hand.

## Data and licences

| Dataset | Source | Licence | Accessed |
|---|---|---|---|
| Financial PhraseBank (Malo et al., 2014) | `takala/financial_phrasebank` on Hugging Face | CC BY-NC-SA 3.0 (non-commercial, share-alike; authors ask to be contacted for commercial use) | 11 Nov 2026 |
| Twitter Financial News Sentiment (attribute to Ricky Costa) | `zeroshot/twitter-financial-news-sentiment` on Hugging Face | MIT, per the dataset card | 11 Nov 2026 |
| EDGAR-CORPUS (Loukas et al., 2021) | `eloukas/edgar-corpus` on Hugging Face | Apache 2.0, per the dataset card | 11 Nov 2026 |

Data and trained models are not stored in this repository. The PhraseBank loading script on Hugging Face no longer works with current library versions, so the data were read directly from the original zip file in the dataset repository.

**Models used:** ProsusAI/finbert, sentence-transformers/all-MiniLM-L6-v2, facebook/bart-large-mnli.

## Reproducibility
Random seed 42 everywhere. Library versions are logged in `results/environment.json`. Developed on Google Colab (free tier, NVIDIA T4 GPU), with data and results stored on Google Drive.

## Limitations
- FinBERT's original training likely overlaps Financial PhraseBank, so its in-domain scores are probably optimistic. The cross-dataset test is the fairer comparison.
- The tweet dataset differs in both style and label meaning (bearish is not identical to negative news), so the drop mixes distribution shift and label mismatch.
- PhraseBank labels are sentiment from an investor's viewpoint, not risk labels, and 50%-agreement labels contain noise. The test set has only 91 negative sentences, so confidence intervals are wide.
- Change detection flags additions only (removed risks are not detected), is sensitive to minor rewording, and paragraph splitting is imperfect.
- Risk categories use one label per paragraph, are not validated against a hand-labelled sample, and "market risk" is probably over-assigned.
- Only three companies were analysed.
- Results show **association** with negative language and disclosure change. They are **not predictions** of stock prices or risk events.
