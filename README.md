## Typological Language Identification Project

This repository houses a supervised learning pipeline that classifies short text passages into a limited set of world languages. We target typological cues such as orthography, character n-gram structure, and diacritics to distinguish languages that commonly co-occur in multilingual corpora (e.g., English, French, German, Spanish, Italian).

### Project Goals
- Build a reproducible dataset pipeline around the WiLI-2018 benchmark and optionally curated subsets.
- Engineer linguistic features suitable for linear Support Vector Machines (SVM) that emphasize character composition, diacritics, and language-specific patterns.
- Train, validate, and benchmark an SVM classifier that scales to a practical subset of languages (5–20 target labels) with balanced class coverage.
- Deliver interpretable evaluation artifacts: accuracy, confusion matrix, per-language precision/recall/F1, and top-k error analyses.

### Data Source
- **WiLI-2018**: Wikipedia Language Identification dataset (235 languages; ~10k sentences per language).
- We'll script utilities to:
  - Download the official release from <https://zenodo.org/record/841984>.
  - Filter to a manageable subset of languages (configurable list in `config/languages.yaml`).
  - Split into train/validation/test folds (default 70/15/15) with stratification.
  - Persist processed splits as parquet/CSV and optionally Hugging Face datasets for reproducibility.
- Optional augmentation: include balanced synthetic corpora (news, EuroParl) if we need more robust diacritic coverage, but keep scope small for now.

### Feature Engineering
- Character n-gram counts (1–5 grams) with hashing or limited vocabulary to control dimensionality.
- TF-IDF weighting to emphasize discriminative n-grams; includes weighting for special characters (é, ü, ß, ñ, etc.).
- Script-specific tokens (Latin, Cyrillic) captured via Unicode blocks or binary flags.
- Proportions of uppercase, punctuation, whitespace, numerals to capture stylistic cues.
- Optional: subword frequency ratios, vowel/consonant clusters, stop-word presence per language.
- Feature vectorization handled via `scikit-learn` pipeline objects for consistency (`TfidfVectorizer`, `HashingVectorizer`, custom transformers).

### Modeling Approach
- **Classifier**: Linear SVM (`LinearSVC`) for strong margin-based separation; optionally experiment with `SGDClassifier` hinge loss for scalability.
- Hyperparameters tuned via cross-validation (C, class weights, n-gram ranges, max features).
- We'll benchmark against lightweight baselines:
  - Majority class predictor (sanity check).
  - Multinomial Naive Bayes on same features.
- Model artifacts saved with metadata (languages list, vectorizer settings) to support inference scripts.

### Evaluation Metrics
- Overall accuracy on held-out test split.
- Confusion matrix to visualize commonly confused language pairs.
- Macro/micro precision, recall, F1 to capture per-language quality.
- Top-k accuracy (k=3) for UX scenarios where model can propose multiple guesses.
- Error analysis notebooks highlighting misclassified samples and diacritic coverage gaps.

### Project Structure (proposed)
```
├── data/
│   ├── raw/            # WiLI downloads
│   ├── processed/      # Train/val/test splits
├── notebooks/          # Exploratory analysis & prototyping
├── src/
│   ├── config/         # Language lists, vectorizer params
│   ├── data/           # Downloading & preprocessing scripts
│   ├── features/       # Vectorizers & custom feature transformers
│   ├── models/         # Training, hyperparameter search, inference
│   └── eval/           # Metrics, plotting, reporting utilities
├── tests/              # Unit tests for data + feature code
├── README.md           # Project overview (this file)
└── requirements.txt    # Python dependencies (scikit-learn, pandas, etc.)
```

### Current Implementation Snapshot
- `src/config/languages.yaml`: declarative list of seven supported languages (code, name, family, sample text) consumed by data prep + UI.
- `src/data/wili_downloader.py`: downloads WiLI, computes archive checksums, balances language counts, and emits stratified train/val/test CSVs + metadata JSON.
- `src/features/text_vectorizer.py`: char + word TF-IDF builders; `src/features/linguistic_features.py` adds handcrafted ratios (whitespace, diacritics, vowels, etc.).
- `src/models/training_pipeline.py`: Typer CLI with optional grid search; saves sklearn Pipeline artifacts plus JSON metrics + confusion plots.
- `src/models/inference.py`: exposes ranked predictions with normalized probabilities for UI/API consumers.
- `src/eval/metrics.py` & `src/eval/reporting.py`: reusable helpers for top-k accuracy, confusion matrices, and artifact persistence.
- `ui/app.py`: Streamlit UI with model picker, metrics viewer, top-k bar charts, language metadata cards, and an inline CSV-training workflow.
- `notebooks/baseline_svm.ipynb`: quick experiment to validate the enriched TF-IDF + LinearSVC pipeline on the bundled sample data.
- `tests/`: pytest-based smoke tests for feature builders and the inference stack.

### Getting Started
1. Install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the sample model (fast, for demo purposes):
   ```
   python -m src.models.training_pipeline sample
   ```
3. Launch the UI:
   ```
   streamlit run ui/app.py
   ```
   Type text in any supported language (EN/FR/ES/DE/IT) and click **Detect Language**.
4. When ready, swap the dataset with WiLI subsets and retrain via
   ```
   python -m src.models.training_pipeline custom data/processed/wili_subset.csv --output-path artifacts/wili_language_svm.joblib --grid-search --jobs 4
   ```
- Prepare a WiLI subset with your chosen languages:
   ```
   python -m src.data.wili_downloader download
   python -m src.data.wili_downloader prepare-subset \
       --lang en --lang fr --lang es --lang de --lang it --lang pt --lang nl \
       --limit-per-language 4000 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
   ```
   The resulting CSV lives under `data/processed/wili_subset.csv` by default and plugs straight into the training CLI.
- Explore metrics interactively via `notebooks/baseline_svm.ipynb`.
- Inspect artifacts:
   - `artifacts/*.joblib` – sklearn Pipelines
   - `artifacts/*.metrics.json` – accuracy, macro averages, top-k
   - `artifacts/*.confusion.png` – confusion heatmaps
- Run tests:
   ```
   pytest
   ```

### UI Preview
- Sidebar lets you select any artifact in `artifacts/`, view metrics, or upload a CSV to trigger fresh training (with optional grid search).
- Main panel accepts free-form text, renders top-k predictions as bars/table, and surfaces metadata (language name, family, sample phrase).
- Tabs supply ready-to-use sample sentences per language for quick sanity checks.

### Next Steps
1. Add additional corpora (newswire, EuroParl) for cross-domain robustness and expand `languages.yaml` as needed.
2. Introduce alternative classifiers (multilingual transformers, fastText) for comparison against the SVM baseline.
3. Automate evaluation reports (Markdown/HTML) summarizing metrics, top errors, and indicative n-grams per language.
4. Ship a lightweight API (FastAPI or Streamlit Cloud) so others can query the model remotely.
5. Track experiment metadata (MLflow/W&B) to simplify hyperparameter sweeps and artifact versioning.

Feel free to expand feature sets (phonotactic markers, byte-pair encodings, etc.) or swap classifiers if requirements evolve. This scaffold now supports richer experimentation while keeping scope manageable.

# Typological-Language-Classification
