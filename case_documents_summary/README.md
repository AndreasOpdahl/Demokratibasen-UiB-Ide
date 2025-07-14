# `case_documents_summary` – mappeoversikt og arbeidsflyt

Denne mappen automatiserer hele løpet fra rå kommunale saksdokumenter til evaluerte sammendrag. Alle stier gjelder relativt til `case_documents_summary/`.

## Miljøvariabler

Legg til en .env i denne mappen med minst
- ANTHROPIC_API_KEY=...
- OPENAI_API_KEY=...
- GOOGLE_API_KEY=...
---

## Katalogstruktur
```text
case_documents_summary/
├── archive/                      # Eldre eksperimenter, kun til referanse
├── baseline/                     # Baseline-sammendrag hentet fra Demokratibasen, med tilsvarende dokumenter
├── cleaning_preprocessing/       # Rensing av råtekst, skrelling av metadata i saksdokumenter
├── data_raw/                     # Rå .txt/.pdf (ikke committed)
├── LLM_as_judge/                 # Parvis evaluering med LLM-dommere
│   ├── create_input_file.py
│   ├── judge_input/              # 6 × *.jsonl fra create_input_file.py
│   ├── openai_judge_results/     # Når dommermodellen er openai
│   ├── gemini_judge_results/     # Når dommermodellen er gemini
│   ├── claude_judge_results/     # Når dommermodellen er claude
│   ├── compute_winners.py
│   ├── winner_stats.py
│   └── build_win_matrix.py
├── summary_generation/           # Ett skript per sammendragsmodell
│   └── <modell>_generate.py
├── majority_vote.py              # Flertallsavstemning på tvers av dommere
├── majority_vote_per_document.csv
├── majority_vote_totals.csv
└── README.md                     # Denne filen
```

## Pipeline
### Rensing og sammendragsgenereing
1. Hente ut dokumenter har typen case_document for å få saksfremlegg. Hent ut første 300 som jeg jobber med videre: `cleaning_preprocessing/raw_data_preprocessing.py`
2. Skrelle dokumenter ved hjelp av regex. Fjerne metadata konservativt, heller beholde for mye enn å fjerne noe viktig: `cleaning_preprocessing/regex_document_cleaning`
3. Generere sammendrag, en gang per modell: `summary_generation/generate_summary_openai.py` / `summary_generation/generate_summary_gemini.py` / `summary_generation/generate_summary_claude.py`

### Vurdering med LLM-dommer
4. Lage inputfiler til LLM-dommere: `LLM_as_judge/create_input_file.py``
5. Kjør parvis vurdering, gjentas for hver. Eksempel med openai som dommer: `LLM_as_judge/openai_run_judge_scores_pairwise.py` gir resultatfiler i `openai_judge_results/judge_scores_<modelA>_vs_<modelB>.csv`
6. Berege vektet gjennomsnitt: `LLM_as_judge/compute_winners.py` gir resultatfiler i `openai_judge_results/judge_winners_<modelA>_vs_<modelB>.csv`
7. Teller seiere, tap og uavgjort i judge_winners resultater: `LLM_as_judge/winner_stats.py` gir resultatfiler i `openai_judge_results/judge_stats_<modelA>_vs_<modelB>.csv`
8. Kombinere seks stats-filer og lage en matriseoversikt over hvor mange prosent av dokumentene hver modell/baseline slo en annen: `LLM_as_judge/build_win_matrix.py` lagrer resultater i `openai_judge_results/model_win_matrix.csv`

### Flertallsvotering
9. `majority_vote.py` Leser alle judge_scores_* fra de tre dommermodellene og skriver `majority_vote_per_document.csv` – endelig vinner per dokument, og `majority_vote_totals.csv` – summerte seire per modell
