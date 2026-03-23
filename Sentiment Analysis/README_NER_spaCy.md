# Named Entity Recognition (NER) Using spaCy

A Python project that performs **Named Entity Recognition (NER)** using the `spaCy` NLP library. The project extracts and categorizes named entities from text, visualizes them, adds custom entity types, and counts entity distributions.

## Overview

Named Entity Recognition (NER) is an NLP technique that identifies and classifies key information (entities) in text into predefined categories such as person names, organizations, locations, dates, and monetary values. This project uses spaCy's pre-trained English model (`en_core_web_sm`) to perform NER on product and business-related sentences.

## Features

- Extract named entities from multiple sentences
- Visualize entities with color-coded labels using `displacy`
- Add custom entity types using `EntityRuler`
- Count and analyze entity type distribution using `Counter`

## Requirements

Install the required libraries:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Project Structure

```
Named Entity Recognition/
│
├── ner_spacy.py      # Main script
└── README.md         # Project documentation
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Install and import spaCy, load `en_core_web_sm` model |
| 2 | Define sample text dataset |
| 3 | Perform NER — extract and print entities from each sentence |
| 4 | Visualize entities using `displacy` |
| 5 | Add custom entity types using `EntityRuler` |
| 6 | Count entity type occurrences using `Counter` |

## Usage

Run the main script:

```bash
python ner_spacy.py
```

## spaCy Entity Labels

| Label | Meaning | Example |
|-------|---------|---------|
| `PERSON` | People and names | Dr. Sarah Blake |
| `ORG` | Organizations, companies | MetaBrains |
| `GPE` | Geopolitical entity (cities, countries) | Berlin, Germany |
| `DATE` | Dates and time expressions | 2025, last month |
| `MONEY` | Monetary values | $50 million |
| `LOC` | Non-GPE locations, landmarks | The Great Wall of China |

## Custom Entity Types

Custom entities can be added using `EntityRuler`:

```python
patterns = [
    {"label": "PROJECT", "pattern": "MetaBrains AI Project"},
    {"label": "INVESTOR", "pattern": "Quantum Ventures"}
]
```

| Label | Meaning |
|-------|---------|
| `PROJECT` | Custom project names |
| `INVESTOR` | Custom investor/venture names |

## Example Output

**NER Extraction:**
```
Text: MetaBrains is expanding its operations to Berlin, Germany, in 2025.
Entities:
  - MetaBrains (ORG)
  - Berlin (GPE)
  - Germany (GPE)
  - 2025 (DATE)
```

**Custom Entity Recognition:**
```
- Quantum Ventures (INVESTOR)
- MetaBrains AI Project (PROJECT)
- last month (DATE)
```

**Entity Counts:**
```
Entity Counts: Counter({'GPE': 3, 'ORG': 2, 'DATE': 2, 'PERSON': 1, 'MONEY': 1})
```

## Visualization

The project uses `displacy` to render color-coded entity highlights directly in Jupyter Notebook:

```python
displacy.render(doc, style="ent", jupyter=True)
```

For running outside Jupyter:

```python
displacy.serve(doc, style="ent")
# Open http://localhost:5000 in your browser
```

## Technologies Used

| Library | Purpose |
|---------|---------|
| `spacy` | NLP pipeline and NER |
| `spacy.displacy` | Entity visualization |
| `spacy.pipeline.EntityRuler` | Custom entity patterns |
| `collections.Counter` | Entity frequency counting |

## Sample Dataset

```python
text_data = [
    "MetaBrains is expanding its operations to Berlin, Germany, in 2025.",
    "Dr. Sarah Blake, the CTO of MetaBrains, announced the new AI project in New York.",
    "The Great Wall of China is one of the most famous landmarks in the world.",
    "MetaBrains secured a $50 million investment from Quantum Ventures last month.",
]
```

## Limitations

- The `en_core_web_sm` model is lightweight and may miss some entities
- For higher accuracy, use `en_core_web_lg` (large model)
- Custom `EntityRuler` patterns are exact string matches only
- Does not handle abbreviations or misspellings automatically

## Upgrading to a Larger Model

For better accuracy on complex text:

```bash
python -m spacy download en_core_web_lg
```

Then replace in code:

```python
nlp = spacy.load("en_core_web_lg")
```

## License

This project is open-source and available for educational use.
