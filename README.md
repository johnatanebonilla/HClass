
*"Haber" Classifier: Existentiality and Normativity Analysis

This repository contains a Python script to process a corpus of YouTube transcriptions, identify and analyze instances of the verb "haber" in various contexts, and classify these instances based on specific linguistic norms. The script is designed to work with Parquet files generated using the [Y3C (YouTube Captions Corpus Creator)](https://github.com/johnatanebonilla/y3c) script.

## Requirements

- The input Parquet files must be created using the Y3C script.
- spaCy models for Spanish (es_core_news_sm, es_core_news_md, es_core_news_lg, es_dep_news_trf).

## Y3C - YouTube Captions Corpus Creator

[Y3C GitHub Repository](https://github.com/johnatanebonilla/y3c)

The Y3C project provides a Python script to create a corpus of YouTube transcriptions. The script fetches video URLs from YouTube channels specified in an Excel file, downloads the transcriptions in the desired language, restores punctuation and capitalization, cleans unnecessary spaces, and saves the results in Parquet files. It also generates an Excel file with corpus statistics.

## Installation

To install the required dependencies, run:

```bash
git clone https://github.com/johnatanebonilla/class_haber.git
cd class_haber
pip install pandas spacy fastparquet tqdm openpyxl
```

## Usage

To run the script, use the following command:

```bash
python class_haber.py --parquet path/to/input/folder --output path/to/output/folder --model [sm|md|lg|trf]
```

- `--parquet`: Path to the input folder containing Parquet files.
- `--output`: Path to the output folder where the results will be saved.
- `--model`: spaCy model size to use (sm, md, lg, or trf).

## Output Files

The script produces the following output files:

1. `corpus_haber_yt_parsed_no_ex.parquet`: Contains instances of "haber" used in non-existential contexts.
2. `corpus_haber_yt_parsed_ex.parquet`: Contains instances of "haber" used in existential contexts.
3. `haber_ex_norm_plur_yt.xlsx`: An Excel file with further classification of existential uses of "haber".

## Existential vs. Non-Existential Uses of "Haber"

The script classifies instances of "haber" into two categories:

1. **Existential Uses (ex)**: Instances where "haber" is used existentially (e.g., "hay un libro en la mesa").
2. **Non-Existential Uses (no_ex)**: Instances where "haber" is used in other contexts (e.g., as an auxiliary verb in perfect tenses).

### Further Classification of Existential Uses

The script further classifies existential uses of "haber" in the `haber_ex_norm_plur_yt.xlsx` file into the following categories:

1. **Normative Use of "Haber"**: These are instances where the use of "haber" follows the normative grammar rules of Spanish.
   - **Pluralization Norm**: According to standard Spanish grammar, "haber" in existential constructions should always be in the singular form, regardless of the plurality of the subject. For example, "hay libros en la mesa" (correct) vs. "hay libros en la mesa" (incorrect if pluralized).

2. **Non-Normative Pluralized Use of "Haber"**: These are instances where "haber" is incorrectly pluralized in existential constructions, which is non-normative in Spanish.
   - Examples include "habían libros en la mesa" instead of "había libros en la mesa".
     
## Contact

For more information, contact johnatanebonilla@gmail.com.
