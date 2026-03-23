# Data Availability

## Overview
This project uses a curated dataset constructed by integrating multiple public 3′UTR-related reference sources and external sequence annotation resources. The final dataset is intended for downstream classification and generation tasks involving 3′UTR sequence analysis and expression-associated properties.

## Data Sources
The integrated dataset was constructed using the following reference studies:

1. **Mixed tailing by TENT4A and TENT4B shields mRNA from rapid deadenylation**
2. **Massively parallel screen uncovers many rare 3′UTR variants regulating mRNA abundance of cancer driver genes**
3. **The quantitative impact of 3′UTRs on gene expression**

## Sequence Collection
Expression-related reference tables were merged primarily at the **gene ID** level.  
For 3′UTR sequence assignment, transcript-level sequence information was retrieved from **BioMart**. When multiple transcript IDs were available for the same gene, the **longest transcript-associated 3′UTR sequence** was selected. This choice was made under the assumption that the longest transcript is likely to retain the richest sequence information for downstream modeling.

## Data Integration Strategy
- Reference datasets from multiple studies were collected and organized in `data/reference/`.
- Key expression-related measurements and gene-level identifiers were extracted from each source.
- Datasets were merged using **gene ID** as the primary integration key.
- Transcript-level 3′UTR sequences were assigned using BioMart-derived transcript annotations.
- For genes with multiple transcript candidates, the **longest available 3′UTR sequence** was selected.

## Variables
The integrated dataset contains multiple feature columns, including gene- and transcript-level identifiers, sequence information, and promoter-specific expression-related measurements.

Representative columns include:
- `ensembl_gene_id`
- `gene_symbol`
- `ensembl_transcript_id`
- `utr3`
- `utr_len`
- `PGK`
- `CAG`
- `PGK_pro`
- `CAG_pro`

The full dataset contains **19,266 rows and 126 columns** after preprocessing and filtering.

## Filtering and Exclusion Criteria
To construct the final modeling-ready dataset, entries with **zero expression values under promoter-specific measurements** were excluded. After this filtering step, the final curated dataset size was:

- **19,266 rows × 126 columns**

## Classification Dataset
For the classification task, the model is designed to use:
- **3′UTR sequence**
- **PGK expression value**

These variables are used to define downstream classification settings such as high/low expression grouping.

## File Organization
- `data/reference/`  
  Stores source-specific reference files collected from published studies.
- `data/processed/`  
  Stores merged, cleaned, and modeling-ready datasets generated after integration and filtering.

## Notes
This repository may not redistribute all raw source files if redistribution is restricted by the original source. In such cases, only processed datasets, metadata, or derived tables permitted for sharing should be included.

Further updates to integration rules, filtering criteria, and derived dataset versions should be documented as the project evolves.