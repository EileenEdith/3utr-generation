# 3utr-generation

A collaborative project for 3′UTR dataset curation, expression modeling, and generative design exploration.

## Overview
This project aims to develop computational frameworks for analyzing and generating 3′UTR sequences associated with desirable expression properties. Our work focuses on promoter-specific GFP expression under PGK and CAG conditions and explores both discriminative and generative modeling strategies for 3′UTR design.

## Project Goals
1. **Build a 3′UTR classification model**  
   Develop classification models that predict whether a 3′UTR sequence belongs to the high- or low-expression group based on GFP expression measured under PGK and CAG promoter conditions.

2. **Fine-tune the GEMORNA 3′UTR model using high-expression sequences**  
   Use pretrained GEMORNA 3′UTR model weights and fine-tune the model with high-expression 3′UTR sequences identified from the classification task, in order to bias generation toward desirable expression-related sequence patterns.

3. **Develop a control-tag-based conditional generation framework**  
   Extend the GEMORNA-based generation framework by introducing control tags, enabling the model to generate 3′UTR sequences with desired properties under specific promoter-related conditions.

## Planned Workflow
1. Curate and merge reference datasets into a unified 3′UTR-expression table.
2. Define promoter-specific high/low labeling strategies.
3. Train and evaluate baseline classification models.
4. Select high-expression 3′UTR subsets for generative fine-tuning.
5. Fine-tune the pretrained GEMORNA 3′UTR model.
6. Design and evaluate control-tag-based conditional generation.

## Repository Structure
```text
3utr-generation/
├── README.md
├── main.py
├── environment.yml
├── .gitignore
├── config/
│   ├── data_config.yaml
│   ├── classification_config.yaml
│   └── generation_config.yaml
├── data/
│   ├── raw/
│   ├── reference/
│   └── processed/
├── src/
│   ├── data/
│   ├── classification/
│   ├── generation/
│   ├── models/
│   └── utils/
├── weights/
│   ├── pretrained/
│   └── finetuned/
├── notebooks/
├── results/
│   ├── figures/
│   ├── tables/
│   └── generated_sequences/
└── docs/
    └── data_availability.md
