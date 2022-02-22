# Debiasing-KGE

Download OpenKE embeddings from : http://139.129.163.161/download/wikidata

Clone this repo, and place the extracted folder `Wikidata` in the base directory (`Wikidata` has been added to `.gitignore`)

Path to the Indian dataset - `knowledge_graph/`

For first paper, Part A is in - `Bias in Data.ipynb`  [Incomplete??]

The file `data_handler.py` has helper functions related to data handling. They are - 

- `load_data()` - Returns triplets, entity_labels, property_labels

- `get_male_female_entities` - Returns male_entities, female_entities

- `get_occupation_triplets` - Returns occupation_triplets, male_occupation_triplets, female_occupation_triplets

- `get_male_female_neutral_occupations` - Returns male_occupations, female_occupations, neutral_occupations