import numpy as np
import pandas as pd

# gender related information
gender_relation_name = "'sex or gender'"
gender_relation_id = "P21"
male_categories = ["'male'","'male organism'"]
female_categories = ["'female'"]
transgender_categories = ["'transgender female'","'intersex'","'genderqueer'","'non-binary'", "somevalue"]


occupation_relation_name = "'occupation'"
occupation_relation_id = "P106"

def load_data():
	"""
	Return the triplets, entity and property labels after 
	dropping rows with missing values
	"""

	# load data
	entity_labels = pd.read_csv("knowledge_graph/entities_labels.tsv",delimiter='\t', header=None, names=["id","name"])
	property_labels = pd.read_csv("knowledge_graph/properties_labels.tsv",delimiter='\t',names=["id","name"])
	triplets = pd.read_csv("knowledge_graph/India_final.tsv",delimiter='\t',header=None,names=["head","relation","tail"])

	# drop nan rows
	triplets = triplets.dropna()
	entity_labels = entity_labels.dropna()
	property_labels = property_labels.dropna()

	return triplets, entity_labels, property_labels


def get_male_female_entities(triplets, entity_labels, property_labels):
	"""
	Return list of male entities and female entities
	"""
	gender_triplets = triplets[triplets["relation"]==gender_relation_name]
	male_entities = gender_triplets[gender_triplets["tail"].isin(male_categories)]["head"].values
	female_entities = gender_triplets[gender_triplets["tail"].isin(female_categories)]["head"].values
	return male_entities, female_entities


def get_occupation_triplets(triplets, entity_labels, property_labels, male_entities=None, female_entities=None):
	"""
	Return all occupation triplets, and occupation triplets of males and females
	"""
	if male_entities is None or female_entities is None:
		male_entities, female_entities = get_male_female_entities(triplets, entity_labels, property_labels)
	occupation_triplets = triplets[triplets["relation"]==occupation_relation_name]
	male_occupation_triplets = occupation_triplets[occupation_triplets["head"].isin(male_entities)]
	female_occupation_triplets = occupation_triplets[occupation_triplets["head"].isin(female_entities)]	
	return occupation_triplets, male_occupation_triplets, female_occupation_triplets

def get_male_female_neutral_occupations(triplets, entity_labels, property_labels, cutoff, male_occupation_triplets=None, female_occupation_triplets=None):
	"""
	For the given cutoff, return the list of male, female and neutral occupations
	"""

	# getting probabilities per occupation

	if male_occupation_triplets is None or female_occupation_triplets is None:
		_, male_occupation_triplets, female_occupation_triplets = get_occupation_triplets(triplets, entity_labels, property_labels)

	male_counts = male_occupation_triplets["tail"].value_counts()
	female_counts = female_occupation_triplets["tail"].value_counts()
	probs = pd.DataFrame({"male":male_counts,"female":female_counts}).fillna(0)
	probs["male"]/=male_occupation_triplets.shape[0]
	probs["female"]/=female_occupation_triplets.shape[0]

	probs["diff"] = probs["male"]-probs["female"]
	
	male_occupations = probs[probs["diff"] > cutoff].index
	female_occupations = probs[probs["diff"] < -cutoff].index
	neutral_occupations = probs[(probs["diff"]>=-cutoff) & (probs["diff"]<=cutoff)].index
	return male_occupations, female_occupations, neutral_occupations
