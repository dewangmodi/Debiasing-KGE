import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from constants import *
from embedding_helper import *
from data_handler import *
from debias import *

DIMS=50
CUTOFF=0.000018

def create_entity_index2id(entity_dict):
	entityindex2id = {}
	for key, value in entity_dict.items():
		entityindex2id[value] = key
	return entityindex2id

def get_male_female_names(triples, occupation=None):
	male_names = triples[(triples["relation"] == gender_relation_name) & (triples["tail"] == "'male'")]["head"]
	female_names = triples[(triples["relation"] == gender_relation_name) & (triples["tail"] == "'female'")]["head"]

	male_names = list(set(male_names))
	female_names = list(set(female_names))

	if occupation is not None:

		occupation_subdf = triples[((triples["relation"]==occupation_relation_name) & (triples["tail"] == occupation ))]
		male_names = occupation_subdf[occupation_subdf["head"].isin(male_names)]["head"]
		female_names = occupation_subdf[occupation_subdf["head"].isin(female_names)]["head"]

	return male_names, female_names

def get_named_embeddings(entity_vec, entity_labels, entity_dict, names):
	ids = []
	embeddings = []
	names_short = []

	for name in names:
		try:
			ids.append(entity_dict[get_entity_id(entity_labels, name)[0]])
			names_short.append(name)
		except:
			pass
	ids = np.array(ids)
	embeddings = entity_vec[ids]
	return embeddings, names_short

def top_x_gender_gt(male_names, female_names, triples, occupation, X=[10]):
	print('Ground Truth')
	male_counts = 0
	female_counts = 0

	# for name in male_names:
	# 	if ((triples["head"] == name) & (triples["relation"] == occupation_relation_name) & (triples["tail"] == occupation)).any():
	# 		male_counts +=1

	occupation_subdf = triples[((triples["relation"]==occupation_relation_name) & (triples["tail"]==occupation))]

	male_counts = occupation_subdf[occupation_subdf["head"].isin(male_names)].shape[0]


	print('male done')

	# for name in female_names:
	# 	if ((triples["head"] == name) & (triples["relation"] == occupation_relation_name) & (triples["tail"] == occupation)).any():
	# 		female_counts +=1

	female_counts = occupation_subdf[occupation_subdf["head"].isin(female_names)].shape[0]


	print('female done')
	print('__________')
	for x in X:
		print('top ', x)
		est_male, est_female = (male_counts * x)/(male_counts + female_counts) , (female_counts * x)/(male_counts + female_counts)
		print("est_male, est_female: ", est_male, est_female)
	print('__________')


def top_x_gender_pred(entity_vec, entity_dict, entity_labels, relation_vector, occupation, male_embeddings, female_embeddings, X=[10]):
	print('Prediction')

	occupation_vector = entity_vec[entity_dict[get_entity_id(entity_labels, occupation)[0]]]

	male_scores = np.linalg.norm(male_embeddings + relation_vector - occupation_vector, axis=1)
	female_scores = np.linalg.norm(female_embeddings + relation_vector - occupation_vector, axis=1)

	num_male_names = male_embeddings.shape[0]
	num_female_names = female_embeddings.shape[0]

	combined_scores = np.concatenate((male_scores, female_scores), 0)
	print(combined_scores.shape)

	print(combined_scores.shape[0], num_male_names, num_female_names)

	sorted_scores = [(combined_scores[index], index) for index in np.argsort(combined_scores)]
	print('__________')
	for x in X:
		sorted_scores_x = sorted_scores[:x]
		male_counts = sum([int(sorted_scores_x[i][1] < num_male_names) for i in range(x)])
		female_counts = x - male_counts
		print('top ', x)
		print(male_counts)
		print(female_counts)
	print('___________')

def hits(entity_vec, relation_vector, occupation_vecs, occupation_names, male_embeddings, female_embeddings, occupation, direction_vector, lambd=0):
	occupation_vecs = debias(occupation_vecs, direction_vector, lambd)
	male_hits = 0
	female_hits = 0
	# Loop through male
	for i, embedding in enumerate(male_embeddings):
		if i%100==0:
			print(i)
		scores = np.linalg.norm(embedding + relation_vector - occupation_vecs, axis=1)
		sorted_scores = [(scores[index], index) for index in np.argsort(scores)]
		sorted_scores = sorted_scores[:10]
		occupations = [occupation_names[idx] for _,idx in sorted_scores]
		if occupation in occupations:
			male_hits +=1 

	for i, embedding in enumerate(female_embeddings):
		if i%100==0:
			print(i)
		scores = np.linalg.norm(embedding + relation_vector - occupation_vecs, axis=1)
		sorted_scores = [(scores[index], index) for index in np.argsort(scores)]
		sorted_scores = sorted_scores[:10]
		occupations = [occupation_names[idx] for _,idx in sorted_scores]
		if occupation in occupations:
			female_hits +=1 

	# print(male_hits, female_hits)
	return male_hits, female_hits



def main():
	# loading embeddings
	entity_vec, relations_vec = load_embeddings(dims=DIMS)
	entity_dict, relations_dict = load_ids()

	male_vector, female_vector = get_gender_vectors(entity_vec, entity_dict, DIMS)
	direction_vector = (female_vector - male_vector)/np.linalg.norm(female_vector - male_vector)

	# getting occupation sets
	triplets, entity_labels, property_labels = load_data()
	male_occupations, female_occupations, neutral_occupations = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF)
	relation_vector = relations_vec[relations_dict[occupation_relation_id]]

	# male_names, female_names = get_male_female_names(triplets)
	# top_x_gender_gt(male_names, female_names, triplets, "'lawyer'", [10, 100, 500, 1000])

	# print(len(male_names), len(female_names))
	# male_embeddings, male_names = get_named_embeddings(entity_vec, entity_labels, entity_dict, male_names)
	# print(male_embeddings.shape)
	# female_embeddings, female_names = get_named_embeddings(entity_vec, entity_labels, entity_dict, female_names)
	# print(female_embeddings.shape)

	# top_x_gender_pred(entity_vec, entity_dict, entity_labels, relation_vector, "'lawyer'", male_embeddings, female_embeddings, [10, 100, 500, 1000])


	male_hits = 0
	female_hits = 0
	num_male = 0
	num_female = 0
	lambd = 0
	occs = ["'politician'", "'cricketer'", "'film director'", "'writer'", "'author'", "'actor'", "'television actor'", "'model'", "'singer'", "'film actor'", "'dancer'"]
	for i in range(len(occs)):
		male_names, female_names = get_male_female_names(triplets, occupation = occs[i])
		print(len(male_names), len(female_names))

		#get male-female embeddings
		male_embeddings, male_names = get_named_embeddings(entity_vec, entity_labels, entity_dict, male_names)
		female_embeddings, female_names = get_named_embeddings(entity_vec, entity_labels, entity_dict, female_names)
		print(male_embeddings.shape)
		print(female_embeddings.shape)
		num_male += male_embeddings.shape[0]
		num_female += female_embeddings.shape[0]

		#get embeddings of all occupations
		# print(len(male_occupations.tolist()))
		occupations_name_list = male_occupations.tolist() + female_occupations.tolist() + neutral_occupations.tolist()
		occupation_vecs, occupation_names = get_named_embeddings(entity_vec, entity_labels, entity_dict, occupations_name_list)
		print(occupation_vecs.shape)
		mh, fh = hits(entity_vec, relation_vector, occupation_vecs, occupation_names, male_embeddings, female_embeddings, occs[i], direction_vector, lambd)
		male_hits += mh
		female_hits +=fh

	print('Male counts, Female counts: ', male_hits/num_male, female_hits/num_female)

if __name__=='__main__':
	main()