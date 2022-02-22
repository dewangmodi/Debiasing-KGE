import numpy as np
from constants import *

def load_embeddings(dims=50):
	"""
	Returns the embeddings for entities and relations as nxdims ndarrays 
	"""
	if dims==50:
		entity_vec = np.memmap(ENTITY_EMBEDDING_PATH_50 , dtype='float32', mode='r')
		entity_vec = entity_vec.reshape(int(entity_vec.shape[0]/dims), dims)

		relations_vec = np.memmap(RELATION_EMBEDDING_PATH_50 , dtype='float32', mode='r')
		relations_vec = relations_vec.reshape(int(relations_vec.shape[0]/dims), dims)

		return entity_vec, relations_vec

def load_ids():
	"""
	Returns dictionaries for entities and relations with ids as key and index in embedding ndarray as value 
	"""
	with open(ENTITY_ID_PATH,"r") as r:
		entity_data = r.readlines()[1:]
		entity_dict = {}
		for line in entity_data:
			iD, index = line.strip().split("\t")
			entity_dict[iD] = int(index)
	with open(RELATION_ID_PATH,"r") as r:
		relations_data = r.readlines()[1:]
		relations_dict = {}
		for line in relations_data:
			iD, index = line.strip().split("\t")
			relations_dict[iD] = int(index)
	return entity_dict, relations_dict


def get_averaged_entity_vector(entity_vec=None, entity_dict=None, dims=50, id_list = []):
	"""
	Returns an averaged vector for ids given in id_list
	"""
	if entity_vec is None:
		entity_vec, _ = load_embeddings(dims)
	if entity_dict is None:
		entity_dict, _ = load_ids()

	vec = np.zeros((dims,1))
	for i in id_list:
		vec += entity_vec[entity_dict[i]].reshape(-1,1)
	vec /= len(id_list)
	return vec

def get_gender_vectors(entity_vec=None, entity_dict=None, dims=50):
	"""
	Returns vectors for male and female (categories taken from constants.py)
	"""
	if entity_vec is None:
		entity_vec, _ = load_embeddings(dims)
	if entity_dict is None:
		entity_dict, _ = load_ids()

	male_vector = get_averaged_entity_vector(entity_vec, entity_dict, dims, male_category_ids)
	female_vector = get_averaged_entity_vector(entity_vec, entity_dict, dims, female_category_ids)
	return male_vector, female_vector

