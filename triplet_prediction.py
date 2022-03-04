import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from constants import *
from embedding_helper import *
from data_handler import *

DIMS=50
CUTOFF=0.00002

def create_entity_index2id(entity_dict):
  entityindex2id = {}
  for key, value in entity_dict.items():
    entityindex2id[value] = key
  return entityindex2id

def get_predictions_sorted(entity_vec, entity_labels, entity_dict, relation, occupation):
    #get occupation vector
    occupation_vector = get_averaged_entity_vector(entity_vec, entity_dict, DIMS, get_entity_id(entity_labels, occupation))

    entityindex2id = create_entity_index2id(entity_dict)
    occupation_vector = occupation_vector.reshape(1, -1)
    scores = np.linalg.norm(entity_vec + relation - occupation_vector, ord=2, axis=1)

    sorted_list = dict([(entityindex2id[index], score) for index, score in zip(np.argsort(scores), np.sort(scores))])
    print(len(sorted_list.items()))
    sorted_list_minimized = []
    for ent_id in entity_labels["id"]:
      try:
        sorted_list_minimized.append((ent_id, sorted_list[ent_id]))
      except:
        pass
    return sorted_list_minimized
    

def top_x_gender(triples, entity_vec, entity_labels, relations_vec, relations_dict, entity_dict, occupation, X=10):
    #relation vector
    relation_vector = relations_vec[relations_dict[occupation_relation_id]]
    #get sorted list
    sorted_scores = get_predictions_sorted(entity_vec, entity_labels, entity_dict, relation_vector, occupation)
    #top x
    top_x_scores = dict(sorted_scores[-X:])
    #for each entity

    male_counts = 0
    female_counts = 0
    for entity_id in top_x_scores.keys():
        #get entity name
        # print(entity_labels[entity_labels["id"] == entity_id])
        entity_name = np.array(entity_labels[entity_labels["id"] == entity_id]["name"])[0]
        #check gender and add
        print(triples[(triples["head"] == entity_name)])
        male_counts += triples[(triples["head"] == entity_name) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'male'")].shape[0]
        female_counts += triples[(triples["head"] == entity_name) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'female'")].shape[0]
    #return male and female numbers
    return male_counts, female_counts

def top_x_gender_gt(triples, occupation, X=10):
    #list of people in occupation
    people_in_occupation = triples[(triples["tail"]==occupation) & (triples["relation"]==occupation_relation_name)]["head"]

    male_counts = 0
    female_counts = 0
    for person in people_in_occupation:
        male_counts += triples[(triples["head"] == person) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'male'")].shape[0]
        female_counts += triples[(triples["head"] == person) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'female'")].shape[0]

    return (male_counts * X)/(male_counts + female_counts) , (female_counts * X)/(male_counts + female_counts)

def main():
  # loading embeddings
  entity_vec, relations_vec = load_embeddings(dims=DIMS)
  entity_dict, relations_dict = load_ids()

  male_vector, female_vector = get_gender_vectors(entity_vec, entity_dict, DIMS)

  # getting occupation sets
  triplets, entity_labels, property_labels = load_data()
  male_occupations, female_occupations, neutral_occupations = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF)
  # print(male_occupations[20])
  print(top_x_gender_gt(triplets, "'lawyer'"))
  # entityindex2id = create_entity_index2id(entity_dict)
  print(top_x_gender(triplets, entity_vec, entity_labels, relations_vec, relations_dict, entity_dict, "'lawyer'"))

if __name__=='__main__':
  main()