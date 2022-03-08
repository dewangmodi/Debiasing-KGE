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
    get_id = get_entity_id(entity_labels, occupation)
    print(len(get_id))
    print(get_id)
    # exit()
    occupation_vector = get_averaged_entity_vector(entity_vec, entity_dict, DIMS, get_entity_id(entity_labels, occupation))

    entityindex2id = create_entity_index2id(entity_dict)
    occupation_vector = occupation_vector.reshape(1, -1)
    scores = -np.linalg.norm(entity_vec + relation - occupation_vector, ord=1, axis=1)

    print("scores: ", scores.shape)

    # sorted_list = dict([(entityindex2id[index], score) for index, score in zip(np.argsort(scores), np.sort(scores))])
    sorted_list = dict([(entityindex2id[index], scores[index]) for index in np.argsort(scores)])
    print("sorted_list:")
    print(len(sorted_list.items()))
    print(list(sorted_list.items())[:10])
    # exit()
    sorted_list_minimized = []
    for ent_id in entity_labels["id"]:
      try:
        sorted_list_minimized.append((ent_id, sorted_list[ent_id]))
      except:
        pass
    sorted_list_minimized = sorted(sorted_list_minimized, key=lambda t:t[1])
    print('sorted_list_minimized: ', len(sorted_list_minimized), sorted_list_minimized[-10:])
    return sorted_list_minimized
    

def top_x_gender(triples, entity_vec, entity_labels, relations_vec, relations_dict, entity_dict, occupation, X=10):
    #relation vector
    relation_vector = relations_vec[relations_dict[occupation_relation_id]]
    #get sorted list
    sorted_scores = get_predictions_sorted(entity_vec, entity_labels, entity_dict, relation_vector, occupation)
    #top x
    # top_x_scores = dict(sorted_scores[-X:])
    sorted_scores.reverse()
    top_x_scores = dict(sorted_scores)
    #for each entity

    male_counts = 0
    female_counts = 0
    for entity_id in top_x_scores.keys():
        #get entity name
        # print(entity_labels[entity_labels["id"] == entity_id])
        entity_name = np.array(entity_labels[entity_labels["id"] == entity_id]["name"])[0]
        print("entity_id:: ", entity_id)
        print("entity_name:: ", entity_name)
        #check gender and add
        print(triples[(triples["head"].str.contains(entity_name))])
        print("="*50)
        male_counts += int(triples[(triples["head"] == entity_name) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'male'")].shape[0] > 0)
        female_counts += int(triples[(triples["head"] == entity_name) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'female'")].shape[0] > 0)
        if (male_counts+female_counts)==X:
          break
    #return male and female numbers
    print("male_counts, female_counts: ", male_counts, female_counts)
    return male_counts, female_counts

def top_x_gender_gt(triples, occupation, X=10):
    #list of people in occupation
    people_in_occupation = triples[(triples["tail"]==occupation) & (triples["relation"]==occupation_relation_name)]["head"]

    male_counts = 0
    female_counts = 0
    print('people_in_occupation', people_in_occupation.shape)
    print('people_in_occupation', people_in_occupation)
    
    for person in people_in_occupation:
        male_counts += int(triples[(triples["head"] == person) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'male'")].shape[0] > 0)
        female_counts += int(triples[(triples["head"] == person) & (triples["relation"] == gender_relation_name) & (triples["tail"] == "'female'")].shape[0] > 0)

    print("male_counts, female_counts: ", male_counts, female_counts)
    est_male, est_female = (male_counts * X)/(male_counts + female_counts) , (female_counts * X)/(male_counts + female_counts)
    print("est_male, est_female: ", est_male, est_female)

    return est_male, est_female

def main():
  # loading embeddings
  entity_vec, relations_vec = load_embeddings(dims=DIMS)
  entity_dict, relations_dict = load_ids()

  print("entity_vec: ", entity_vec.shape)
  # print(entity_vec[:10])
  print("relations_vec: ", relations_vec.shape)
  # print(relations_vec[:10])

  print("entity_dict: ", len(entity_dict.items()))
  print(list(entity_dict.items())[:10])
  print("relations_dict: ", len(relations_dict.items()))
  print(list(relations_dict.items())[:10])

  male_vector, female_vector = get_gender_vectors(entity_vec, entity_dict, DIMS)

  print("male_vector: ", male_vector.shape)
  # print(male_vector)
  print("female_vector: ", female_vector.shape)
  # print(female_vector)

  # getting occupation sets
  triplets, entity_labels, property_labels = load_data()
  print("triplets: ", triplets.shape)
  print(triplets[:10])
  print("entity_labels: ", entity_labels.shape)
  print(entity_labels[:10])
  print("property_labels: ", property_labels.shape)
  print(property_labels[:10])
  print("VERIFYING:: ", property_labels[property_labels["id"]=="P106"])
  male_occupations, female_occupations, neutral_occupations = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF)

  print("male_occupations: ", male_occupations.shape)
  print(male_occupations[:10])
  print("female_occupations: ", female_occupations.shape)
  print(female_occupations[:10])
  # print(male_occupations[20])
  
  print(top_x_gender_gt(triplets, "'engineer'"))
  # entityindex2id = create_entity_index2id(entity_dict)
  print(top_x_gender(triplets, entity_vec, entity_labels, relations_vec, relations_dict, entity_dict, "'engineer'"))

if __name__=='__main__':
  main()