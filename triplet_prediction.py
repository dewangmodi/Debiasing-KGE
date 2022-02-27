import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
from constants import *
from embedding_helper import *
from data_handler import *

def create_entity_index2id(entity_dict):
  entityindex2id = {}
  for key, value in entity_dict:
    entityindex2id[value] = key
  return entityindex2id

def triple_score(e, r, o):
  return np.linalg.norm(e + r - o, ord=2)

def update_scores_dict(scores, entityindex2id, index, e, r, o):
    score = triple_score(e, r, o)
    #entity index -> entity id
    entity_id = entityindex2id[index]
    #store <score, entity id>
    scores[entity_id] = score

def get_predictions_sorted(entity_vec, entityindex2id, relation, occupation):
    #get occupation vector
    occuptation_vector = get_averaged_entity_vector(entity_vec, entity_dict, DIMS, occupation)
    manager = Manager()
    scores = manager.dict()
    with Pool(8) as p:
        p.imap_unordered(update_score, [scores, entityindex2id, i, e, relation, occupation_vector for i, e in enumerate(entity_vec)] )
    # for index, entity in enumerate(entity_vec):
    #     score = triple_score(entity, relation, occupation)
    #     #entity index -> entity id
    #     entity_id = entityindex2id[index]
    #     #store <score, entity id>
    #     score[entity_id] = score

    #sort and return
    return dict(sorted(scores.items(), key=lambda item : item[0]))

def top_x_gender(triples, entity_vec, entity_labels, relations_vec, relations_dict, entityindex2id, occupation, X=10):
    #relation vector
    relation_vector = relations_vec[relations_dict[occupation_relation_id]]
    #get sorted list
    sorted_scores = get_predictions_sorted(entity_vec, entityindex2id, relation_vector, occupation)
    #top x
    top_x_scores = dict(sorted_scores.items()[:X])
    #for each entity

    male_counts = 0
    female_counts = 0
    for entity_id in top_x_scores.values():
        #get entity name
        entity_name = entity_labels[entity_labels["id"].str == entity_id]["name"]
        #check gender and add
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