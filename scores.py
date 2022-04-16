import numpy as np
from data_handler import *
from embedding_helper import *
from debias import *
import os

def projection_score(occupation_vectors, direction_vector):
    return np.mean(np.dot(occupation_vectors, direction_vector))

def cosine_sim(v1, v2):
    v1_norm = v1/(np.linalg.norm(v1, axis=1).reshape(-1, 1))
    return np.dot(v1_norm, v2)

def g_score(direction_vector, occupation_vectors):
    return np.mean(cosine_sim(direction_vector, occupation_vectors))

def compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, occupation_list=[], lambd=0):
    occupation_vecs = []
    #Compute occupation ids and create occupation vector array
    for occupation in occupation_list:
        occupation_id = get_entity_id(entity_labels, occupation)
        try:
            occupation_vec = entity_vec[entity_dict[occupation_id[0]]]
            occupation_vecs.append(occupation_vec)
        except:
            pass

    occupation_vecs = np.array(occupation_vecs)

    #Debias
    debiased_vecs = debias(occupation_vecs, direction_vector, lambd)

    #Call projection score
    return projection_score(debiased_vecs, direction_vector)

def compute_g_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, occupation_list=[], lambd=0):
    occupation_vecs = []
    #Compute occupation ids and create occupation vector array
    for occupation in occupation_list:
        occupation_id = get_entity_id(entity_labels, occupation)
        try:
            occupation_vec = entity_vec[entity_dict[occupation_id[0]]]
            occupation_vecs.append(occupation_vec)
        except:
            pass

    occupation_vecs = np.array(occupation_vecs)

    #Debias
    debiased_vecs = debias(occupation_vecs, direction_vector, lambd)

    #Call projection score
    return g_score(debiased_vecs, direction_vector)

def main():
    DIMS = 50
    CUTOFF = 0.00002
    CALC_REMAINING = True
    REGION = "India"
    REGION_FOLDER = "Indian"

    entity_path = os.path.join("knowledge_graphs", REGION_FOLDER, "entities_labels.tsv")
    property_path = os.path.join("knowledge_graphs", REGION_FOLDER, "properties_labels.tsv")
    triplets_path = os.path.join("knowledge_graphs", REGION_FOLDER, REGION+"_final.tsv")

    # getting occupation sets
    triplets, entity_labels, property_labels = load_data(entity_path, property_path, triplets_path)


    if CALC_REMAINING:
        male_categories, male_category_ids = get_remaining_categories(triplets, entity_labels)

    male_occupations, female_occupations, neutral_occupations = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF, male_categories=male_categories, female_categories=female_categories)


    # loading embeddings
    entity_vec, relations_vec = load_embeddings(dims=DIMS)
    entity_dict, relations_dict = load_ids()

    male_vector, female_vector = get_gender_vectors(entity_vec, entity_dict, DIMS, male_category_ids=male_category_ids, female_category_ids=female_category_ids)


    direction_vector = (female_vector - male_vector)/np.linalg.norm(female_vector - male_vector)
    print("P scores")
    print("lambda 0")
    print(compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, male_occupations, 0))
    print(compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, female_occupations, 0))

    print("lambda 0.5")
    print(compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, male_occupations, 0.5))
    print(compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, female_occupations, 0.5))

    print("lambda 1")
    print(compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, male_occupations, 1))
    print(compute_projection_score_entire(entity_vec, entity_labels, entity_dict, direction_vector, female_occupations, 1))

    print("G scores")
    print("lambda 0")
    print("male")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, male_occupations, 0))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, male_occupations, 0))

    print("female")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, female_occupations, 0))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, female_occupations, 0))

    print("neutral")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, neutral_occupations, 0))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, neutral_occupations, 0))

    print("lambda 0.5")
    print("male")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, male_occupations, 0.5))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, male_occupations, 0.5))

    print("female")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, female_occupations, 0.5))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, female_occupations, 0.5))

    print("neutral")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, neutral_occupations, 0.5))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, neutral_occupations, 0.5))

    print("lambda 1")
    print("male")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, male_occupations, 1))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, male_occupations, 1))

    print("female")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, female_occupations, 1))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, female_occupations, 1))

    print("neutral")
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, male_vector, neutral_occupations, 1))
    print(compute_g_score_entire(entity_vec, entity_labels, entity_dict, female_vector, neutral_occupations, 1))

if __name__ == '__main__':
    main()
