from embedding_helper import *
from data_handler import *

DIMS = 50
CUTOFF = 0.00002

# loading embeddings
entity_vec, relations_vec = load_embeddings(dims=DIMS)
entity_dict, relations_dict = load_ids()

male_vector, female_vector = get_gender_vectors(entity_vec, entity_dict, DIMS)

# getting occupation sets
triplets, entity_labels, property_labels = load_data()
male_occupations, female_occupations, neutral_occupations = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF)

def get_occupation_projection_score(occupation_list, lamb=0):
    # here we skip the occupations whose vector is not available
    gender_vector = (female_vector - male_vector)/np.linalg.norm(female_vector - male_vector)
    projection_score = 0
    count = 0
    for i in occupation_list:
        occupation_id = get_entity_id(entity_labels, i)
        occupation_id_with_vector = []
        for occ_id in occupation_id:
            if occ_id in entity_dict:
                occupation_id_with_vector.append(occ_id)
        if len(occupation_id_with_vector)==0:
            continue
        occupation_vector = get_averaged_entity_vector(entity_vec, entity_dict,DIMS,occupation_id_with_vector)
        projection = np.dot(gender_vector.transpose(), occupation_vector)
        debiased_occupation_vector = occupation_vector - lamb * projection * gender_vector
        debiased_projection = np.dot(gender_vector.transpose(), debiased_occupation_vector)
        projection_score += debiased_projection
        count += 1
    return projection_score/count

# Table III
print("Projection Scores")
print("Lambda\tMale\t\tFemale\t\tNeutral")
for lamb in [0,0.5,1]:
	male_occupation_score = get_occupation_projection_score(male_occupations, lamb)
	female_occupation_score = get_occupation_projection_score(female_occupations, lamb)
	neutral_occupation_score = get_occupation_projection_score(neutral_occupations, lamb)
	print("%.1f\t%f\t%f\t%f"%(lamb,male_occupation_score, female_occupation_score, neutral_occupation_score))
