from data_handler import *
import sys
import os

top_count = 5

if len(sys.argv)>1:
	top_count = int(sys.argv[1])


REGION_NAMES = ["India","Russia","Arabia"]
REGION_FOLDER_NAMES = ["Indian","Russian", "Arabian"]
PROTECTED = [["'Hinduism'", "'Islam'"], ["'Eastern Orthodoxy'", "'Islam'"], ["'Catholic Church'", "'Islam'"]]
CUTOFFS = [[0.002, 0.002],[0.002, 0.002],[0.002, 0.002]]


occupation_sets_region_wise = []

for i,region in enumerate(REGION_NAMES):
	print("Region : ",region)
	region_folder = REGION_FOLDER_NAMES[i]
	entity_path = os.path.join("knowledge_graphs", region_folder, "entities_labels.tsv")
	property_path = os.path.join("knowledge_graphs", region_folder, "properties_labels.tsv")
	triplets_path = os.path.join("knowledge_graphs", region_folder, region+"_final.tsv")

	# getting occupation sets
	triplets, entity_labels, property_labels = load_data(entity_path, property_path, triplets_path)

	for gno in range(len(PROTECTED[i])):

		print("Religion : ", PROTECTED[i][gno])
		protected_categories = [PROTECTED[i][gno]]
		CUTOFF = CUTOFFS[i][gno]
		protected_ids = []
		for cat in protected_categories:
			protected_ids += get_entity_id(entity_labels, cat)
		unprotected_categories, unprotected_ids = get_remaining_categories(triplets, entity_labels, female_categories=protected_categories)
		prob_df = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF, get_df=True, male_categories=unprotected_categories, female_categories=protected_categories)


		male_df = prob_df[prob_df["diff"]>CUTOFF].sort_values(by=["diff"],axis=0,ascending=False)
		print("Total Unprotected Occupations : ",male_df.shape[0])
		if male_df.shape[0]<top_count:
			print("WARNING : male occupations count < required count")
		print(male_df["diff"].head(top_count))

		female_df = prob_df[prob_df["diff"]<CUTOFF].sort_values(by=["diff"],axis=0,ascending=True)
		print("Total Protected Occupations : ",female_df.shape[0])
		if female_df.shape[0]<top_count:
			print("WARNING : female occupations count < required count")
		print(female_df["diff"].head(top_count))

		occupations_set = set(list(male_df.head(top_count).index) + list(female_df.head(top_count).index))
		occupation_sets_region_wise.append(occupations_set)


# regions = len(REGION_NAMES)
# print("Jacquard Coefficient")
# for i in range(regions):
# 	for j in range(i+1,regions):
# 		intersect = occupation_sets_region_wise[i].intersection(occupation_sets_region_wise[j])
# 		union = occupation_sets_region_wise[i].union(occupation_sets_region_wise[j])
# 		# print("intersection : ",intersect)
# 		# print("Union : ",union)
# 		print(REGION_NAMES[i],"-",REGION_NAMES[j], "%.5f"%(len(intersect)/len(union)))
