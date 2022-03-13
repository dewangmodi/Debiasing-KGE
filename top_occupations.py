from data_handler import *
import sys
import os

top_count = 5

if len(sys.argv)>1:
	top_count = int(sys.argv[1])

DIMS = 50
CUTOFF = 0.00002

REGION_NAMES = ["India","Russia","Arabia"]
REGION_FOLDER_NAMES = ["Indian","Russian", "Arabian"]

occupation_sets_region_wise = []

for i,region in enumerate(REGION_NAMES):
	print("Region : ",region)
	region_folder = REGION_FOLDER_NAMES[i]
	entity_path = os.path.join("knowledge_graphs", region_folder, "entities_labels.tsv")
	property_path = os.path.join("knowledge_graphs", region_folder, "properties_labels.tsv")
	triplets_path = os.path.join("knowledge_graphs", region_folder, region+"_final.tsv")

	# getting occupation sets
	triplets, entity_labels, property_labels = load_data(entity_path, property_path, triplets_path)
	prob_df = get_male_female_neutral_occupations(triplets, entity_labels, property_labels, CUTOFF, get_df=True)

	# print("Occupations DataFrame")
	# print(prob_df.head())

	male_df = prob_df[prob_df["diff"]>CUTOFF].sort_values(by=["diff"],axis=0,ascending=False)
	print("Total Male Occupations : ",male_df.shape[0])
	if male_df.shape[0]<top_count:
		print("WARNING : male occupations count < required count")
	print(male_df["diff"].head(top_count))

	female_df = prob_df[prob_df["diff"]<CUTOFF].sort_values(by=["diff"],axis=0,ascending=True)
	print("Total Female Occupations : ",female_df.shape[0])
	if female_df.shape[0]<top_count:
		print("WARNING : female occupations count < required count")
	print(female_df["diff"].head(top_count))

	occupations_set = set(list(male_df.head(top_count).index) + list(female_df.head(top_count).index))
	occupation_sets_region_wise.append(occupations_set)


regions = len(REGION_NAMES)
print("Jacquard Coefficient")
for i in range(regions):
	for j in range(i+1,regions):
		intersect = occupation_sets_region_wise[i].intersection(occupation_sets_region_wise[j])
		union = occupation_sets_region_wise[i].union(occupation_sets_region_wise[j])
		# print("intersection : ",intersect)
		# print("Union : ",union)
		print(REGION_NAMES[i],"-",REGION_NAMES[j], "%.5f"%(len(intersect)/len(union)))
