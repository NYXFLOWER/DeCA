import random
import numpy as np
import torch
import data_utils
from collections import defaultdict
import pandas as pd
from evaluate import compute_acc

print()


seed = 2020
dataset = 'ml-100k'


datadir = './data/'
data_path = f"{datadir}/{dataset}"

if dataset == 'ml-100k':
	K_list = [3, 20]
else:
	K_list = [5, 20]

torch.manual_seed(seed)  # cpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms

ratings_train = pd.read_csv(f"{data_path}/{dataset}.train.rating", header=None, sep='\t').to_numpy()
ratings_valid = pd.read_csv(f"{data_path}/{dataset}.valid.rating", header=None, sep='\t').to_numpy()

item_count = defaultdict(int)
total_term = 0

for _, item, _ in ratings_train:
	item_count[item] += 1
	total_term += 1

print(f"==========================================")
print(f"Dataset, Seed: {dataset}, {seed}")

# for predicting based on the popularity
most_popular = [(item_count[x], x) for x in item_count]
most_popular.sort()
most_popular.reverse()


items_per_user = defaultdict(list)
users_per_item = defaultdict(list)
for u, b, r in ratings_train:
	items_per_user[u].append((b, r))
	users_per_item[b].append((u, r))


global_average = sum([r for (_, _, r) in ratings_train]) / len(ratings_train)
user_average = {}
for u in items_per_user:
	user_average[u] = sum([r for (b, r) in items_per_user[u]]) / len(items_per_user[u])

# 1.1 negative sampling for validation
valid_data = [(u, i) for (u, i, _) in ratings_valid]
valid_label = np.ones(len(valid_data))

all_items = list(users_per_item.keys())

# 1.2 constructing the validation set with positive and negative samples
valid_data_all = []
for u, i in valid_data:
	train_i = [i for (i, r) in items_per_user[u]]
	left_items = np.setdiff1d(all_items, train_i)
	neg_i = np.random.choice(left_items)
	valid_data_all.append((u, neg_i))
valid_data += valid_data_all
valid_label = np.concatenate([valid_label, np.zeros(len(valid_data_all))])

# # 1.3 baseline model evaluation
# return1 = set()
# count = 0
# for ic, i in most_popular:
# 	count += ic
# 	return1.add(i)
# 	if count > total_term / 2:
# 		break

# y_predictions = [i in return1 for (_, i) in valid_data]
# baseline_acc = np.sum(np.array(y_predictions).astype(float) == valid_label) / len(valid_label)
# print("Baseline Accuracy:", baseline_acc)

# 1.4 prediction model based on both Jaccard Similarity and the popularity
def compute_Jaccard(s1, s2):
	"""Compute the jaccard similarity of two sets"""
	s1 = set(s1)
	s2 = set(s2)
	numer = len(s1.intersection(s2))
	denom = len(s1.union(s2))
	if denom == 0:
		return 0
	return numer / denom


def get_max_jaccard(u, b):
	"""Get the maximum jaccard similarity between users on their item ratings"""
	items = [i for (i, r) in items_per_user[u]]
	users_b = [user for (user, r) in users_per_item[b]]

	similarities = []
	for b2 in items:
		if b2 == b:
			return 1
		users_b2 = [user for (user, r) in users_per_item[b2]]
		similarities.append(compute_Jaccard(users_b, users_b2))
	if len(similarities) == 0:
		return 0
	return max(similarities)


def predict(threshold_jaccard, threshold_popularity, data, jaccard_similarities=None):
	if jaccard_similarities is None:
		jaccard_similarities = []
		for (u, i) in data:
			jaccard_similarities.append(get_max_jaccard(u, i))
		jaccard_similarities = np.array(jaccard_similarities)	
	
	sim_threshold = np.sort(jaccard_similarities)[: int(len(jaccard_similarities) * threshold_jaccard)][-1]
	true_set = set()
	count = 0
	for ic, i in most_popular:
		count += ic
		true_set.add(i)
		if count > total_term * threshold_popularity:
			break

	predictions = []
	for idx, (_, i) in enumerate(data):
		max_sim = jaccard_similarities[idx]
		if max_sim > sim_threshold or i in true_set:
			predictions.append(1)
		else:
			predictions.append(0)
	return np.array(predictions)


# 1.5 hyperparameter tuning for finding the best thresholds for the prediction model
jaccard_similarities = []
for (u, i) in valid_data:
	jaccard_similarities.append(get_max_jaccard(u, i))
jaccard_similarities = np.array(jaccard_similarities)


threshold_jaccard_candidates = np.arange(1, 100) / 100
threshold_popularity_candidates = np.arange(1, 100) / 100

best_acc, best_threshold_jaccard, best_threshold_popularity = 0, 0, 0
for threshold1 in threshold_jaccard_candidates:
	for threshold2 in threshold_popularity_candidates:
		predictions = predict(threshold1, threshold2, valid_data, jaccard_similarities)
		acc = np.sum(np.array(predictions) == valid_label) / len(valid_label)
		if acc > best_acc:
			best_acc = acc
			best_threshold_jaccard = threshold1
			best_threshold_popularity = threshold2

print(f"The best accuracy of the baseline model is {best_acc}, with threshold_jaccard {best_threshold_jaccard} and threshold_popularity {best_threshold_popularity}")


# 1.6 prediction on the test set
def test_all_users(item_num, test_data_pos, user_pos, top_k):
	predictedIndices = []
	GroundTruth = []
	
	for u in test_data_pos:
		test_pairs = [[u, i] for i in range(item_num)]
		predictions = predict(best_threshold_jaccard, best_threshold_popularity, test_pairs) 
		
		test_data_mask = [0] * item_num
		if u in user_pos:
			for i in user_pos[u]:
				test_data_mask[i] = -9999
		predictions = torch.Tensor(predictions) + torch.Tensor(test_data_mask).float()
		_, indices = torch.topk(predictions, top_k[-1])
		indices = indices.numpy().tolist()
		predictedIndices.append(indices)
		GroundTruth.append(test_data_pos[u])
		if u % 100 == 0:
			print(u)

	precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
	return precision, recall, NDCG, MRR

_, _, test_data_pos, _, user_pos, _, item_num, _, _, _, _, _, _ = data_utils.load_all(dataset, data_path+"/")
clean_precision, clean_recall, clean_NDCG, clean_MRR = test_all_users(item_num, test_data_pos, user_pos, K_list)
print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_recall[0], clean_recall[1]))
print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_NDCG[0], clean_NDCG[1]))