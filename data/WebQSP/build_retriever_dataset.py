import pandas as pd
import pprint

df = pd.read_parquet(
    r"D:\KGQAArmazenamento\DATASET RoG\WEBQSP\Train\train-00000-of-00002-d810a36ed97bc2cc.parquet"
)

sample = df.iloc[0]

answer_entities = set(sample["a_entity"])

answer_triples = []

for triple in sample["graph"]:

    h, r, t = triple

    if h in answer_entities or t in answer_entities:

        answer_triples.append(triple)

print("QUESTION:")
print(sample["question"])

print("\nANSWER TRIPLES:")
pprint.pp(answer_triples)