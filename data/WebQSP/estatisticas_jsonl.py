# import json
# from collections import Counter

# JSONL_FILE = r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\retriever_dataset_cwq.jsonl"

# total = 0

# candidate_sizes = []
# gold_sizes = []

# with open(JSONL_FILE, "r", encoding="utf8") as f:

#     for line in f:

#         item = json.loads(line)

#         total += 1

#         candidate_sizes.append(
#             len(item["candidate_relations"])
#         )

#         gold_sizes.append(
#             len(item["gold_relations"])
#         )

# # =====================================================
# # ESTATÍSTICAS GERAIS
# # =====================================================

# print(f"Total de exemplos: {total}")

# print(
#     f"Média candidate_relations: "
#     f"{sum(candidate_sizes)/len(candidate_sizes):.2f}"
# )

# print(
#     f"Média gold_relations: "
#     f"{sum(gold_sizes)/len(gold_sizes):.2f}"
# )

# print(
#     f"Máximo candidate_relations: "
#     f"{max(candidate_sizes)}"
# )

# print(
#     f"Máximo gold_relations: "
#     f"{max(gold_sizes)}"
# )

# print(
#     f"Mínimo candidate_relations: "
#     f"{min(candidate_sizes)}"
# )

# print(
#     f"Mínimo gold_relations: "
#     f"{min(gold_sizes)}"
# )

# # =====================================================
# # DISTRIBUIÇÃO DAS GOLD RELATIONS
# # =====================================================

# gold_counter = Counter(gold_sizes)

# print("\nDistribuição de gold_relations:\n")

# for tamanho, qtd in sorted(gold_counter.items()):

#     porcentagem = 100 * qtd / total

#     print(
#         f"{tamanho:2d} relações -> "
#         f"{qtd:5d} exemplos "
#         f"({porcentagem:.2f}%)"
#     )

with open(r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\retriever_train.jsonl", "w", encoding="utf8") as out:

    for arquivo in [
        r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\retriever_dataset_webqsp.jsonl",
        r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\retriever_dataset_cwq.jsonl"
    ]:

        with open(arquivo, "r", encoding="utf8") as f:

            for linha in f:

                out.write(linha)