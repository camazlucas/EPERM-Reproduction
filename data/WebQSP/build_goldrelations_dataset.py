import json
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

# =====================================================
# CONFIGURAÇÕES
# =====================================================

INPUT_DIR = r"D:\KGQAArmazenamento\DATASET RoG\CWQ\Train"

OUTPUT_JSONL = r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\retriever_dataset_cwq.jsonl"

OUTPUT_DISCARDED = r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\ids_descartados_cwq.txt"

MAX_PATHS = 3

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def build_graph(triples):
    """
    Constrói um DiGraph a partir das triplas do subgrafo.
    """

    G = nx.DiGraph()

    for triple in triples:

        if len(triple) != 3:
            continue

        head, relation, tail = triple

        G.add_edge(
            str(head),
            str(tail),
            relation=str(relation)
        )

    return G


def extract_candidate_relations(triples):
    """
    Extrai todas as relações presentes no subgrafo.
    """

    relations = set()

    for triple in triples:

        if len(triple) != 3:
            continue

        relations.add(str(triple[1]))

    return sorted(relations)


def extract_gold_relations(
    G,
    q_entities,
    a_entities,
    max_paths=3
):
    """
    Encontra os caminhos mínimos entre
    entidades da pergunta e da resposta.
    """

    gold_relations = set()

    total_paths = 0

    for q in q_entities:

        for a in a_entities:

            q = str(q)
            a = str(a)

            if q not in G:
                continue

            if a not in G:
                continue

            try:

                shortest_paths = nx.all_shortest_paths(
                    G,
                    source=q,
                    target=a
                )

                for path in shortest_paths:

                    for i in range(len(path) - 1):

                        rel = G[path[i]][path[i + 1]]["relation"]

                        gold_relations.add(rel)

                    total_paths += 1

                    if total_paths >= max_paths:

                        return sorted(gold_relations)

            except (
                nx.NetworkXNoPath,
                nx.NodeNotFound
            ):
                pass

    return sorted(gold_relations)

# =====================================================
# ENCONTRA TODOS OS PARQUETS
# =====================================================

parquet_files = sorted(
    Path(INPUT_DIR).rglob("*.parquet")
)

print(f"Arquivos encontrados: {len(parquet_files)}")

# =====================================================
# ESTATÍSTICAS
# =====================================================

total_questions = 0
valid_questions = 0
discarded_questions = 0

candidate_sizes = []
gold_sizes = []

discarded_ids = []

# =====================================================
# PROCESSAMENTO
# =====================================================

with open(
    OUTPUT_JSONL,
    "w",
    encoding="utf8"
) as fout:

    for parquet_file in parquet_files:

        print(f"\nProcessando: {parquet_file}")

        df = pd.read_parquet(parquet_file)


        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            total_questions += 1

            try:

                question_id = str(row["id"])

                question = row["question"]

                q_entities = list(row["q_entity"])

                a_entities = list(row["a_entity"])

                graph = row["graph"]

                G = build_graph(graph)

                candidate_relations = (
                    extract_candidate_relations(
                        graph
                    )
                )

                gold_relations = (
                    extract_gold_relations(
                        G,
                        q_entities,
                        a_entities,
                        MAX_PATHS
                    )
                )

                # -------------------------
                # DESCARTA SEM CAMINHO
                # -------------------------

                if len(gold_relations) == 0:

                    discarded_questions += 1

                    discarded_ids.append(
                        question_id
                    )

                    continue

                # -------------------------
                # EXEMPLO FINAL
                # -------------------------

                example = {

                    "id":
                        question_id,

                    "question":
                        question,

                    "topic_entities":
                        q_entities,

                    "answer_entities":
                        a_entities,

                    "candidate_relations":
                        candidate_relations,

                    "gold_relations":
                        gold_relations
                }

                fout.write(
                    json.dumps(
                        example,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                valid_questions += 1

                candidate_sizes.append(
                    len(candidate_relations)
                )

                gold_sizes.append(
                    len(gold_relations)
                )

            except Exception as e:

                discarded_questions += 1

                discarded_ids.append(
                    str(row.get("id", "UNKNOWN"))
                )

                print(
                    f"\nErro em "
                    f"{row.get('id','UNKNOWN')}: "
                    f"{e}"
                )

# =====================================================
# SALVA IDS DESCARTADOS
# =====================================================

with open(
    OUTPUT_DISCARDED,
    "w",
    encoding="utf8"
) as f:

    for qid in discarded_ids:

        f.write(qid + "\n")

# =====================================================
# ESTATÍSTICAS FINAIS
# =====================================================

print("\n==============================")
print("ESTATÍSTICAS FINAIS")
print("==============================")

print(
    f"Total de perguntas: "
    f"{total_questions}"
)

print(
    f"Perguntas válidas: "
    f"{valid_questions}"
)

print(
    f"Perguntas descartadas: "
    f"{discarded_questions}"
)

if total_questions > 0:

    print(
        f"Taxa de descarte: "
        f"{100 * discarded_questions / total_questions:.2f}%"
    )

if candidate_sizes:

    print(
        f"Média candidate_relations: "
        f"{sum(candidate_sizes)/len(candidate_sizes):.2f}"
    )

if gold_sizes:

    print(
        f"Média gold_relations: "
        f"{sum(gold_sizes)/len(gold_sizes):.2f}"
    )

print("\nDataset gerado com sucesso.")