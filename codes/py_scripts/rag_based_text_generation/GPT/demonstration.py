import sys
sys.path.insert(0, "../../")
from utility import *


CONTEXT_VOLUME = 150
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = 75
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = 0.5

CHAT_MODEL_ID = "gpt-35-turbo"
CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID
VECTOR_DB_PATH = "/data/somank/llm_data/vectorDB/disease_nodes_chromaDB_using_all_MiniLM_L6_v2_sentence_transformer_model_with_chunk_size_650"
NODE_CONTEXT_PATH = "/data/somank/llm_data/spoke_data/context_of_disease_which_has_relation_to_genes.csv"
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = "pritamdeka/S-PubMedBert-MS-MARCO"


# GPT config params
temperature = 0

SYSTEM_PROMPT = """
You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context.
"""

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)


question = input("Enter your question : ") 

input("Press enter for starting the KG-RAG process")
print("Step 1: Disease entity extraction using GPT-3.5-Turbo")
entities = disease_entity_extractor(question)
print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
print(" ")

input("Press enter for proceeding to Step 2")
print("Step 2: Match extracted Disease entity to SPOKE nodes")
print("Finding vector similarity ...")
node_hits = []
for entity in entities:
    node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
    node_hits.append(node_search_result[0][0].page_content)
print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
print(" ")

input("Press enter for proceeding to Step 3")
print("Step 3: Context extraction from SPOKE")
node_context = []
for node_name in node_hits:
    node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
print("Extracted Context is : ")
print(". ".join(node_context))




