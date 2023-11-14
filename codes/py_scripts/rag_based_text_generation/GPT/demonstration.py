import sys
sys.path.insert(0, "../../")
from utility import *


response_type = sys.argv[1]

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
TEMPERATURE = 0

RAG_SYSTEM_PROMPT = """
You are an expert biomedical researcher. For answering the Question at the end, you need to first read the Context provided. Then give your final answer by considering the context.
"""
PROMPT_SYSTEM_PROMPT = """
You are an expert biomedical researcher. Answer the Question at the end
"""

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)


question = input("Enter your question : ") 
print(" ")

def main():
    if response_type == "rag":
        rag_demo(question)
    else:
        prompt_demo(question)
        
        
def prompt_demo(question):
    print("Calling GPT ...")
    output = get_GPT_response(question, PROMPT_SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE)
    print(output)
    print("")


def rag_demo(question):
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor(question)
    max_number_of_high_similarity_context_per_node = int(CONTEXT_VOLUME/len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")

    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")

    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    question_embedding = embedding_function_for_context_retrieval.embed_query(question)
    node_context_extracted = ""
    for node_name in node_hits:
        node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        node_context_list = node_context.split(". ")        
        node_context_embeddings = embedding_function_for_context_retrieval.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD)
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        node_context_extracted += ". ".join(high_similarity_context)
        node_context_extracted += ". "
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")

    input("Press enter for Step 5 - LLM prompting")
    enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
    print("Here is the enriched input to the LLM :")
    print(enriched_prompt)
    print(" ")

    input("Press enter for Step 6 - LLM response")
    print("Calling GPT ...")
    output = get_GPT_response(enriched_prompt, RAG_SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE)
    print(output)
    print(" ")



if __name__ == "__main__":
    main()
