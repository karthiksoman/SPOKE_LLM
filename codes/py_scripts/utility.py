import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import openai
import os
import time
from dotenv import load_dotenv, find_dotenv
import torch
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
# from auto_gptq import exllama_set_max_input_length





# Config openai library
config_file = os.path.join(os.path.expanduser('~'), '.gpt_config.env')
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = "azure"
openai.api_key = api_key
openai.api_base = resource_endpoint
openai.api_version = api_version

torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

BASE_URI = "https://spoke.rbvi.ucsf.edu"

def get_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)
    
@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def get_context_using_api(node_value):
    type_end_point = "/api/v1/types"
    result = get_api_resp(BASE_URI, type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': 3,
        'cutoff_Protein_source': ['SwissProt'],
        'cutoff_DaG_diseases_sources': ['knowledge', 'experiments'],
        'cutoff_DaG_textmining': 3,
        'cutoff_CtD_phase': 3,
        'cutoff_PiP_confidence': 0.7,
        'cutoff_ACTeG_level': ['Low', 'Medium', 'High']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_api_resp(BASE_URI, nbr_end_point, params=api_params)
    node_context = result.json()
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    provenance = "SPOKE-KG"                                    
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
    merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is from " + merge_2.provenance + "."
    context = merge_2['context'].str.cat(sep=' ')
    return context




def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                             revision=branch_name,
                                             cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name,                                             
                                        device_map='auto',
                                        torch_dtype=torch.float16,
                                        revision=branch_name,
                                        cache_dir=cache_dir
                                        )
    pipe = pipeline("text-generation",
                model = model,
                tokenizer = tokenizer,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                max_new_tokens = max_new_tokens,
                do_sample = True
                )    
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":temperature, "top_p":top_p})
    return llm



def create_mcq(df, source_column, target_column, node_type, predicate):
    disease_pairs = df[source_column].unique()
    disease_pairs = [(disease1, disease2) for disease1 in disease_pairs for disease2 in disease_pairs if disease1 != disease2]

    new_data = []

    #For each source pair, find a common target and 4 negative samples
    for disease1, disease2 in disease_pairs:
        common_gene = set(df[df[source_column] == disease1][target_column]).intersection(set(df[df[source_column] == disease2][target_column]))
        common_gene = list(common_gene)[0] if common_gene else None
        # Get 4 random negative samples
        negative_samples = df[(df[source_column] != disease1) & (df[source_column] != disease2)][target_column].sample(4).tolist()
        new_data.append(((disease1, disease2), common_gene, negative_samples))

    new_df = pd.DataFrame(new_data, columns=["disease_pair", "correct_node", "negative_samples"])
    new_df.dropna(subset = ["correct_node"], inplace=True)
    new_df.loc[:, "disease_1"] = new_df["disease_pair"].apply(lambda x: x[0])
    new_df.loc[:, "disease_2"] = new_df["disease_pair"].apply(lambda x: x[1])
    new_df.negative_samples = new_df.negative_samples.apply(lambda x:", ".join(x[0:4]))
    new_df.loc[:, "text"] = "Out of the given list, which " + node_type + " " + predicate + " " + new_df.disease_1 + " and " + new_df.disease_2 + ". Given list is: " + new_df.correct_node + ", " + new_df.negative_samples
    return new_df



def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    try:
        response = openai.ChatCompletion.create(
            temperature=temperature, 
            deployment_id=chat_deployment_id,
            model=chat_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ]
        )
        if 'choices' in response \
        and isinstance(response['choices'], list) \
        and len(response) >= 0 \
        and 'message' in response['choices'][0] \
        and 'content' in response['choices'][0]['message']:
            return response['choices'][0]['message']['content']
        else:
            return 'Unexpected response'
    except:
        return None


def disease_entity_extractor(text):
    chat_deployment_id = 'gpt-35-turbo'
    chat_model_id = 'gpt-35-turbo'
    temperature = 0
    system_prompt = """
    You are an expert disease entity extractor from a sentence and report it as JSON in the following format:
    Diseases : <List of extracted entities>
    Note that, only report Diseases. Do not report any other entities like Genes, Proteins, Enzymes etc.
    """
    resp = get_GPT_response(text, system_prompt, chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, api=True):
    entities = disease_entity_extractor(question)
    node_hits = []
    if entities:
        max_number_of_high_similarity_context_per_node = int(context_volume/len(entities))
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node_name in node_hits:
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context = get_context_using_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
        return node_context_extracted
    else:
        node_hits = vectorstore.similarity_search_with_score(question, k=5)
        max_number_of_high_similarity_context_per_node = int(context_volume/5)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node in node_hits:
            node_name = node[0].page_content
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context = get_context_using_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
        return node_context_extracted
    