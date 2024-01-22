import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
from os.path import join
import openai
import os


def embedding_computation_contriever(tokenizer, retriever, dire, to_save):
    with open(dire, "r") as f:
        safety_regulations = json.load(f)

    domains = list(safety_regulations.keys())

    for d in tqdm(domains):
        regulations = safety_regulations[d]

        # Apply tokenizer
        inputs = tokenizer(
            regulations, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        outputs = retriever(**inputs)

        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings

        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])

        embedding_directory = join(
            to_save, "safety_regulation_{}_embedding.pt".format(d)
        )

        torch.save(embeddings, embedding_directory)

    return domains


def embedding_computation_mistral(tokenizer, retriever, dire, to_save):
    def last_token_pool(last_hidden_states,attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    with open(dire, "r") as f:
        safety_regulations = json.load(f)

    domains = list(safety_regulations.keys())

    for d in tqdm(domains):
        regulations = safety_regulations[d]

        # Apply tokenizer
        max_length = 4096
        # Tokenize the input texts
        batch_dict = tokenizer(regulations, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

        # Compute token embeddings
        outputs = retriever(**batch_dict)

        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        embedding_directory = join(
            to_save, "safety_regulation_{}_embedding.pt".format(d)
        )

        torch.save(embeddings, embedding_directory)

    return domains



def get_openai_embedding(text, model="text-embedding-ada-002"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def embedding_computation_openai(dire, to_save):
    with open(dire, "r") as f:
        safety_regulations = json.load(f)

    domains = list(safety_regulations.keys())

    for d in tqdm(domains):
        embeddings = []
        regulations = safety_regulations[d]
        for regulation in regulations:
            embedding = get_openai_embedding(regulation)
            embeddings.append(torch.tensor(embedding).unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)

        embedding_directory = join(
            to_save, "safety_regulation_{}_embedding.pt".format(d)
        )

        torch.save(embeddings, embedding_directory)

    return domains
