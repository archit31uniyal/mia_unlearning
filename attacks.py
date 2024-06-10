import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import transformers
import datasets
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from collections import defaultdict
import os
import accelerate
from tqdm import tqdm
import zlib

def load_csv(path):
    """
    Function to load csv files
    """
    return pd.read_csv(path)

def write_csv(df, path):
    """
    Function to write data to csv
    """
    df.to_csv(path, index=False)

def get_lira(loader, progress= True):
    """
    Function to perform LiRA attack on a particular data loader
    
    Inputs:
        loader: DataLoader
        progress: flag to activate/deactivate progress bar

    Output:
        texts: text block
        lira: LiRA scores for corresponding text blocks
    """ 
    with torch.no_grad():
        lira = []
        texts = []
        if progress:
            progress_bar = tqdm(loader, desc="Calculating LIRA")
        else:
            progress_bar = loader

        for text in progress_bar:
            texts.append(text)
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            tokenized_ref = ref_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels_ref = tokenized_ref.input_ids
            tokenized = {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}
            tokenized_ref = {"input_ids": tokenized_ref.input_ids, "attention_mask": tokenized_ref.attention_mask}
            # Computing loss for loader using base model
            lls =  -base_model(**tokenized, labels=labels).loss.item()

            # Computing loss for loader using reference model
            lls_ref = -ref_model(**tokenized_ref, labels=labels_ref).loss.item()

            # Computing LiRA score
            lira.append(lls - lls_ref)
    return texts, lira

def calculatePerplexity(sentence, model, tokenizer, device='cuda'):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def get_minK(loader, progress= True):
    """
    Function to perform min_k attack on a particular set of data
    Inputs:
        loader: DataLoader
        progress: flag to activate/deactivate progress bar

    Output:
        texts: text block
        min_k: Min_k scores for corresponding text blocks
    """
    min_k = []
    texts = []
    if progress:
        progress_bar = tqdm(loader, desc="Calculating min-k")
    else:
        progress_bar = loader

    with torch.no_grad():
        for text in progress_bar:
            texts.append(text)

            # Calculate perplexity scores of text blocks using base_model and ref_model
            _, all_prob, _ = calculatePerplexity(text, base_model, base_tokenizer, device=DEVICE)
            _, all_prob_ref, _ = calculatePerplexity(text, ref_model, ref_tokenizer, device=DEVICE)

            # min-k prob
            ratio = args.ratio
            # Extracting k% tokens with lowest probability for base model
            k_length = int(len(all_prob)*ratio)
            topk_prob = np.sort(all_prob)[:k_length]
            pred_base = -np.mean(topk_prob).item()
            
            # Extracting k% tokens with lowest probability for reference model
            k_length = int(len(all_prob_ref)*ratio)
            topk_prob = np.sort(all_prob_ref)[:k_length]
            pred_ref = -np.mean(topk_prob).item()

            score = pred_base/pred_ref
            # score = pred_base

            min_k.append(score)
    
    return texts, min_k

def get_loss(loader, progress= True):
    """
    Function to perform LOSS attack on a particular data loader
    
    Inputs:
        loader: DataLoader
        progress: flag to activate/deactivate progress bar

    Output:
        texts: text block
        loss: loss scores for corresponding text blocks
    """
    with torch.no_grad():
        loss = []
        texts = []
        if progress:
            progress_bar = tqdm(loader, desc="Calculating loss")
        else:
            progress_bar = loader

        for text in progress_bar:
            texts.append(text)
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            tokenized = {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}
            loss.append(base_model(**tokenized, labels=labels).loss.item())
    return texts, loss

def zlib_entropy(loader, progress= True):
    """
    Function to perform zlib entropy attack on a particular data loader
    
    Inputs:
        loader: DataLoader
        progress: flag to activate/deactivate progress bar

    Output:
        texts: text block
        zlib: zlib scores for corresponding text blocks
    """
    with torch.no_grad():
        zlib_score = []
        texts = []
        if progress:
            progress_bar = tqdm(loader, desc="Calculating loss")
        else:
            progress_bar = loader

        for text in progress_bar:
            texts.append(text)
            zlib_compress = len(zlib.compress(bytes(text, 'utf-8')))
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            tokenized = {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}
            zlib_score.append(base_model(**tokenized, labels=labels).loss.item()/zlib_compress)
    return texts, zlib_score
            
def get_model_and_tokenizer(model_name):
    """
    Utility function to load model and tokenizer

    model_name: HF name or model_path
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code = True, cache_dir=cache_dir).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer

def strip_newlines(text):
    """
    Function to remove newline escape sequences
    """
    return ' '.join(text.split())

def generate_data(dataset, key, args, train=True):
    """
    Data loading and preprocessing function

    Inputs:
        dataset: could be the name/path of the dataset or a List object
        key: column to be fetched from dataset
        args: other global arguments that are utilized by the function
        train: flag to fetch data from the train set or test set
    
    Output:
        data: list containing preprocessed text
    """
    # load data
    data_split = 'train' if train else 'test'
    if type(dataset) == str:
        if dataset == 'the_pile' and data_split=='train':
            data = datasets.load_dataset("json", data_files= 'cache_100_200_1000_512/train/the_pile_pubmed_abstracts.json', cache_dir=cache_dir, trust_remote_code=True)[data_split][key]
        elif dataset == 'the_pile' and data_split=='test':
            print("test")
            data = datasets.load_dataset("json", data_files="cache_100_200_1000_512/train/the_pile_pubmed_abstracts.json", cache_dir=cache_dir, trust_remote_code=True)[data_split][key]
        elif 'book' in dataset:
            # Book Corpus dataset
            data = datasets.load_dataset('csv', data_files = {'train': dataset}, cache_dir=cache_dir, trust_remote_code=True)["train"][key]
        elif dataset == 'wikitext':
            data = datasets.load_dataset('wikitext', 'wikitext-2-v1', cache_dir=cache_dir, ignore_verifications=True, trust_remote_code=True)[data_split][key]
        elif dataset == 'imdbhface':
            # imdb dataset
            data = datasets.load_dataset('imdb', split=data_split, cache_dir=cache_dir, trust_remote_code=True)[key]
            data = data[:5000]
        elif 'harry_potter' in dataset or 'hp' in dataset:
            # Preprocessed Harry Potter books dataset
            data = datasets.load_dataset('csv', data_files={'train': dataset}, cache_dir=cache_dir, trust_remote_code=True)[data_split][key]
            # data = data[:149]
        else:
            # Other datasets
            data = datasets.load_dataset(dataset, split=f'train[:10000]', cache_dir=cache_dir, trust_remote_code=True)[key]
        
        # remove duplicates from the data
        data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
        
        # strip whitespace around each example
        data = [x.strip() for x in data]

        # remove newlines from each example
        data = [strip_newlines(x) for x in data]
        # print(len(data))
    else:
        data = dataset

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples and shuffle
    # then generate n_samples samples

    # try to keep only examples with > 100 words
    #if dataset in ['writing', 'squad', 'xsum']:
    # print(len(data))
    long_data = [x for x in data if len(x.split()) > 100]
    # print(len(long_data))
    if len(long_data) > 0:
        data = long_data
    
    # print(len(data))
    
    not_too_long_data = [x for x in data if len(x.split()) < args.max_length]
    if len(not_too_long_data) > 0:
        data = not_too_long_data
    # print(len(data))

    random.seed(args.seed)
    random.shuffle(data)

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data

def generate_threshold(criterion_fn, name, non_member, target_false_positive_rate=0.01):
    """
    Generate a threshold for low false positive rates in a membership inference attack.
    Inputs:
        criterion_fn: function to calculate particular MIA score (LiRA, Min-k, loss, etc)
        name: name of attack
        non_member: non-member dataset
        target_false_positive_rate: desired false positive rate
    """
    preds = []
    for text in tqdm(non_member, desc= f"Calculating threshold {name}", total = len(non_member)):
        _, crit = criterion_fn(text, progress=False)
        preds.append(crit[0])
    
    preds.sort()
    threshold = preds[int(len(preds)*(1-target_false_positive_rate))]
    print(f'Threshold {name}_criterion: {threshold}')
    return threshold

def attack(criterion_fn, examples, name, threshold, label):
    """
    Function to generate and save attack data
    Inputs:
        criterion_fn: function to calculate particular MIA score (LiRA, Min-k, loss, etc)
        examples: data loader
        name: name of attack
        threshold: threshold for the attack
        label: label for the attack
    """
    os.makedirs(os.path.join(args.output_dir, args.target_model), exist_ok=True)
    
    df1 = defaultdict(list)
    for i, example in tqdm(enumerate(examples), desc=f"Calculating {name}", total = len(examples)):
        # df = df.append({'text': example, 'threshold': threshold, 'criterion': criterion_fn(example)[1], 'label': label, 'prediction': 1 if df.iloc[i, 2] > df.iloc[i, 1] else 0})
        df1['text'].append(example)
        df1['threshold'].append(threshold)
        df1['criterion'].append(criterion_fn(example)[1][0])
        df1['label'].append(label)
        df1['prediction'].append(1 if df1['criterion'][i] > df1['threshold'][i] else 0)
    
    df = pd.DataFrame(df1)
    write_csv(df, f"{name}.csv")

if __name__ == "__main__":
    global args, cache_dir

    DEVICE='cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="gpt2", help="Target model name")
    parser.add_argument("--ref_model", type=str, default="gpt2", help="Reference model name")
    parser.add_argument("--member1", type=str, default="wikitext", help="member1 dataset")
    parser.add_argument("--key1", type=str, default="text", help="key to extract from member1 dataset")
    parser.add_argument("--member2", type=str, default="wikitext", help="member2 dataset")
    parser.add_argument("--key2", type=str, default="text", help="key to extract from member2 dataset")
    parser.add_argument("--nonmember", type=str, default="wikitext", help="nonmember dataset")
    parser.add_argument("--key", type=str, default="text", help="key to extract from nonmember dataset")
    parser.add_argument("--max_length", type=int, default=512, help="max length of examples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--recompute", action="store_true", help="recompute the threshold and data")
    parser.add_argument("--ratio", type=float, default=0.20, help="Ratio of labels to be considered in Min-k attack")
    parser.add_argument("--cache_dir", type=str, default="./", help="Path to cache directory")
    parser.add_argument("--output_dir", type=str, default = "./")
    args = parser.parse_args()
    cache_dir = args.cache_dir
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # accelerator1 = accelerate.Accelerator()
    # accelerator2 = accelerate.Accelerator()
    base_model, base_tokenizer = get_model_and_tokenizer(args.target_model)
    ref_model, ref_tokenizer = get_model_and_tokenizer(args.ref_model)
    member1 = generate_data(args.member1, args.key1, args, train=True)
    member2 = generate_data(args.member2, args.key2, args, train=True)
    nonmember = generate_data(args.nonmember, args.key, args, train=True)

    member1_loader = DataLoader(member1, batch_size=1, shuffle=False)
    member2_loader = DataLoader(member2, batch_size=1, shuffle=False)
    nonmember_loader = DataLoader(nonmember, batch_size=1, shuffle=False)

    # base_model = accelerator1.prepare(base_model)
    # ref_model = accelerator2.prepare(ref_model)
    # member1_loader, member2_loader, nonmember_loader = accelerator1.prepare(member1_loader, member2_loader, nonmember_loader)
    
    for name in ['lira_threshold', 'min_k_threshold', 'loss_threshold', 'zlib_threshold']:
        if name == 'lira_threshold':
            criterion_fn = get_lira
        elif name == 'min_k_threshold':
            criterion_fn = get_minK
        elif name == 'loss_threshold':
            criterion_fn = get_loss
        elif name == 'zlib_threshold':
            criterion_fn = zlib_entropy

        # print(os.path.exists(os.path.join(args.output_dir, f"{args.target_model}_{name}_nonmember.csv")))
        # exit(0)
        if os.path.exists(os.path.join(args.output_dir, f"{args.target_model}_{name}_nonmember.csv")) and not args.recompute:
            print(f"File already exists. Skipping {name}....")
            continue

        threshold = generate_threshold(criterion_fn, name, nonmember_loader, target_false_positive_rate=0.05)

        attack(criterion_fn, member1_loader, os.path.join(args.output_dir, f"{args.target_model}_{name}_member1"), threshold, 1)
        attack(criterion_fn, member2_loader, os.path.join(args.output_dir, f"{args.target_model}_{name}_member2"), threshold, 1)
        attack(criterion_fn, nonmember_loader, os.path.join(args.output_dir, f"{args.target_model}_{name}_nonmember"), threshold, 0)