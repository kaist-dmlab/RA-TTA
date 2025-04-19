import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn.functional as F
import clip
import numpy as np
import argparse
import pickle as pkl
import logging
import ot
import json
from collections import defaultdict
from tqdm import tqdm
from torchvision import transforms
from src.dataloader.templates import ucf101, caltech256, dtd, oxford_flowers, cub200, resisc45, sun397, food101, fgvc_aircraft, stanford_cars, caltech101, oxford_pets
from src.utils.utils import AverageMeter, accuracy
from src.utils.utils import ensure_path, set_seed
from src.utils.logging import get_logger

parser = argparse.ArgumentParser() 
parser.add_argument('--db_root', type=str)
parser.add_argument('--tst_embedding_root', type=str, default='./test_embedding')
parser.add_argument('--dataset', type=str)
parser.add_argument('--database', type=str, default='LAION2B')
parser.add_argument('--net_name', type=str, default='ViT-B/16')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int)
parser.add_argument('--max_threads', type=int, default=4)
parser.add_argument('--output_dir', type=str, default='./logs/ratta')

# RA-TTA
parser.add_argument('--augmentation_size', type=int, default=100)
parser.add_argument('--top_k_d', type=int, default=20)
parser.add_argument('--top_k_s', type=int, default=20)
parser.add_argument('--qthres', type=float, default=0.75)
args = parser.parse_args()
device = f'cuda:{args.gpu}'
set_seed(args.seed)

dataset  = args.dataset
database = args.database
net_name = args.net_name
cupl_root = './descriptions/CuPL_prompts_{}.json'
tst_embedding_root = args.tst_embedding_root
args.output_dir = os.path.join(args.output_dir, net_name.replace('/','_'))
ensure_path('./imgs')
ensure_path(args.output_dir)

logger = logging.getLogger('RATTA')
logger = get_logger('RATTA', os.path.join(args.output_dir, dataset))
logger.info(f"Seed: {args.seed}")

# RATTA config
AS = args.augmentation_size
TOPK_D = args.top_k_d
TOPK_S = args.top_k_s
THRES = args.qthres
BS=args.batch_size

logger.info(f"Augmentation size: {AS}")
logger.info(f"# descriptions: {TOPK_D}")
logger.info(f"# samples: {TOPK_S}")
logger.info(f"Q threshold: {THRES}")

# Load external vector DB
if 'food101' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = food101
    pre_template = 'a photo of a {}, a type of food. ' 
elif 'fgvc_aircraft' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = fgvc_aircraft
    pre_template = 'a photo of a {} from Fine-Grained Visual Classification of Aircraft. ' 
elif 'stanford_cars' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = stanford_cars
    pre_template = 'a photo of a {}. ' 
elif 'caltech101' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = caltech101
    pre_template = 'a photo of a {}. '
elif 'caltech256' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = caltech256
    pre_template = 'a photo of a {}. ' 
elif 'ucf101' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = ucf101
    pre_template = 'a photo of a person doing {}. ' 
elif 'sun397' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = sun397
    pre_template = 'a photo of a {}. '
elif 'resisc45' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = resisc45
    pre_template = 'a satellite photo of the {}. '
elif 'cub200' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = cub200
    pre_template = 'a photo of a {}, a type of bird. ' 
elif 'flowers102' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = oxford_flowers
    pre_template = 'a photo of a {}, a type of flower. ' 
elif 'oxford_pets' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = oxford_pets
    pre_template = 'a photo of a {}, a type of pet. ' 
elif 'dtd' in dataset:
    _net_name = net_name.replace('/', '_')
    aux_embedding_path = os.path.join(args.db_root, f'{database}_{dataset}_aux_{_net_name}')
    templates = dtd
    pre_template = 'a photo of a {}, which is a type of texture, or a type of pattern. '
else:
    exit()
    
with open(f'{aux_embedding_path}', 'rb') as f:
    file = pkl.load(f)
    aux_embeds = file['embedding'].astype('float16')
aux_size = aux_embeds.shape
logger.info(f"Database size: {aux_embeds.shape}")

array_size_in_bytes = aux_embeds.nbytes  
array_size_in_mb = array_size_in_bytes / (1024 ** 2) 
logger.info(f"Array size: {array_size_in_mb:.2f} MB")

import faiss                   
dimension = aux_embeds.shape[1]
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(dimension)
index = faiss.index_cpu_to_gpu(res, args.gpu, index)
index.add(aux_embeds)

from src.dataloader.stanford_cars import CARS_CLASSNAMES
from src.dataloader.oxford_flowers import FLOWERS102_CLASSNAMES
from src.dataloader.food101 import FOOD101_CLASSNAMES
from src.dataloader.oxford_pets import PETS_CLASSNAMES
from src.dataloader.dtd import DTD_CLASSNAMES
from src.dataloader.cub200 import CUB_CLASSNAMES
from src.dataloader.fgvc_aircraft import AIRCRAFT_CLASSNAMES
from src.dataloader.resisc45 import RESISC45_CLASSNAMES
from src.dataloader.sun397 import SUN397_CLASSNAMES
from src.dataloader.ucf101 import UCF101_CLASSNAMES
from src.dataloader.caltech101 import CALTECH101_CLASSNAMES
from src.dataloader.caltech256 import CALTECH256_CLASSNAMES

_CLASSNAMES = {
        'stanford_cars': CARS_CLASSNAMES,
        'flowers102': FLOWERS102_CLASSNAMES,
        'food101': FOOD101_CLASSNAMES,
        'oxford_pets': PETS_CLASSNAMES,
        'dtd': DTD_CLASSNAMES,
        'cub200': CUB_CLASSNAMES,
        'fgvc_aircraft': AIRCRAFT_CLASSNAMES,
        'resisc45': RESISC45_CLASSNAMES,
		'sun397': SUN397_CLASSNAMES,
		'ucf101': UCF101_CLASSNAMES,
		'caltech101': CALTECH101_CLASSNAMES,
		'caltech256': CALTECH256_CLASSNAMES,
        }

IMAGE_SIZE = 336 if '336' in net_name else 224
CLASSNAMES = [i.replace('_', ' ') for i in _CLASSNAMES[args.dataset]]
CHANNEL_SIZE = 3
NUM_CLASSES = len(CLASSNAMES)

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
        ])

def aug_crop(image, preprocess):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed

class CropAugmenter(object):
    def __init__(self, base_transform, n_views):
        self.base_transform = base_transform
        self.preprocess = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                                ),
                            ])
        self.n_views = n_views

    def __call__(self, x):
        image = self.base_transform(x)
        views = [aug_crop(x, self.preprocess) for _ in range(self.n_views)]
        return image, views

pre_features = torch.load(f"{tst_embedding_root}/{net_name.replace('/', '_')}/{str(args.seed)}/{dataset}.pth")
clip_model, _ = clip.load(net_name, 
                          device=device) 

clip_model.eval() 
for _, param in clip_model.named_parameters():
        param.requires_grad_(False)
SCALING = clip_model.logit_scale.exp().cpu()

def CuPL_classifier(classnames, gpt3_templates, model, templates=None):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if templates is not None:
                texts = [template.format(classname.replace('_', ' ')) for template in templates]
            else:
                texts = []
            for t in gpt3_templates[classname.replace('_', ' ')]:
                texts.append(t)
            texts = clip.tokenize(texts, truncate=True).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # L2 normalise text embedding
            class_embedding = class_embeddings.mean(dim=0) # take mean over all text embeddings for all prompts
            class_embedding /= class_embedding.norm() # L2 normalise mean embedding
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

# description embeddings
cupl_prompts = json.load(open(cupl_root.format(dataset)))
cupl_weights = CuPL_classifier(CLASSNAMES, cupl_prompts, clip_model, templates)
DESC_TXT_LST = np.concatenate(list(cupl_prompts.values()))
description_encodings = defaultdict()

for c_idx, c_name in enumerate(CLASSNAMES):
    v = [pre_template.format(c_name.replace('_', ' ')) + v_i for v_i in cupl_prompts[c_name]]
    v.append(pre_template.format(c_name.replace('_', ' '))[:-1])
    tokens = clip.tokenize(v, truncate=True).to(device)
    description_encodings[c_idx] = F.normalize(clip_model.encode_text(tokens))
description_embeddings = torch.cat(list(description_encodings.values()))

NAMES = []
for c_idx, c_name in enumerate(CLASSNAMES):
    NAMES += [int(c_idx)] * (len(cupl_prompts[c_name])+1)
NAMES = torch.Tensor(NAMES).to(device=device, dtype=torch.int16)

def _get_global_prototype(sample_cls_name2desc_id, device):
    # If there is no top-k descriptions,
    if len(sample_cls_name2desc_id.values()) == 0:
        return None
    texts_embeds = []
    with torch.no_grad():
        texts_embeds += list(torch.unbind(description_embeddings[np.concatenate(list(sample_cls_name2desc_id.values())).tolist()], dim=0))
        texts_embeds = torch.stack(texts_embeds, dim=0)
        cls_txt_embeds = texts_embeds.mean(dim=0)
        cls_txt_embeds /= cls_txt_embeds.norm() 
    tst_prototype = cls_txt_embeds.unsqueeze(1).to(device)
    return tst_prototype

def _get_set_prototypes(sample_cls_name2desc_id, device):
    selected_prompts_embeds = []
    selected_prompts_keys = []
    # If there is no top-k descriptions,
    if len(sample_cls_name2desc_id.values()) == 0:
        return None, None
    with torch.no_grad():
        for key, prompt_ids in sample_cls_name2desc_id.items():
            texts_embeds = []
            texts_embeds += list(torch.unbind(description_embeddings[prompt_ids], dim=0))
            texts_embeds = torch.stack(texts_embeds, dim=0)
            cls_txt_embeds = texts_embeds.mean(dim=0)
            cls_txt_embeds /= cls_txt_embeds.norm() 
            selected_prompts_embeds.append(cls_txt_embeds)
            selected_prompts_keys.append(key)
        selected_prompts_keys = torch.Tensor(selected_prompts_keys).to(device=device, dtype=torch.int64)
        return torch.stack(selected_prompts_embeds, dim=1).to(device), F.one_hot(selected_prompts_keys, num_classes=NUM_CLASSES)

def _compute_prototypes(sample_id, sample_topk_desc_ids, device, threshold=5):
    # 1. Preprocess extracted description IDs
    sample_cls_name2desc_id = defaultdict(list)
    selected_classes = NAMES[sample_topk_desc_ids]
    for x, y in zip(selected_classes, sample_topk_desc_ids):
        sample_cls_name2desc_id[x.item()].append(y.item())
    keys_to_delete = [key for key, val in sample_cls_name2desc_id.items() if len(set(val)) < threshold]
    for key in keys_to_delete:
        del sample_cls_name2desc_id[key]
        
    # 2. Compute prototypes using the preprocessed descriptions
    tst_prototype  = _get_global_prototype(sample_cls_name2desc_id, device)
    set_prototypes, set_pseudo_labels = _get_set_prototypes(sample_cls_name2desc_id, device)
    return sample_id, tst_prototype, set_prototypes, set_pseudo_labels

def compute_prototypes(topk_desc_ids, device):
    batch_tst_results = defaultdict(dict)
    batch_set_results = defaultdict(dict)
    batch_set_pl_results = defaultdict(dict)
    batch_id = 0
    args_lst = [(batch_id, topk_desc_ids[batch_id], device, 5)]
    batch_id, tst_prototype, set_prototypes, set_pseudo_labels = _compute_prototypes(*args_lst[0])
    batch_tst_results[batch_id] = tst_prototype
    batch_set_results[batch_id] = set_prototypes
    batch_set_pl_results[batch_id] = set_pseudo_labels
    batch_tst_results = dict(sorted(batch_tst_results.items()))
    batch_set_results = dict(sorted(batch_set_results.items()))
    batch_set_pl_results = dict(sorted(batch_set_pl_results.items()))
    return batch_tst_results, batch_set_results, batch_set_pl_results

def _compute_ot_based_relevance_score(id, 
                                      tst_x_embed, 
                                      aug_x_embeds, 
                                      retrieved_embeds_lst, 
                                      tst_prototype,
                                      set_prototypes, 
                                      device):
    set_distance_lst = []
    augmented_tst_embeds = torch.cat([tst_x_embed, aug_x_embeds], dim=0)
    tst_distribution = (SCALING * augmented_tst_embeds @ tst_prototype).softmax(dim=0).squeeze().cpu().numpy().astype('float16')
    for set_id in range(set_prototypes.size(1)):
        # calculate cost mtx
        retrieved_set_embeds = retrieved_embeds_lst[set_id].to(device)
        set_prototype = set_prototypes[:, set_id]
        A = (augmented_tst_embeds @ set_prototype).unsqueeze(0).repeat_interleave(retrieved_set_embeds.shape[0], dim=0).T
        B = (retrieved_set_embeds @ set_prototype).unsqueeze(0).repeat_interleave(augmented_tst_embeds.shape[0], dim=0)
        C = torch.abs((A-B)).cpu().numpy().astype('float16')
        # reference distribution
        retrieved_distribution = (SCALING * retrieved_set_embeds @ set_prototype).softmax(dim=-1).cpu().numpy().astype('float16')
        # calculate OT distance
        ot_reg = ot.sinkhorn(tst_distribution, retrieved_distribution, C, reg=0.1, numItermax=100, stopThr=1e-2)
        set_distance_lst.append((np.sum(ot_reg * C)))
    set_relevance = (SCALING * 1/(np.array(set_distance_lst)+1))
    set_relevance = set_relevance.softmax(dim=-1).to(device=device, dtype=torch.float16)
    return id, set_relevance

def compute_set_relevance_score(tst_x_embeds, 
                                aug_x_embeds, 
                                batch_retrieved_embeds, 
                                batch_tst_prototype,
                                batch_set_prototypes, 
                                device):
    batch_results = defaultdict(dict)
    args_lst = []
    batch_id = 0
    args_lst.append((batch_id, 
                     tst_x_embeds, 
                     aug_x_embeds[batch_id], 
                     batch_retrieved_embeds, 
                     batch_tst_prototype,
                     batch_set_prototypes,
                     device))
    batch_id, batch_set_relevance = _compute_ot_based_relevance_score(*args_lst[0])
    batch_results[batch_id] = batch_set_relevance
    batch_results = dict(sorted(batch_results.items()))
    return batch_results

def _compute_retrieval_pred(sample_id, 
                           sample_relevance_scores, 
                           sample_set_level_pl):
    retrieval_pred = sample_relevance_scores @ sample_set_level_pl
    return sample_id, retrieval_pred

def compute_pred(batch_set_relevance_scores, batch_set_level_pl):
    batch_results = defaultdict(dict)
    args_lst = []
    batch_id = 0
    args_lst.append((batch_id, 
                     batch_set_relevance_scores[batch_id],
                     batch_set_level_pl))
    batch_id, batch_retrieval_pred = _compute_retrieval_pred(*args_lst[0])
    batch_results[batch_id] = batch_retrieval_pred
    batch_results = dict(sorted(batch_results.items()))
    return batch_results

with torch.no_grad():
    cupl_top1 = AverageMeter('Acc@1', ':6.2f')
    cupl_top5 = AverageMeter('Acc@5', ':6.2f')
    adapt_top1 = AverageMeter('Acc@1', ':6.2f')
    adapt_top5 = AverageMeter('Acc@5', ':6.2f') 
    ret_top1 = AverageMeter('Acc@1', ':6.2f')
    ret_top5 = AverageMeter('Acc@5', ':6.2f') 
    for i, (tst_x_embeds, aug_x_embeds, y) in enumerate(tqdm(pre_features)):
        # Load pre-computed embeddings
        tst_x_embeds = tst_x_embeds.to(device=device)
        aug_x_embeds = aug_x_embeds.to(device=device)
        tst_y = y.to(device=device)
        
        # Test Image Embedding
        tst_img_desc_sim = [tst_x_embeds @ desc.T for _, desc in description_encodings.items()]
        tst_img_desc_sim = torch.cat(tst_img_desc_sim, dim=1)
        tst_logits = SCALING * tst_x_embeds @ cupl_weights
        tst_entp = -(tst_logits.softmax(-1) * tst_logits.log_softmax(-1)).sum(-1)
        tst_preds = tst_logits.softmax(dim=-1)
        
        # Augmented Images Embeddings
        aug_img_desc_sim = [aug_x_embeds @ desc.T for _, desc in description_encodings.items()]
        aug_img_desc_sim = torch.cat(aug_img_desc_sim, dim=-1)
        aug_x_embeds = aug_x_embeds.reshape(BS, AS, -1) # [BS, AS, ED]
        aug_img_desc_sim = aug_img_desc_sim.reshape(BS, AS ,-1) # [BS, AS, FD]

        augmented_desc_sims = torch.cat([tst_img_desc_sim.unsqueeze(1), aug_img_desc_sim], dim=1) # [BS, (AS+1), FD]
        augmented_dsc_quantile_sims = torch.quantile(augmented_desc_sims.to(dtype=torch.float32), THRES, dim=1) # [BS, FD]
        top_k_desc_vals, top_k_desc_ids= augmented_dsc_quantile_sims.topk(k=TOPK_D, dim=-1) # [BS, TOPK_D]
        batch_tst_prototypes, batch_set_prototypes, batch_set_pl_results = compute_prototypes(top_k_desc_ids, device)

        if batch_set_prototypes[0] == None:
            adapted_preds = tst_preds
            batch_retrieval_preds = tst_preds
            (cupl_acc1, cupl_acc5), _ = accuracy(tst_preds, tst_y, topk=(1, 5))
            cupl_top1.update(cupl_acc1[0], tst_x_embeds.size(0))
            cupl_top5.update(cupl_acc5[0], tst_x_embeds.size(0))

            (adapt_acc1, adapt_acc5), _ = accuracy(adapted_preds, tst_y, topk=(1, 5))
            adapt_top1.update(adapt_acc1[0], tst_x_embeds.size(0))
            adapt_top5.update(adapt_acc5[0], tst_x_embeds.size(0))
            continue

        batch_tst_prototypes = torch.cat(list(batch_tst_prototypes.values()), dim=1)
        batch_set_prototypes = torch.cat(list(batch_set_prototypes.values()), dim=1)
        batch_set_pseudo_labels = torch.cat(list(batch_set_pl_results.values()), dim=0).to(dtype=torch.float16)
        batch_retrieved_embeds = []
        _, top_aux_ids = index.search(batch_set_prototypes.T.cpu().numpy(), TOPK_S)
        for col_id in range(batch_set_prototypes.shape[1]):
            batch_retrieved_embeds.append(torch.from_numpy(aux_embeds[top_aux_ids[col_id], :]))
        # Calculate OT-based set relevance scores
        batch_set_relevance_scores = compute_set_relevance_score(tst_x_embeds,
                                                                 aug_x_embeds,
                                                                 batch_retrieved_embeds,
                                                                 batch_tst_prototypes,
                                                                 batch_set_prototypes,
                                                                 device)
        
        # Retrieval-based prediction
        batch_retrieval_preds = compute_pred(batch_set_relevance_scores, batch_set_pseudo_labels)
        batch_retrieval_preds = torch.stack(list(batch_retrieval_preds.values()), dim=0)
        batch_retrieval_entp = (-1) * (batch_retrieval_preds * torch.log((batch_retrieval_preds + 1e-7))).sum(dim=-1)
        
        # Prediction ensemble
        lambda_tst = (1/(1+tst_entp)).to(device).exp()
        lambda_ret = (1/(1+batch_retrieval_entp)).to(device).exp()
        lambda_ = ((lambda_tst/(lambda_tst + lambda_ret)).unsqueeze_(1))
        adapted_preds = lambda_ * tst_preds + (1-lambda_) * batch_retrieval_preds
            
        (cupl_acc1, cupl_acc5), _ = accuracy(tst_preds, tst_y, topk=(1, 5))
        cupl_top1.update(cupl_acc1[0], tst_x_embeds.size(0))
        cupl_top5.update(cupl_acc5[0], tst_x_embeds.size(0))

        (adapt_acc1, adapt_acc5), _ = accuracy(adapted_preds, tst_y, topk=(1, 5))
        adapt_top1.update(adapt_acc1[0], tst_x_embeds.size(0))
        adapt_top5.update(adapt_acc5[0], tst_x_embeds.size(0))

        if (i+1) % 100 == 0:
            logger.info(f"Iter: {i}")
            logger.info(f"CuPL - Top-1 acc: {cupl_top1.avg} \t Top-5 acc: {cupl_top5.avg}")
            logger.info(f"RA-TTA - Top-1 acc: {adapt_top1.avg} \t Top-5 acc: {adapt_top5.avg}")
        
logger.info(f"Dataset: {dataset}")
logger.info(f"CuPL - Top-1 acc: {cupl_top1.avg} \t Top-5 acc: {cupl_top5.avg}")
logger.info(f"RA-TTA - Top-1 acc: {adapt_top1.avg} \t Top-5 acc: {adapt_top5.avg}")
logger.info(f"Exp end")