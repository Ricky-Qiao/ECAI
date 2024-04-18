import os

import torch
import torch.nn as nn
from clip import clip
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity

from metrics import compute_accuracy
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES


def kmedoids(features, k, max_iter=100):
    # Initialize medoids by randomly selecting data points
    n_samples = features.shape[0]
    medoids_idx = torch.randperm(n_samples)[:k]
    medoids = features[medoids_idx]
    
    for _ in range(max_iter):
        # Assign each sample to the nearest medoid
        sims = cosine_similarity(features.unsqueeze(1), medoids.unsqueeze(0), dim=2)
        labels = torch.argmax(sims, dim=1)
        
        # Update medoids
        new_medoids = torch.empty_like(medoids)
        for i in range(k):
            cluster_samples = features[labels == i]
            cluster_sims = cosine_similarity(cluster_samples.unsqueeze(1), cluster_samples.unsqueeze(0), dim=2)
            medoid_idx = torch.argmax(cluster_sims.sum(dim=0))
            new_medoids[i] = cluster_samples[medoid_idx]
        
        # Check for convergence
        if torch.all(torch.eq(medoids, new_medoids)):
            break
        
        medoids = new_medoids
    
    return medoids


def kcenter_greedy(features, confidences, k):
    # Find the indices of the two least confident samples
    least_confident_indices = torch.argsort(confidences)[:2]

    # Initialize the first two centers with the least confident samples
    centers = torch.empty((k, features.shape[1]), dtype=features.dtype, device=features.device)
    centers[:2] = features[least_confident_indices]

    # Compute similarity between features and the first center
    sims = cosine_similarity(features, centers[0].unsqueeze(0), dim=1)

    for i in range(2, k):
        # Find the data point farthest from the current centers
        farthest_idx = torch.argmin(sims)
        
        # Update the centers
        centers[i] = features[farthest_idx]

        # Update distances to the new center
        new_sims = cosine_similarity(features, centers[i].unsqueeze(0), dim=1)
        sims = torch.max(sims, new_sims)

    return centers


def cal_criterion(cfg, prototypes, text_features):
    ratio = cfg.MODEL.Dorset.LAMBDA
    num_features = cfg.MODEL.Dorset.NUM_FEATURES
    mul = prototypes * text_features.unsqueeze(0)
    sim = torch.mean(mul, dim=0)
    criterion =  ratio * sim + (-1) * (1 - ratio) * torch.var(prototypes, dim=0)
    _, indices = torch.topk(criterion, k=num_features)
    return indices


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, text_features, keys, values, domains, confidences):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.cuda.current_device()

        self.text_features = text_features

        self.keys, self.indices = self.build_coreset(keys, values, domains, confidences)
        
    def build_coreset(self, keys, values, domains, confidences):
        class_keys = []
        class_indices = []
        class_nums = torch.unique(values)
        domain_nums = torch.unique(domains)
        for i in class_nums:
            class_coresets = []
            for j in domain_nums:
                class_idx = values == i
                subset_keys = keys[class_idx]
                subset_domains = domains[class_idx]
                subset_confidences = confidences[class_idx]
                domain_idx = subset_domains == j
                if torch.sum(domain_idx) > 0:
                    coreset = kcenter_greedy(subset_keys[domain_idx], subset_confidences[domain_idx], self.cfg.MODEL.Dorset.K)
                    class_coresets.append(coreset)
            class_coresets = torch.cat(class_coresets, dim=0)
            class_keys.append(class_coresets)
            class_indices.append(cal_criterion(self.cfg, class_coresets, self.text_features[i]))
        
        return class_keys, class_indices

    def forward(self, image, alpha, beta, gamma):
        # Zeroshot logits
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        clip_logits = logit_scale * image_features @ self.text_features.t()

        # Domain prototypes logits
        cache_similarities =[]
        soft_values = []
        for i in range(len(self.keys)):
            indices = self.indices[i]
            keys = self.keys[i][:, indices]
            new_image_features = image_features[:, indices]
            similarities = new_image_features @ keys.t()
            similarities = similarities / keys.shape[0]
            cache_similarities.append(similarities)

            new_text_features = self.text_features[:, indices]
            key_logits = keys @ new_text_features.t()
            key_logits = key_logits.softmax(1)
            values = torch.full((key_logits.shape[0],key_logits.shape[1]), 0, dtype=self.dtype, device=self.device)
            values[:, i] = 1
            KL_div = torch.sum(values * torch.log2((values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]
            values = values * (KL_div * gamma).exp()
            soft_values.append(values)

        cache_similarities = torch.cat(cache_similarities, dim=1)
        soft_values = torch.cat(soft_values, dim=0)
        cache_logits = cache_similarities @ soft_values

        # Combine the logits
        cache_logits = ((-1) * (beta - beta * cache_logits)).exp() 
        logits = (
            alpha * cache_logits + clip_logits
        )
        
        return logits


@MODEL_REGISTRY.register()
class Dorset(Trainer):
    def build_cache(self):
        keys = []
        values = []
        domains = []
        confidences = []

        with torch.no_grad():
            for _, batch_data in enumerate(self.data_loader_train):
                image, class_label, domain_label = self.parse_batch_cache(batch_data)
                image_features = self.clip_model.encode_image(image)

                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ self.text_features.t()
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[torch.arange(probs.shape[0]), class_label]

                keys.append(image_features)
                values.append(class_label)
                domains.append(domain_label)
                confidences.append(confidence)
            
            keys = torch.cat(keys, dim=0)
            keys = keys / keys.norm(dim=-1, keepdim=True)
            values = torch.cat(values, dim=0)
            domains = torch.cat(domains, dim=0)
            confidences = torch.cat(confidences, dim=0)
            data = {'keys': keys,
                    'values': values,
                    'domains': domains,
                    'confidences': confidences}
            torch.save(data, 'source_domains.pth')

        return keys, values, domains, confidences
    
    def search_hp(self):
        alpha_list = [i * (self.cfg.MODEL.Dorset.SEARCH_SCALE[0] - 0.1) / self.cfg.MODEL.Dorset.SEARCH_STEP[0] + 0.1 for i in range(self.cfg.MODEL.Dorset.SEARCH_STEP[0])]
        beta_list = [i * (self.cfg.MODEL.Dorset.SEARCH_SCALE[1] - 0.1) / self.cfg.MODEL.Dorset.SEARCH_STEP[1] + 0.1 for i in range(self.cfg.MODEL.Dorset.SEARCH_STEP[1])]
        gamma_list = [i * self.cfg.MODEL.Dorset.SEARCH_SCALE[2] / self.cfg.MODEL.Dorset.SEARCH_STEP[2] for i in range(self.cfg.MODEL.Dorset.SEARCH_STEP[2])]

        best_acc = 0
        best_alpha, best_beta, best_gamma = 0, 0, 0
        for alpha in alpha_list:
            for beta in beta_list:
                for gamma in gamma_list:
                    acc = 0
                    num_images = 0
                    with torch.no_grad():
                        for _, batch_data in enumerate(self.data_loader_val):
                            image, class_label, _ = self.parse_batch_cache(batch_data)
                            output = self.model(image, alpha, beta, gamma)
                            acc += compute_accuracy(output, class_label)[0].item()*len(image)
                            num_images += len(image)
                    acc = acc / num_images
                    if acc > best_acc:
                        print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(alpha, beta, gamma, acc))
                        best_acc = acc
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

        return best_alpha, best_beta, best_gamma
    
    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.Dorset.BACKBONE))
        self.clip_model, _ = clip.load(
            self.cfg.MODEL.Dorset.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        class_names = self.data_manager.dataset.class_names

        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [
            prompt_template.format(class_name.replace("_", " "))
            for class_name in class_names
        ]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(prompts)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

        keys, values, domains, confidences = self.build_cache()

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.clip_model, self.text_features, keys, values, domains, confidences
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        if self.cfg.MODEL.Dorset.TUNING:
            self.alpha, self.beta, self.gamma = self.search_hp()
        else:
            self.alpha = self.cfg.MODEL.Dorset.ALPHA
            self.beta = self.cfg.MODEL.Dorset.BETA
            self.gamma = self.cfg.MODEL.Dorset.GAMMA

    def parse_batch_cache(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label
    
    def model_inference(self, input_data):
        return self.model(input_data, self.alpha, self.beta, self.gamma)
