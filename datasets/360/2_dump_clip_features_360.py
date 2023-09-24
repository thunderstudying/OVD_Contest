# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import numpy as np
import itertools
import sys


def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'


def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res


multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]


if __name__ == '__main__':
    phase = 'test'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ann', default=f'1_{phase}_name.npy')  # output from 1_zh2en.py
    parser.add_argument('--ann', default=f'5_{phase}_cat_info.json')  # output from 5_csv2json.py
    # parser.add_argument('--ann', default=f'5_tt.json')  # output from 5_csv2json.py
    # parser.add_argument('--out_path', default=f'2_360_{phase}_clip_a+cname.npy')
    parser.add_argument('--out_path', default=f'2_360_{phase}_clip_a+cname_new_mutiTemp.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--avg_synonyms', action='store_true', default=True)
    parser.add_argument('--use_wn_name', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    if 'npy' in args.ann:
        data = np.load(args.ann)
        cat_names = [x for x in data]
    else:
        data = json.load(open(args.ann))
        cat_names = [item['en_name'] for item in data]
    # if 'synonyms' in data[0].keys():
    #     synonyms = [[xx for xx in item['synonyms'].split('_') if xx != ''] for item in data]
    #     for ind, it in enumerate(synonyms):
    #         it.append(data[ind]['en_name'])
    # else:
    synonyms = []
    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        # sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
        sentences_synonyms = [[
            template.format(processed_name(category, rm_dot=True), article=article(category))
            for template in multiple_templates
        ] for category in cat_names]
        sentences_synonyms = [[
            "This is " + text if text.startswith("a") or text.startswith("the") else text
            for text in sentence
        ] for sentence in sentences_synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
            for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
            for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
        sum(len(x) for x in sentences_synonyms))
    if args.model == 'clip':
        import clip
        print('Loading CLIP')
        model, preprocess = clip.load(args.clip_model, device=device)
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        text = clip.tokenize(sentences).to(device)
        with torch.no_grad():
            if len(text) > 10000:
                text_features = torch.cat([
                    model.encode_text(text[:len(text) // 2]),
                    model.encode_text(text[len(text) // 2:])],
                    dim=0)
            else:
                text_features = model.encode_text(text)
        print('text_features.shape', text_features.shape)
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.cpu().numpy()
    elif args.model in ['bert', 'roberta']:
        from transformers import AutoTokenizer, AutoModel
        if args.model == 'bert':
            model_name = 'bert-large-uncased'
        if args.model == 'roberta':
            model_name = 'roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        inputs = tokenizer(sentences, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_outputs = model(**inputs)
            outputs = model_outputs.pooler_output
        text_features = outputs.detach().cpu()
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.numpy()
        print('text_features.shape', text_features.shape)
    else:
        assert 0, args.model
    if args.out_path != '':
        print('saveing to', args.out_path)
        np.save(open(args.out_path, 'wb'), text_features)
    # import pdb; pdb.set_trace()
