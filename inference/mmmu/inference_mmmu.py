import torch
import os
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from argparse import ArgumentParser

from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_llava_engine_df, llava_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in samples:
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_7b_val.json', help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--use-attpack', default=False, action='store_true', help='whether to use attpack')
    parser.add_argument('--svd_mode', type=str, default='t_hd', help='svd mode (t_hd or t_h_d)')
    parser.add_argument('--svd_T', type=int, default=500, help='svd period')
    parser.add_argument('--rank_k', type=int, default=64, help='rank for key')
    parser.add_argument('--rank_v', type=int, default=0, help='rank for value')
    parser.add_argument('--output-path', type=str, required=True, help='the path to save the output json file')

    pargs = parser.parse_args()

    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'
        pca_prompt_data = ''
        pca_img_dir = ''
        output_path = ''

    # %%
    args = InferenceArgs()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    pargs.config = load_yaml(pargs.config_path)
    for key, value in pargs.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            pargs.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(pargs.data_path, subject, split=pargs.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base,
                                                                           model_name, args.load_8bit, args.load_4bit,
                                                                           device_map="auto", device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    if not pargs.use_attpack:
        pargs.rank_k = 0
        pargs.rank_v = 0

    model.config.svd_mode = pargs.svd_mode
    model.config.svd_T = pargs.svd_T
    model.config.rank_k = pargs.rank_k
    model.config.rank_v = pargs.rank_v
    model.model.init_cache()

    out_samples = []
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, pargs.config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], image_processor).to(device)

        # run ex
        out_sample = run_model(args, [sample], model, call_model_engine, tokenizer, processor)
        out_samples.append(out_sample)
        print(out_sample)

        model.model.init_cache()

    save_json(pargs.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

