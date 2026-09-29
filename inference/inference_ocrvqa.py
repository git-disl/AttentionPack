import argparse

import matplotlib

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

matplotlib.use('TkAgg')

import requests
from PIL import Image
from io import BytesIO

from datasets import load_dataset
import torch
import json
from tqdm import tqdm

import re

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
                "dont": "don't", "hadnt": "hadn't", \
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
                "hed've": "he'd've", \
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                "Id've": "I'd've", "I'dve": "I'd've", \
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
                "itll": "it'll", "let's": "let's", \
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
                "mightve": "might've", \
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
                "oughtnt": "oughtn't", \
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                "shed've": "she'd've", "she'dve": "she'd've", \
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
                "shouldn'tve": "shouldn't've", \
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
                "somebodyll": "somebody'll", \
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                "someone'dve": "someone'd've", \
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                "somethingd've": "something'd've", \
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                "thered": "there'd", "thered've": "there'd've", \
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                "theyd've": "they'd've", \
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
                "wasnt": "wasn't", \
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
                "whatre": "what're", \
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
                "whereve": "where've", \
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
                "whove": "who've", "whyll": "why'll", \
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
                "wouldnt've": "wouldn't've", \
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                "yall'd've": "y'all'd've", \
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
                "you'dve": "you'd've", \
                "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap = {'none': '0',
             'zero': '0',
             'one': '1',
             'two': '2',
             'three': '3',
             'four': '4',
             'five': '5',
             'six': '6',
             'seven': '7',
             'eight': '8',
             'nine': '9',
             'ten': '10'
             }
articles = ['a',
            'an',
            'the'
            ]

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",
                              outText,
                              re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def clean_text(pred):
    pred = pred.replace('\n', ' ')
    pred = pred.replace('\t', ' ')
    pred = pred.strip()
    pred = processPunctuation(pred)
    pred = processDigitArticle(pred)

    return pred


# %%
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# %%
# generate test aokvqa dataset

# %%
# generate test nocap dataset

dataset_ocrvqa_images = []
dataset_ocrvqa_prompts = []
dataset_ocrvqa_labels = []

dataset = load_dataset("howard-hou/OCR-VQA")['test']  # load from huggingface dataset

N_examples = 500  # sample first 500 examples for fast testing
cur_num = 0

for i in dataset:
    cur_num += 1
    if cur_num > N_examples:
        break

    for question, answer in zip(i['questions'], i['answers']):
        dataset_ocrvqa_images.append(i['image'])
        dataset_ocrvqa_prompts.append(question)
        dataset_ocrvqa_labels.append([answer])

print(dataset_ocrvqa_prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--use-attpack', default=False, action='store_true', help='whether to use attpack')
    parser.add_argument('--svd_mode', type=str, default='t_hd', help='svd mode (t_hd or t_h_d)')
    parser.add_argument('--svd_T', type=int, default=500, help='svd period')
    parser.add_argument('--rank_k', type=int, default=64, help='rank for key')
    parser.add_argument('--rank_v', type=int, default=64, help='rank for value')
    parser.add_argument('--bs', type=int, default=1, help='batch_size')
    # output path
    parser.add_argument('--output-path', type=str, required=True, help='the path to save the output json file')

    pargs = parser.parse_args()

    # %%
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


    # %%
    args = InferenceArgs()

    # %%
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit,
                                                                           device=args.device)

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


    # %%
    def inference(prompts, images):
        outputs = []
        bs = pargs.bs
        # for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
        for i in tqdm(range(0, len(prompts), bs)):
            cur_prompts = prompts[i * bs: (i + 1) * bs]
            cur_images = images[i * bs: (i + 1) * bs]

            input_ids_list = []
            image_tensor_list = []
            for j in range(len(cur_prompts)):
                prompt = cur_prompts[j]
                image = cur_images[j]
                image = image.convert('RGB')
                image_tensor = process_images([image], image_processor, args)
                conv = conv_templates[args.conv_mode].copy()
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                inp = prompt

                if image is not None:
                    # first message
                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp  # False
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                    conv.append_message(conv.roles[0], inp)
                    # image = None
                else:
                    # later messages
                    conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()  # + "In the image, there is a kitchen with a refrigerator, a sink, and a cup on the counter. The cup is placed on the counter, and there is a backpack nearby. The room appears to be empty, and there are no other objects or people in the scene.\n\nTo bring a clean cup to the person, the robot should first look for a cup. Since the cup is already on the counter, the robot can proceed to pick up the cup. After picking up the cup, the robot should then put the cup into the sink to clean it. Finally, the robot can return the clean cup to the person.\n\nBased on the image, the correct sequence of actions for the robot is (A) Look for a cup, (B) Pick up a cup, and (C) Put the cup into the sink."

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                    0).cuda()
                input_ids_list.append(input_ids)
                image_tensor_list.append(image_tensor)

            max_len = max([ii.shape[-1] for ii in input_ids_list])
            for j in range(len(input_ids_list)):
                input_ids_list[j] = torch.cat([torch.zeros(1, max_len - input_ids_list[j].shape[-1],
                                                           dtype=input_ids_list[j].dtype,
                                                           device=input_ids_list[j].device), input_ids_list[j]], dim=-1)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            input_ids = torch.cat(input_ids_list, dim=0)
            image_tensor = torch.cat(image_tensor_list, dim=0)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=None,
                    do_sample=False,
                    max_new_tokens=50,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            output = tokenizer.batch_decode(output_ids['sequences'][:, input_ids.shape[1]:], skip_spectial_tokens=True)
            output = [o.strip().replace("</s>", "").replace("<unk>", "") for o in output]
            outputs.extend(output)
            print(output)

            model.model.init_cache()

        return outputs


    ocrvqa_inference_outputs = inference(dataset_ocrvqa_prompts, dataset_ocrvqa_images)

    from pycocoevalcap.rouge.rouge import Rouge


    def get_cider(labels, predictions):
        scorer = Rouge()
        labels_dict = {}
        predictions_dict = {}

        for i in range(len(labels)):
            labels_dict[i] = [clean_text(t) for t in labels[i]]
            predictions_dict[i] = [clean_text(predictions[i])]

        (score, scores) = scorer.compute_score(labels_dict, predictions_dict)
        return score, scores


    score, scores = get_cider(dataset_ocrvqa_labels, ocrvqa_inference_outputs)
    output_path = pargs.output_path

    with open(output_path, "w") as f:
        # json dumps
        json.dump({"avg": str(score), "scores": scores.tolist(), "output": ocrvqa_inference_outputs}, f, indent=4)
