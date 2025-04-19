# Script to generate GPT-3 prompts given a set of category names
# Used for CuPL prompting strategy (refer Sec. 3.1 of paper)
# Adapted from: https://github.com/sarahpratt/CuPL/blob/main/generate_image_prompts.py

import os
import json
import argparse
import time
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from descriptions.prompts_helper import CUPL_PROMPTS as cupl_prompts
import openai
openai.organization = "Personal"
your_key = ""
client = openai.OpenAI(api_key=your_key)

from src.dataloader.imagenet import IN_CLASSNAMES
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
from src.dataloader.imagenet_sketch import IN_SKETCH_CLASSNAMES
from src.dataloader.imagenet_rendition import IN_R_CLASSNAMES

_CLASSNAMES = {
        'imagenet_1k': IN_CLASSNAMES,
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
		'imagenet_sketch': IN_SKETCH_CLASSNAMES,
		'imagenet_rendition': IN_R_CLASSNAMES,
        }

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset to download', type=str, default='cifar10')
parser.add_argument('--n', help='the number of descriptions per prompt', type=int, default=10)
args = parser.parse_args()

dataset = args.dataset
output_dir = './descriptions'
json_name = "CuPL_prompts_{}.json".format(dataset)
classname_lst = _CLASSNAMES[dataset]
string_classnames = [s.replace('_', ' ') for s in classname_lst]
category_list = string_classnames
all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']

# mechanism to restart download smartly
index_to_restart = 0
if(os.path.exists(os.path.join(output_dir, json_name))):
	with open(os.path.join(output_dir, json_name), 'r') as f:
		json_dict = json.load(f)
		for index, item in enumerate(json_dict.items()):
			class_name = item[0]
			if(class_name==string_classnames[index]):
				index_to_restart += 1
				all_responses[class_name] = item[1]

for ind, category in tqdm(enumerate(category_list)):
	if(ind < index_to_restart):
		continue
	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"
	prompts = cupl_prompts[args.dataset]
	if(args.dataset=='ucf101' or args.dataset=='country211'):
		prompts = [p.format(category) for p in prompts]
	else:
		prompts = [p.format(article, category) for p in prompts]
	all_result = []
	prompt_id = 0
	while(prompt_id < len(prompts)):
		curr_prompt = prompts[prompt_id]
		try:
			messages = [
							{"role": "system", "content": """You are a helpful assistant. 
															 Your answer should be self-contained in the given number of tokens. 
															 Your answer should be informative.
															 Your answer should not contain executes and emojis."""},
							{"role": "user", "content": curr_prompt}
						]
			# parameters taken directly from CuPL paper
			response = client.chat.completions.create(
				model='gpt-3.5-turbo', # 'gpt-3.5-turbo'
				messages=messages,
				max_tokens=55,
				temperature=.99,
				n=args.n)
			for choice in response.choices:
				result = choice.message.content
				if('![' in result or '])' in result or '](' in result):
					continue
				if(len(result) < 5):
					continue
				if(len(result) > 1 and result[0]=='?'):
					continue
				result = result.replace("\n", "")
				result = result.replace('\"', "")
				result = result.replace('\"', "")
				if result[-1] != '.':
					result = result + '.'
				all_result.append(result)
			# sleep to ensure no timeout error
			time.sleep(1)
			prompt_id += 1
		except openai.OpenAIError as e:
			# if we hit rate limit, retry for same prompt again
			print(f"OpenAI error: {e}")
			time.sleep(5)
			pass
	
	all_responses[category] = all_result
	# sleep to ensure no timeout error
	time.sleep(1)
	
	os.makedirs(os.path.join(output_dir), exist_ok=True)
	with open(os.path.join(output_dir, json_name), 'w') as f:
		json.dump(all_responses, f, indent=4)
