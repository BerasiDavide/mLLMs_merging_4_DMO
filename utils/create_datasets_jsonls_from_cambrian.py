import os
import json

from tqdm import tqdm
from collections import defaultdict

tag2name = {
    # OCR
    ('dvqa_2325k.json', 'dvqa'): 'dvqa_2325k',
    ('synthdog_500k_modified.json', 'synthdog'): 'synthdog_modified_500k',
    ('arxivqa_100k.json', 'arxivqa'): 'arxivqa_100k',
    ('sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', 'ocr_vqa'): 'ocr_vqa_80k',
    ('screenqa_79k.json', 'screen_qa'): 'screenqa_79k',
    ('docvqa_39k.json', 'docvqa'): 'docvqa_39k',
    ('idefics375k.json', 'hfdata/robut_wtq'): 'robut_wtq_38k',
    ('chartqa_28k.json', 'chartqa'): 'chartqa_28k',
    ('idefics375k.json', 'hfdata/iconqa'): 'iconqa_27k',
    ('idefics375k.json', 'hfdata/chart2text'): 'chart2text_26k',
    ('idefics375k.json', 'hfdata/tabmwp'): 'tabmwp_23k',
    ('sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', 'textvqa'): 'textCaps_22k', # Not 100% sure, but probably this is textCaps 22k
    ('clean_llava_instruct_150k_llavar_20k.json', 'llavar'): 'llavar_20k',
    ('idefics375k.json', 'hfdata/st_vqa'): 'st_vqa_17k',
    ('ai2d_15k.json', 'ai2d'): 'ai2d_15k',
    ('idefics375k.json', 'hfdata/rendered_text'): 'rendered_text_10k',
    ('idefics375k.json', 'hfdata/vistext'): 'vistext_9k',
    ('idefics375k.json', 'hfdata/finqa'): 'finqa_6k',
    ('idefics375k.json', 'hfdata/infographic_vqa'): 'infographic_vqa_2k',
    ('idefics375k.json', 'hfdata/tat_qa'): 'tat_qa_2k',
    ('idefics375k.json', 'hfdata/hitab'): 'hitab_2k',

    # GENERAL
    ('allava-laion-500k.json', 'allava'): 'allava_laion_500k',
    ('allava-vflan-200k.json', 'allava'): 'allava_vflan_200k',
    ('q-instruct_200k.json', 'Q-Instruct-DB'): 'qinstruct_200k',
    ('qalign_200k.json', 'Q-Instruct-DB'): 'qalign_200k',
    ('lnqa_302k.json', 'lnqa'): 'lnqa_302k',
    ('lvis_instruct4v_220k.json', 'coco'): 'lvis_instruct4v_220k',
    ('sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', 'vg'): 'vg_86k',    # 86k
    ('gpt77k.json', 'chartqa'): 'gpt_77k',
    ('gpt77k.json', 'tallyqa'): 'gpt_77k',
    ('gpt77k.json', 'clevr'): 'gpt_77k',
    ('gpt77k.json', 'dvqa'): 'gpt_77k',
    ('gpt77k.json', 'Q-Instruct-DB'): 'gpt_77k',
    ('gpt77k.json', 'gpt4v-dataset'): 'gpt_77k',
    ('gpt77k.json', 'gqa'): 'gpt_77k',
    ('gpt77k.json', 'coco'): 'gpt_77k',
    ('gpt77k.json', 'screen_qa'): 'gpt_77k',
    ('gpt77k.json', 'ai2d'): 'gpt_77k',
    ('gpt77k.json', 'oodvqa'): 'gpt_77k',
    ('gpt77k.json', 'docvqa'): 'gpt_77k',
    ('gpt77k.json', 'scienceqa'): 'gpt_77k',
    ('gpt77k.json', 'mathvision'): 'gpt_77k',
    ('gpt77k.json', 'vizwiz'): 'gpt_77k',
    ('gpt77k.json', 'geo170k'): 'gpt_77k',
    ('gpt77k.json', 'arxivqa'): 'gpt_77k',
    ('gpt77k.json', 'synthdog'): 'gpt_77k',
    ('gpt77k.json', 'llava'): 'gpt_77k',
    ('gpt77k.json', 'sam'): 'gpt_77k',
    ('gpt77k.json', 'wikiart'): 'gpt_77k',
    ('gpt77k.json', 'share_textvqa'): 'gpt_77k',
    ('gpt77k.json', 'web-celebrity'): 'gpt_77k',
    ('gpt77k.json', 'web-landmark'): 'gpt_77k',
    ('gpt77k.json', 'vg'): 'gpt_77k',
    ('gpt77k.json', 'ocr_vqa'): 'gpt_77k',
    ('gpt77k.json', 'textvqa'): 'gpt_77k',
    ('gpt77k.json', 'llavar'): 'gpt_77k',
    ('gpt77k.json', 'allava'): 'gpt_77k',
    ('gpt77k.json', 'idk'): 'gpt_77k',
    ('gpt77k.json', 'design2code'): 'gpt_77k',
    ('sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', 'gqa'): 'gqa_72k',   #72k
    ('alfworldgpt_45k.json', 'alfworld'): 'alfworldgpt_45k',
    ('vizwiz_20k.json', 'vizwiz'): 'vizwiz_20k',
    ('idefics375k.json', 'hfdata/visual7w'): 'visual7w_14k',
    ('laion_gpt4v_11k.json', 'gpt4v-dataset'): 'gpt4v_11k',
    ('idk_11k.json', 'idk'): 'idk_11k',
    ('idefics375k.json', 'hfdata/hateful_memes'): 'hateful_memes_8k',
    ('oodvqa_8k.json', 'oodvqa'): 'oodvqa_8k',
    ('sketchyvqa_8k.json', 'oodvqa'): 'sketchyvqa_8k',
    ('idefics375k.json', 'hfdata/visualmrc'): 'visualmrc_3k',

    # LANGUAGE
    ('orca_994k.json', None): 'orca_994k',
    ('mathinstruct_262k.json', None): 'mathinstruct_262k',
    ('orca_math_200k.json', None): 'orca_math_200k',
    ('wizardlm_143k.json', None): 'wizardlm_143k',
    ('code_feedback_66k.json', None): 'code_feedback_66k', # OpenCodeInterpreter
    ('dolly_15k.json', None): 'dolly_15k',

    # COUNTING
    ('clevr_700k.json', 'clevr'): 'clevr_700k',
    ('tallyqa_250k.json', 'tallyqa'): 'tallyqa_250k',

    # CODE
    ('websight_800k.json', 'websight'): 'websight_800k',
    ('idefics375k.json', 'hfdata/datikz'): 'datikz_47k',
    ('design2code_0k.json', 'design2code'): 'design2code_0k',

    # MATH
    ('geo170k.json', 'geo170k'): 'geo_170k',
    ('idefics375k.json', 'hfdata/raven'): 'raven_42k',
    ('idefics375k.json', 'hfdata/geomverse'): 'geomverse_9k',
    ('mathvision_3k.json', 'mathvision'): 'mathvision_3k',
    ('idefics375k.json', 'hfdata/intergps'): 'intergps_1k',
    ('idefics375k.json', 'hfdata/tqa'): 'tqa_1k',

    # SCIENCE
    ('filtered_data_engine_161k.json', 'data_engine'): 'filtered_data_engine_161k',
    ('pathvqa_32k.json', 'pathvqa'): 'pathvqa_32k',
    ('scienceqa_12k.json', 'scienceqa'): 'scienceqa_12k',
    ('scienceqa_12k.json', None): 'scienceqa_12k',
}

BADLY_FORMATTED_SAMPLES_IDS = set([650722, 8373676])  # These samples have badly formatted conversations, so we will skip them


# Function to read JSONL files
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def get_tag(sample):
    # Get tag =  json_source, image_folder
    source = sample['source']
    image_path = sample.get('image')

    if image_path:
        parts = image_path.split('/')
        folder = parts[0] if parts[0] != 'hfdata' else 'hfdata/' + parts[1]
    else:
        folder = None

    tag = (source, folder)
    return tag


def main(args):

    cambrian_jsonl = os.path.join(args.cambrian_path, 'jsons/Cambrian10M.jsonl')
    def get_jsonl_output_path(name):
        return os.path.join(args.output_path, f'{name}.jsonl')


    # Divide samples by dataset
    name2samples_idx = defaultdict(list)
    for idx, sample in tqdm(enumerate(read_jsonl(cambrian_jsonl)), total=10_000_000, desc="Dividing samples by dataset"):
        if idx in BADLY_FORMATTED_SAMPLES_IDS:
            continue
        tag = get_tag(sample)
        name = tag2name.get(tag, None)
        if name is not None:
            name2samples_idx[name].append(idx)


    ## Create a jsonl file per dataset name
    skipped_names = set()
    for name in name2samples_idx.keys():
        file_path = get_jsonl_output_path(name)
        if os.path.exists(file_path):
            if name in args.overwrite:
                os.remove(file_path)
                print(f"Removed existing file: {file_path}")
            else:
                print(f"File already exists, skipping: {name}")
                skipped_names.add(name)
                continue
    tag2name_notskipped = {tag: name for tag, name in tag2name.items() if name not in skipped_names}
    
    # Pre-open file handles for all categories
    file_handles = {
        name: open(get_jsonl_output_path(name), 'a')
        for name in tag2name_notskipped.values()
    }

    name2count = {name: 0 for name in tag2name_notskipped.values()}
    skipped_count = 0
    try:
        i = 0
        flush_every = 300_000
        for idx, x in tqdm(enumerate(read_jsonl(cambrian_jsonl)), total=10_000_000, desc="Saving samples to files"):

            tag = get_tag(x)
            name = tag2name_notskipped.get(tag)
            if name is None:
                skipped_count += 1
                continue
            
            # Write to the appropriate file
            json.dump(x, file_handles[name])
            file_handles[name].write('\n')
            name2count[name] += 1

            i += 1
            if i % flush_every == 0:
                for f in file_handles.values():
                    f.flush()
    finally:
        for f in file_handles.values():
            f.close()

    for name, count in name2count.items():
        print(f"Total samples for {name}: {count}")
    print(f"Skipped samples: {skipped_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split Cambrian-10M into datasets.")
    parser.add_argument('--cambrian-path', type=str, default='LLaMA-Factory/data/Cambrian-10M', help='Path to the Cambrian-10M folder.')
    parser.add_argument('--output_path', type=str, default='LLaMA-Factory/data/jsonl_data', help='Path to save the output JSONL files.')
    parser.add_argument('--overwrite', type=list, default=[], help='List of dataset names to overwrite.')
    args = parser.parse_args()
    main(args)
