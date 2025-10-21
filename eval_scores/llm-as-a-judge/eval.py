import os
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import template
import openai

parser = argparse.ArgumentParser()

parser.add_argument("--result_root_dir", type=str, default="/root/Image2CaptionAttack/results")
parser.add_argument("--dataset-name", type = str, default = "COCO2017")
parser.add_argument("--blip-model", type=str, default="blip2-opt-2.7b")
parser.add_argument("--clip-model", type=str, default="ViT-16B")
parser.add_argument("--leaked-feature-layer", type=str, default="vit-base")
parser.add_argument("--out_result_dir", type=str, default="/root/Image2CaptionAttack/eval_scores/llm-as-a-judge/scores")
parser.add_argument("--skip-count", type = int, default = -1)
args = parser.parse_args()

result_root_dir = args.result_root_dir
dst_name = args.dataset_name
blip_model = args.blip_model
clip_model = args.clip_model
leaked_feature_layer = args.leaked_feature_layer
out_result_dir = args.out_result_dir
skip_count = args.skip_count

result_fn = os.path.join(
    result_root_dir,
    f"{dst_name}_{blip_model}_{clip_model}_{leaked_feature_layer}",
    "eval_results.jsonl",
)

out_result_fn = os.path.join(out_result_dir, f"eval_llm-as-a-judge_{dst_name}_{blip_model}_{clip_model}_{leaked_feature_layer}.jsonl")


client = OpenAI(
    api_key="YOUR_OWN_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 写一个函数，根据msg_content，分别解析Fluency、Coherence、Relevance的分数，以及解析最终分数，将四部分分数提取出来并返回
def parse_caption_scores(response):
    try:
        fluency_score = response.split("Score: ")[1].split("**")[0]
        coherence_score = response.split("Score: ")[2].split("**")[0]
        relevance_score = response.split("Score: ")[3].split("**")[0]
        # final_score = response.split("Final Score: ")[1].split("**")[0]
        # final_score = final_score.split(" (")[0]
        final_score = round(
            (float(fluency_score) + float(coherence_score) + float(relevance_score))
            / 3,
            2,
        )
    except:
        # print(response)
        # raise ValueError("Error parsing scores.")
        return None, None, None, None
    # fluency_score = float(fluency_score)
    # coherence_score = float(coherence_score)
    # relevance_score = float(relevance_score)
    # final_score = float(final_score)
    return (
        float(fluency_score),
        float(coherence_score),
        float(relevance_score),
        final_score,
    )


# 第二个函数，上面函数是认为模型输出的结果是理想结果，但实际可能出现偏差，这个函数是根据观察偏差结果之后给的纠正的函数，可能对一些有用
def parse_caption_scores_v2(response):
    try:
        fluency_score = response.split("# Fluency: ")[1].split("**")[0]
        coherence_score = response.split("# Coherence: ")[1].split("**")[0]
        relevance_score = response.split("# Relevance: ")[1].split("**")[0]
        fluency_score = float(fluency_score)
        coherence_score = float(coherence_score)
        relevance_score = float(relevance_score)
        final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
    except:
        return None, None, None, None
    return fluency_score, coherence_score, relevance_score, final_score


# 第三个函数，给第二个函数找补
def parse_caption_scores_v3(response):
    try:
        fluency_score = response.split("# Fluency: ")[1].split("\n")[0]
        coherence_score = response.split("# Coherence: ")[1].split("\n")[0]
        relevance_score = response.split("# Relevance: ")[1].split("\n")[0]
        fluency_score = float(fluency_score)
        coherence_score = float(coherence_score)
        relevance_score = float(relevance_score)
        final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
    except:
        return None, None, None, None
    return fluency_score, coherence_score, relevance_score, final_score


# 第四个函数，给第三个函数找补
def parse_caption_scores_v4(response):
    try:
        fluency_score = response.split("**Fluency: ")[1].split("**")[0]
        coherence_score = response.split("**Coherence: ")[1].split("**")[0]
        relevance_score = response.split("**Relevance: ")[1].split("**")[0]
        fluency_score = float(fluency_score)
        coherence_score = float(coherence_score)
        relevance_score = float(relevance_score)
        final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
    except:
        try:
            fluency_score = response.split("** Fluency: ")[1].split("**")[0]
            coherence_score = response.split("** Coherence: ")[1].split("**")[0]
            relevance_score = response.split("** Relevance: ")[1].split("**")[0]
            fluency_score = float(fluency_score)
            coherence_score = float(coherence_score)
            relevance_score = float(relevance_score)
            final_score = round(
                (fluency_score + coherence_score + relevance_score) / 3, 2
            )
        except:
            try:
                fluency_score = response.split("** Fluency: ")[1].split("\n")[0]
                coherence_score = response.split("** Coherence: ")[1].split("\n")[0]
                relevance_score = response.split("** Relevance: ")[1].split("\n")[0]
                fluency_score = float(fluency_score)
                coherence_score = float(coherence_score)
                relevance_score = float(relevance_score)
                final_score = round(
                    (fluency_score + coherence_score + relevance_score) / 3, 2
                )
            except:
                return None, None, None, None
    return fluency_score, coherence_score, relevance_score, final_score


def parse_caption_scores_v5(response):
    try:
        fluency_score = response.split("Final Score:")[1].split("**Fluency:** ")[1].split("\n")[0]
        coherence_score = response.split("Final Score:")[1].split("**Coherence:** ")[1].split("\n")[0]
        relevance_score = response.split("Final Score:")[1].split("**Relevance:** ")[1].split("\n")[0]
        fluency_score = float(fluency_score)
        coherence_score = float(coherence_score)
        relevance_score = float(relevance_score)
        final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
    except:
        
        try:
            fluency_score = response.split("**Score:**")[1].split("\n")[0]
            coherence_score = response.split("**Score:**")[2].split("\n")[0]
            relevance_score = response.split("**Score:**")[3].split("\n")[0]
            fluency_score = float(fluency_score)
            coherence_score = float(coherence_score)
            relevance_score = float(relevance_score)
            final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
        except:
            try:
                fluency_score = response.split("**Fluency:**")[1].split("\n")[0]
                coherence_score = response.split("**Coherence:**")[1].split("\n")[0]
                relevance_score = response.split("**Relevance:**")[1].split("\n")[0]
                fluency_score = float(fluency_score)
                coherence_score = float(coherence_score)
                relevance_score = float(relevance_score)
                final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
            except:
                try:
                    fluency_score = response.split("Fluency:\n**")[1].split("**")[0]
                    coherence_score = response.split("Coherence:\n**")[1].split("**")[0]
                    relevance_score = response.split("Relevance:\n**")[1].split("**")[0]
                    fluency_score = float(fluency_score)
                    coherence_score = float(coherence_score)
                    relevance_score = float(relevance_score)
                    final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
                except:
                    try:
                        fluency_score = response.split("- Fluency Score: ")[1].split("\n")[0]
                        coherence_score = response.split("- Coherence Score: ")[1].split("\n")[0]
                        relevance_score = response.split("- Relevance Score: ")[1].split("\n")[0]
                        fluency_score = float(fluency_score)
                        coherence_score = float(coherence_score)
                        relevance_score = float(relevance_score)
                        final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
                    except:
                        try:
                            fluency_score = response.split("**Fluency**")[1].split("Score: ")[1].split("\n")[0]
                            coherence_score = response.split("**Coherence**")[1].split("Score: ")[1].split("\n")[0]
                            relevance_score = response.split("**Relevance**")[1].split("Score: ")[1].split("\n")[0]
                            fluency_score = float(fluency_score)
                            coherence_score = float(coherence_score)
                            relevance_score = float(relevance_score)
                            final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
                        except:
                            try:
                                fluency_score = response.split("Fluency")[1].split("**Score:** ")[1].split("\n")[0]
                                coherence_score = response.split("Coherence")[1].split("**Score:** ")[1].split("\n")[0]
                                relevance_score = response.split("Relevance")[1].split("**Score:** ")[1].split("\n")[0]
                                fluency_score = float(fluency_score)
                                coherence_score = float(coherence_score)
                                relevance_score = float(relevance_score)
                                final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
                            except:
                                try:
                                    fluency_score = response.split("Final Score Calculation:")[1].split("- Fluency: ")[1].split("\n")[0]
                                    coherence_score = response.split("Final Score Calculation:")[1].split("- Coherence: ")[1].split("\n")[0]
                                    relevance_score = response.split("Final Score Calculation:")[1].split("- Relevance: ")[1].split("\n")[0]
                                    fluency_score = float(fluency_score)
                                    coherence_score = float(coherence_score)
                                    relevance_score = float(relevance_score)
                                    final_score = round((fluency_score + coherence_score + relevance_score) / 3, 2)
                                except:
                                    return None, None, None, None
    return fluency_score, coherence_score, relevance_score, final_score

def clean_generated_caption(caption):
    caption = caption.replace("\ufffd", "")
    caption = caption.replace("\u200b", "")
    caption = caption.replace("\u200d", "")
    caption = caption.replace("\u200c", "")
    caption = caption.replace("\u200e", "")
    caption = caption.strip(" ")
    caption = caption.replace("dfx", "")
    caption = caption.replace("  ", " ")
    return caption


with open(result_fn, "r", encoding = 'utf-8', errors = 'ignore') as fin, open(out_result_fn, 'w', encoding = 'utf-8', errors = 'ignore') as fout:
    error_idx_list = []
    data = fin.readlines()
    for idx, line in tqdm(enumerate(data), total = len(data), desc = "Process"):
        if idx < skip_count:
            continue
        line = line.strip('\n').strip('\r')
        line = json.loads(line)
        generated_caption = line["generated"]
        generated_caption = clean_generated_caption(generated_caption)
        ground_truth_captions = line["ground_truth"]
        prompt = template.get_prompt_from_template(generated_caption, ground_truth_captions, template.prompt_template_v1)
        try:
            completion = client.chat.completions.create(
                model="qwen-max-2025-01-25",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        except openai.BadRequestError as e:
            print(e)
            error_idx_list.append(idx)
            line["evaluation"] = "Error"
            line["fluency_score"] = None
            line["coherence_score"] = None
            line["relevance_score"] = None
            line["final_score"] = None
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")
            continue
        msg_content = completion.choices[0].message.content

        line["evaluation"] = msg_content
        fluency_score, coherence_score, relevance_score, final_score = (
            parse_caption_scores(msg_content)
        )
        if fluency_score is None:
            fluency_score, coherence_score, relevance_score, final_score = (
                parse_caption_scores_v2(msg_content)
            )
        if fluency_score is None:
            fluency_score, coherence_score, relevance_score, final_score = (
                parse_caption_scores_v3(msg_content)
            )
        if fluency_score is None:
            fluency_score, coherence_score, relevance_score, final_score = (
                parse_caption_scores_v4(msg_content)
            )
        if fluency_score is None:
            fluency_score, coherence_score, relevance_score, final_score = (
                parse_caption_scores_v5(msg_content)
            )
        line["fluency_score"] = fluency_score
        line["coherence_score"] = coherence_score
        line["relevance_score"] = relevance_score
        line["final_score"] = final_score
        # eval_results.append(item_result)
        if fluency_score is None:
            print("Error processing item ", idx)
            error_idx_list.append(idx)

        fout.write(json.dumps(line, ensure_ascii = False) + "\n")
    with open(
        os.path.join(
            out_result_dir,
            "error_idx_list_{}_{}_{}_{}_{}.txt".format(
                dst_name, blip_model, clip_model, leaked_feature_layer
            ),
        ),
        "w",
    ) as f:
        json.dump(error_idx_list, f)
