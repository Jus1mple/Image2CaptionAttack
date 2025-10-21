# Description: 通过Qwen模型生成图像描述
from openai import OpenAI
import openai
import os
import base64
import pyarrow.parquet as pq
from tqdm import tqdm
import json
import random

# 从环境变量读取API密钥，如果未设置则抛出错误
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError(
        "DASHSCOPE_API_KEY environment variable not set. "
        "Please set it with your Aliyun DashScope API key."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
model_name = "qwen-vl-plus-2025-01-02"

# 从环境变量读取数据集路径，如果未设置则使用默认值
parquet_root_dir = os.getenv(
    "IMAGENET_PARQUET_DIR",
    "./datasets/imagenet-1k-256x256/data"
)

parquet_files = os.listdir(parquet_root_dir)
parquet_files = [f for f in parquet_files if f.endswith(".parquet")]
# parquet_files = [os.path.join(parquet_root_dir, f) for f in parquet_files]

# parquet_files = [
#     "train-00010-of-00040.parquet" # 测试集随机选2000条 训练集随机选10000条
# ]


for pq_file in tqdm(parquet_files, total = len(parquet_files), desc = "FILE"):
    fn = pq_file.split(".")[0]
    print(fn)
    if (
        fn == "train-00010-of-00040"
        or "val" in fn
        or "test" in fn
        or fn
        in [
            # "train-00000-of-00040",
            # "train-00001-of-00040",
            "train-00004-of-00040",
            "train-00005-of-00040",
            "train-00006-of-00040",
            "train-00008-of-00040",
            "train-00010-of-00040",
            "train-00011-of-00040",
            "train-00012-of-00040",
            "train-00013-of-00040",
            "train-00014-of-00040",
            "train-00015-of-00040",
            "train-00016-of-00040",
            "train-00017-of-00040",
            "train-00018-of-00040",
            "train-00020-of-00040",
        ]
    ):
        continue
    pq_file = os.path.join(parquet_root_dir, pq_file)
    table = pq.read_table(pq_file)
    df = table.to_pydict() # {"image" : "bytes", "path"; "label"}
    df_cnt = len(df["image"])
    random_cnt = 300
    random_choices = random.sample(range(df_cnt), random_cnt)
    print("Number of images: ", df_cnt)
    annotations = []
    out_fn = os.path.join(parquet_root_dir, f"{fn}_annotations.jsonl")
    fout = open(out_fn, "w", encoding = "utf-8")
    for i in tqdm(random_choices, total=random_cnt, desc="IMAGE"):
        image = df["image"][i]
        img_bytes = image["bytes"]
        img_path = image["path"]
        img_label = df["label"][i]
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        while True:
            try:
                completion = client.chat.completions.create(
                    model= model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Summarize and describe this image with no more than 20 words. JUST GIVE ME THE FINAL ANSWER.",
                                },
                            ],
                        }
                    ],
                )
                break
            except openai.BadRequestError as e:
                print("Error: ", e)
                break
            except:
                model_name = "qwen-vl-plus-latest"
                continue
        caption = completion.choices[0].message.content
        fout.write(json.dumps(
            {
                "image_path" : img_path,
                "image_label" : img_label,
                "image_caption" : caption
            }
        ) + "\n")
    fout.close()

# #  base 64 编码格式
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# base64_image01 = encode_image(
#     "/root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test/3689727848_b53f931130.jpg"
# )

# base64_image02 = encode_image(
#     "/root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test/161669933_3e7d8c7e2c.jpg"
# )


# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
#     api_key="sk-ef4089b13df2439689a5c007729238ce",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# completion = client.chat.completions.create(
#     model="qwen-vl-plus-2025-01-02",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 # {
#                 #     "type": "image_url",
#                 #     # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
#                 #     # PNG图像：  f"data:image/png;base64,{base64_image}"
#                 #     # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
#                 #     # WEBP图像： f"data:image/webp;base64,{base64_image}"
#                 #     "image_url": {"url": f"data:image/jpeg;base64,{base64_image01}"},
#                 # },
#                 # {
#                 #     "type": "image_url",
#                 #     "image_url": {"url": f"data:image/jpeg;base64,{base64_image02}"},
#                 # },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image01}"},
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image02}"},
#                 },
#                 {
#                     "type": "text",
#                     "text": "Individually Summarize and Describe these images image with no more than 20 words. JUST GIVE ME THE FINAL ANSWER.",
#                 },
#             ],
#         }
#     ],
# )
# print(completion)
# print(completion.choices)
# print(completion.choices[0].message.content)
# print(type(completion.choices[0].message.content))
