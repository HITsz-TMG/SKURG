import sys

sys.path.append('../')
import json
from tqdm import tqdm
from ofa import OFATokenizer
import pickle
import base64
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO

file_path = './MMQA/dataset/webqa/WebQA_test.json'
# path to the webQA test set

tokenizer = OFATokenizer.from_pretrained('./OFA')

tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<title>', '</title>', 'ROW', '[b_ans]', '[e_ans]', '[b_source]', '[e_source]']})

data = json.load(open(file_path, 'r'))


def dict_slice(adict, start):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:]:
        dict_slice[k] = adict[k]
    return dict_slice


mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([lambda image: image.convert("RGB"),
                                             transforms.Resize((resolution, resolution),
                                                               interpolation=Image.BICUBIC),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)
                                             ])
with open("./MMQA/dataset/webqa/imgs.lineidx", "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
# file of WebQA imgs feat

fp = open("./MMQA/dataset/webqa/imgs.tsv", "r")
# file of WebQA imgs feat

img_feats = {}

pbar = tqdm(total=len(data))
for key, value in data.items():
    pbar.update(1)
    img_Facts = value['img_Facts']
    for img in img_Facts:
        image_id = img['image_id']
        fp.seek(lineidx[int(image_id) % 10000000])
        imgid, img_base64 = fp.readline().strip().split('\t')
        img_feat = Image.open(BytesIO(base64.b64decode(img_base64)))
        patch_img = patch_resize_transform(img_feat)
        title = img['caption']
        title_tokens = tokenizer.tokenize(' ' + title)
        title_tokens = tokenizer.convert_tokens_to_ids(title_tokens)
        tmp_feats = {'ids': title_tokens, 'feats': patch_img}
        save_path = os.path.join('./MMQA/dataset/webqa/images_ofa_feats',
                                 str(image_id) + '.pkl')
        pickle.dump(tmp_feats, open(save_path, 'wb'))
