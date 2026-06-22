import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision import transforms
import pdb
import json
import random
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def img_normalize_train(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def img_normalize_val(img, scale=256/224, input_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

class PIAD(Dataset):
    def __init__(self, run_type, setting_type, point_path, img_path, text_hk_path, text_ok_path, pair=2, img_size=(224, 224)):
        super().__init__()

        self.run_type = run_type
        self.p_path = point_path
        self.i_path = img_path
        self.text_hk_path = text_hk_path
        self.text_ok_path = text_ok_path
        self.pair_num = pair
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab', 'carry', 'ride',
                        'clean', 'play', 'beat', 'speak', 'pull']  # 24
        '''
        Unseen affordance
        '''#41

        if setting_type == 'Unseen_aff':
            number_dict = {'Bag': 0, 'Microphone': 0, 'Toothbrush': 0, 'TrashCan': 0, 'Bicycle': 0,
                           'Guitar': 0, 'Glasses': 0, 'Hat': 0, 'Microwave':0, 'Door':0, 'Scissors': 0, 'Bowl': 0,
                           'Baseballbat': 0, 'Mop': 0, 'Dishwasher': 0, 'Bed': 0, 'Keyboard': 0, 'Clock': 0, 'Vase': 0, 'Knife': 0,
                            'Hammer': 0, 'Refrigerator': 0, 'Chair': 0, 'Umbrella': 0, 'Bucket': 0,
                           'Display': 0, 'Earphone': 0, 'Motorcycle': 0, 'StorageFurniture': 0, 'Fork': 0, 'Broom': 0, 'Skateboard': 0,
                           'Tennisracket': 0, 'Laptop': 0, 'Table':0, 'Bottle': 0, 'Faucet': 0, 'Kettle': 0, 'Surfboard': 0, 'Mug': 0,
                            'Spoon': 0
                            }
        '''
        Unseen object
        '''  # 32

        if setting_type == 'Unseen_obj':
            number_dict = {'Bag': 0, 'Microphone': 0, 'Toothbrush': 0, 'TrashCan': 0, 'Bicycle': 0,
                           'Guitar': 0, 'Glasses': 0, 'Hat': 0, 'Microwave':0, 'Backpack': 0, 'Door':0,  'Bowl': 0,
                            'Dishwasher': 0, 'Bed': 0, 'Keyboard': 0,  'Vase': 0, 'Knife': 0,
                           'Suitcase': 0, 'Hammer': 0,  'Chair': 0, 'Umbrella': 0,
                           'Display': 0, 'Earphone': 0, 'StorageFurniture': 0, 'Broom': 0, 
                           'Tennisracket': 0,  'Table':0, 'Bottle': 0, 'Faucet': 0,  'Surfboard': 0, 'Mug': 0,
                            'Spoon': 0
                            }

        '''
        Seen
        '''  # 43

        if setting_type == 'Seen':
            number_dict = {'Bag': 0, 'Microphone': 0, 'Toothbrush': 0, 'TrashCan': 0, 'Bicycle': 0,
                           'Guitar': 0, 'Glasses': 0, 'Hat': 0, 'Microwave':0, 'Backpack': 0, 'Door':0, 'Scissors': 0, 'Bowl': 0,
                           'Baseballbat': 0, 'Mop': 0, 'Dishwasher': 0, 'Bed': 0, 'Keyboard': 0, 'Clock': 0, 'Vase': 0, 'Knife': 0,
                           'Suitcase': 0, 'Hammer': 0, 'Refrigerator': 0, 'Chair': 0, 'Umbrella': 0, 'Bucket': 0,
                           'Display': 0, 'Earphone': 0, 'Motorcycle': 0, 'StorageFurniture': 0, 'Fork': 0, 'Broom': 0, 'Skateboard': 0,
                           'Tennisracket': 0, 'Laptop': 0, 'Table':0, 'Bottle': 0, 'Faucet': 0, 'Kettle': 0, 'Surfboard': 0, 'Mug': 0,
                            'Spoon': 0 
                           }

            aff_number_dict = {'grasp': 0, 'contain': 0, 'lift': 0, 'open': 0, 
                        'lay': 0, 'sit': 0, 'support': 0, 'wrapgrasp': 0, 'pour': 0, 'move': 0, 'display': 0,
                        'push': 0, 'listen': 0, 'wear': 0, 'press': 0, 'cut': 0, 'stab': 0, 'carry': 0, 'ride': 0,
                        'clean': 0, 'play': 0, 'beat': 0, 'speak': 0, 'pull': 0 
                           }        

        self.img_files = self.read_file(self.i_path)
        self.text_human_files = self.read_file(self.text_hk_path)
        self.text_object_files = self.read_file(self.text_ok_path)
        self.img_size = img_size

        if self.run_type == 'train':
            self.point_files, self.number_dict = self.read_file(self.p_path, number_dict)
            self.object_list = list(number_dict.keys())
            self.object_train_split = {}
            start_index = 0
            for obj_ in self.object_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.object_train_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.point_files = self.read_file(self.p_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]
        text_hd = self.text_human_files[index]
        text_od = self.text_object_files[index]

        if (self.run_type=='val'):
            point_path = self.point_files[index]
        else:
            object_name = img_path.split('/')[-4]
            affordance_name = img_path.split('/')[-2]
            range_ = self.object_train_split[object_name]
            point_sample_idx = random.sample(range(range_[0],range_[1]), self.pair_num)
      

            for i ,idx in enumerate(point_sample_idx):
                while True:
                    point_path = self.point_files[idx]
                    sele_affordance = point_path.split('/')[-2]
                    if sele_affordance == affordance_name:
                        point_sample_idx[i] = idx 
                        break
                    else:
                        idx = random.randint(range_[0],range_[1]-1)  # re-select idx

        Img = Image.open(img_path).convert('RGB')

        if(self.run_type == 'train'):
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Points_List = []
            affordance_label_List = []
            affordance_index_List = []
            for id_x in point_sample_idx:
                point_path = self.point_files[id_x]
                Points, affordance_label = self.extract_point_file(point_path)
                Points,_,_ = pc_normalize(Points)
                Points = Points.transpose()
                affordance_index = self.get_affordance_label(img_path)
                Points_List.append(Points)
                affordance_label_List.append(affordance_label)
                affordance_index_List.append(affordance_index)

        else:
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Point, affordance_label = self.extract_point_file(point_path)
            Point,_,_ = pc_normalize(Point)
            Point = Point.transpose()

        if(self.run_type == 'train'):
            return Img, text_hd, text_od, Points_List, affordance_label_List, affordance_index_List
        else:
            return Img, text_hd, text_od, Point, affordance_label, img_path, point_path

    def read_file(self, path, number_dict=None):
        file_list = []
        base_dir = os.path.dirname(path)
        parent_dir = os.path.dirname(base_dir)
        with open(path, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')

                # 尝试修正相对路径（训练列表中有以 'Data/...' 开头的相对路径）
                if not os.path.isabs(file) and not os.path.exists(file):
                    if file.startswith('Data/'):
                        candidate = os.path.join(parent_dir, file[5:])
                        if os.path.exists(candidate):
                            file = candidate
                    else:
                        candidate = os.path.join(parent_dir, file)
                        if os.path.exists(candidate):
                            file = candidate

                if number_dict != None:
                    object_ = file.split('/')[-4]
                    number_dict[object_] +=1
                file_list.append(file)

        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
    
    def extract_point_file(self, path):
        lines =  np.load(path)
        data_array = np.array(lines)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str):
        cut_str = str.split('/')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)

        return  index


class PIAD_L2(PIAD):
    """Level 2 dataset: replaces the two text strings (text_hd / text_od) with a
    single frozen Qwen2.5-VL visual-intent embedding per image.

    `vis_emb_path` is the .npy written by level2_extract_vis_emb.py; its sibling
    <prefix>.vis_emb.json holds {index: {abs_img_path: row}}. Lookup is by the
    resolved absolute image path (self.img_files[index]), exactly the string the
    extractor keyed on. A miss raises (fail-loud) so a stale/mismatched cache can
    never silently degrade a run.

    The returned tuple keeps PIAD's layout but slots 1 and 2 now carry the SAME
    float32 vis_emb tensor [D] (instead of text_hd / text_od strings); GREAT_L2
    only reads one of them. Default collate stacks [D] -> [B, D].
    """

    def __init__(self, run_type, setting_type, point_path, img_path,
                 text_hk_path, text_ok_path, vis_emb_path, pair=2, img_size=(224, 224)):
        super().__init__(run_type, setting_type, point_path, img_path,
                         text_hk_path, text_ok_path, pair, img_size)
        npy_path = vis_emb_path
        json_path = vis_emb_path[:-len('.npy')] + '.json' if vis_emb_path.endswith('.npy') \
            else vis_emb_path + '.json'
        self.vis_emb = np.load(npy_path).astype(np.float32)
        with open(json_path, 'r') as f:
            meta = json.load(f)
        self.vis_index = meta['index']          # abs_img_path -> row
        self.vis_dim = int(meta.get('dim', self.vis_emb.shape[1]))

    def _lookup_emb(self, img_path):
        row = self.vis_index.get(img_path)
        if row is None:
            # fall back to basename match (path roots can differ across machines)
            base = os.path.basename(img_path)
            for p, r in self.vis_index.items():
                if os.path.basename(p) == base:
                    row = r
                    break
        if row is None:
            raise KeyError(f"[PIAD_L2] no vis_emb cached for image: {img_path}")
        return torch.from_numpy(self.vis_emb[row]).float()

    def __getitem__(self, index):
        out = list(super().__getitem__(index))
        img_path = self.img_files[index]
        emb = self._lookup_emb(img_path)
        out[1] = emb
        out[2] = emb
        return tuple(out)


class PIAD_L3(PIAD):
    """Level 3 dataset: replaces the two text strings (text_hd / text_od) with a
    frozen Qwen2.5-VL visual feature *map* per image (a token sequence, not the
    single pooled vector L2 used).

    `vis_feat_path` is the .npy written by level3_extract_vis_feat.py with shape
    [N, M, D]; its sibling <prefix>.vis_feat.json holds {index:{abs_img_path:row}}.
    Lookup is by the resolved absolute image path (self.img_files[index]) with a
    basename fallback, then fail-loud — identical policy to PIAD_L2.

    The returned tuple keeps PIAD's layout but slots 1 and 2 now carry the SAME
    float32 feature map [M, D] (GREAT_L3 only reads one). Default collate stacks
    [M, D] -> [B, M, D].
    """

    def __init__(self, run_type, setting_type, point_path, img_path,
                 text_hk_path, text_ok_path, vis_feat_path, pair=2, img_size=(224, 224)):
        super().__init__(run_type, setting_type, point_path, img_path,
                         text_hk_path, text_ok_path, pair, img_size)
        npy_path = vis_feat_path
        json_path = vis_feat_path[:-len('.npy')] + '.json' if vis_feat_path.endswith('.npy') \
            else vis_feat_path + '.json'
        self.vis_feat = np.load(npy_path).astype(np.float32)     # [N, M, D]
        with open(json_path, 'r') as f:
            meta = json.load(f)
        self.vis_index = meta['index']          # abs_img_path -> row
        self.vis_dim = int(meta.get('dim', self.vis_feat.shape[2]))
        self.n_tokens = int(meta.get('n_tokens', self.vis_feat.shape[1]))

    def _lookup_feat(self, img_path):
        row = self.vis_index.get(img_path)
        if row is None:
            base = os.path.basename(img_path)
            for p, r in self.vis_index.items():
                if os.path.basename(p) == base:
                    row = r
                    break
        if row is None:
            raise KeyError(f"[PIAD_L3] no vis_feat cached for image: {img_path}")
        return torch.from_numpy(self.vis_feat[row]).float()      # [M, D]

    def __getitem__(self, index):
        out = list(super().__getitem__(index))
        img_path = self.img_files[index]
        feat = self._lookup_feat(img_path)
        out[1] = feat
        out[2] = feat
        return tuple(out)


class PIAD_L3_Online(PIAD):
    """Level 3 (LoRA) dataset: no offline cache. Instead of a feature map, slots
    1 and 2 carry the resolved absolute image path and the object category string,
    so the training loop can run Qwen2.5-VL online (with LoRA in the graph) and
    extract the visual feature map per batch.

    The default DataLoader collate leaves strings as a tuple, so slots 1/2 arrive
    in the loop as `(img_path_0, ...)` / `(object_0, ...)` — `build_qwen_inputs`
    in train.py turns them into Qwen processor tensors. The object is parsed
    exactly like MHACoT.py / level3_extract_vis_feat.py: `img_path.split('/')[-4]`.
    """

    def __init__(self, run_type, setting_type, point_path, img_path,
                 text_hk_path, text_ok_path, pair=2, img_size=(224, 224)):
        super().__init__(run_type, setting_type, point_path, img_path,
                         text_hk_path, text_ok_path, pair, img_size)

    def __getitem__(self, index):
        out = list(super().__getitem__(index))
        img_path = self.img_files[index]
        obj = img_path.split('/')[-4]
        out[1] = img_path
        out[2] = obj
        return tuple(out)

