import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.GREAT import get_GREAT, get_GREAT_L2, get_GREAT_L3, get_GREAT_L3_LoRA
from utils.loss import HM_Loss, kl_div
from utils.eval import evaluating, SIM
from data_utils.dataset_PIAD_GREAT import PIAD, PIAD_L2, PIAD_L3, PIAD_L3_Online
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import pdb
import logging
import random
import yaml
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict


# ---- L3 (LoRA) online Qwen helpers -----------------------------------------
# Mirror level3_extract_vis_feat.py so the online prompt is identical to the
# offline cache it replaces.

def _object_of(image_path):
    return image_path.split('/')[-4]


def _intent_prompt(obj):
    return (
        f"Point out which part of the {obj} in the image interacts with the "
        f"person, and describe the interaction between the {obj} and the person, "
        f"including the interaction type, the interaction part of the {obj}, and "
        f"the interaction part of the person."
    )


def build_qwen_inputs(processor, img_paths, objects, device):
    """Batch a list of (image_path, object) into Qwen2.5-VL processor tensors.

    Returns a dict (input_ids, attention_mask, pixel_values, image_grid_thw, ...)
    on `device`. Uses the same MHACoT-style intent prompt as the offline L3
    extractor; the prompt is never decoded against, it only conditions the fused
    image+language hidden state we read image tokens from.
    """
    from qwen_vl_utils import process_vision_info
    texts, images = [], []
    for p, obj in zip(img_paths, objects):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": p},
            {"type": "text",  "text": _intent_prompt(obj)},
        ]}]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
        img_in, _ = process_vision_info(messages)
        images.extend(img_in)
    inputs = processor(text=texts, images=images, padding=True, return_tensors='pt')
    return {k: v.to(device) for k, v in inputs.items()}


def slim_l3_lora_state(state):
    """Drop the frozen ~14GB Qwen base from a state_dict, keeping only the
    trainable side: GREAT downstream + visual_mhacot + the LoRA delta."""
    return {k: v for k, v in state.items()
            if (not k.startswith('vlm.')) or ('lora_' in k)}


def main(opt, dict):
    
    # l3_lora keeps a 7B Qwen in the training graph -> single GPU, no DDP.
    is_l3_lora = (opt.mode == 'l3_lora')
    distributed = opt.use_gpu and dict['run_type']=='train' and not is_l3_lora
    if distributed:
        dist.init_process_group(backend='gloo', init_method='env://')
        rank = dist.get_rank()
        size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        size = 1
        local_rank = 0
        if opt.use_gpu:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    save_path = opt.save_dir + opt.name
    if rank == 0 and not os.path.exists(save_path):
        os.makedirs(save_path)
    if distributed:
        dist.barrier()

    loger = logging.getLogger('Training')
    loger.setLevel(logging.INFO)
    log_name = opt.save_dir + opt.name + '/' + opt.log_name
    if rank == 0:
        logging.basicConfig(filename=log_name, level=logging.INFO)
    def log_string(str):
        if rank == 0:
            loger.info(str)
            print(str)

    img_train_path = dict['img_train']  
    point_train_path = dict['point_train']
    text_hd_train_path = dict['human_dictionary_train']
    text_od_train_path = dict['object_dictionary_train']
    img_val_path = dict['img_val']
    point_val_path = dict['point_val']
    text_hd_val_path = dict['human_dictionary_val']
    text_od_val_path = dict['object_dictionary_val']
    Setting = dict['Setting']
    batch_size = dict['batch_size']

    is_l2 = (opt.mode == 'l2')
    is_l3 = (opt.mode == 'l3')

    log_string('Start loading train data---')
    if is_l2:
        vis_emb_train = dict['vis_emb_train']
        vis_emb_val = dict['vis_emb_val']
        train_dataset = PIAD_L2('train', Setting, point_train_path, img_train_path,
                                text_hd_train_path, text_od_train_path, vis_emb_train, dict['pairing_num'])
    elif is_l3:
        vis_feat_train = dict['vis_feat_train']
        vis_feat_val = dict['vis_feat_val']
        train_dataset = PIAD_L3('train', Setting, point_train_path, img_train_path,
                                text_hd_train_path, text_od_train_path, vis_feat_train, dict['pairing_num'])
    elif is_l3_lora:
        train_dataset = PIAD_L3_Online('train', Setting, point_train_path, img_train_path,
                                       text_hd_train_path, text_od_train_path, dict['pairing_num'])
    else:
        train_dataset = PIAD('train', Setting, point_train_path, img_train_path, text_hd_train_path, text_od_train_path, dict['pairing_num'])
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, shuffle=(train_sampler is None), drop_last=True)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8 ,shuffle=True, drop_last=True)
    log_string(f'train data loading finish, loading data files:{len(train_dataset)}')

    log_string('Start loading val data---')
    if is_l2:
        val_dataset = PIAD_L2('val', Setting, point_val_path, img_val_path,
                              text_hd_val_path, text_od_val_path, vis_emb_val)
    elif is_l3:
        val_dataset = PIAD_L3('val', Setting, point_val_path, img_val_path,
                              text_hd_val_path, text_od_val_path, vis_feat_val)
    elif is_l3_lora:
        val_dataset = PIAD_L3_Online('val', Setting, point_val_path, img_val_path,
                                     text_hd_val_path, text_od_val_path)
    else:
        val_dataset = PIAD('val', Setting, point_val_path, img_val_path, text_hd_val_path, text_od_val_path)
    test_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    log_string(f'val data loading finish, loading data files:{len(val_dataset)}')

    if is_l2:
        vlm_dim = train_dataset.vis_dim
        model = get_GREAT_L2(vlm_dim=vlm_dim, img_model_path=dict['res18_pre'], pre_train=False,
                             N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                             proj_dim=dict['proj_dim'], num_heads=dict['num_heads'])
        # Load frozen B0 backbone (text-encoder keys absent -> strict=False),
        # then freeze everything except the trainable intent_proj branch.
        if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
            ckpt = torch.load(opt.checkpoint_path, map_location='cpu')
            state = ckpt['model'] if 'model' in ckpt else ckpt
            if any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            log_string(f'[L2] loaded B0 backbone (missing={len(missing)} unexpected={len(unexpected)})')
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith('intent_proj'))
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_string(f'[L2] trainable params (intent_proj only): {n_train}')
    elif is_l3:
        vlm_dim = train_dataset.vis_dim
        model = get_GREAT_L3(vlm_dim=vlm_dim, img_model_path=dict['res18_pre'], pre_train=False,
                             N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                             proj_dim=dict['proj_dim'], num_heads=dict['num_heads'])
        # Warm-start from B0 (visual_mhacot keys absent -> strict=False). Unlike
        # L2, NOTHING is frozen: all weights train end-to-end.
        if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
            ckpt = torch.load(opt.checkpoint_path, map_location='cpu')
            state = ckpt['model'] if 'model' in ckpt else ckpt
            if any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            log_string(f'[L3] warm-started from B0 (missing={len(missing)} unexpected={len(unexpected)})')
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_string(f'[L3] trainable params (end-to-end): {n_train}')
    elif is_l3_lora:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        vlm_model_id = dict['vlm_model_id']
        processor = AutoProcessor.from_pretrained(vlm_model_id)
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16)
        image_token_id = vlm.config.image_token_id
        vlm_dim = vlm.config.hidden_size
        # freeze the whole VLM, then re-open only the LoRA delta; keep the vision
        # tower fully frozen (Hammer-style) to fit a 7B base on a 24GB card.
        for p in vlm.parameters():
            p.requires_grad = False
        for p in vlm.visual.parameters():
            p.requires_grad = False
        vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable()
        lora_targets = [m.strip() for m in opt.lora_target_modules.split(',') if m.strip()]
        lora_cfg = LoraConfig(r=opt.lora_r, lora_alpha=opt.lora_alpha,
                              target_modules=lora_targets, lora_dropout=opt.lora_dropout,
                              bias='none', task_type='CAUSAL_LM')
        vlm = get_peft_model(vlm, lora_cfg)
        model = get_GREAT_L3_LoRA(vlm=vlm, vlm_dim=vlm_dim, image_token_id=image_token_id,
                                  img_model_path=dict['res18_pre'], pre_train=False,
                                  N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                                  proj_dim=dict['proj_dim'], num_heads=dict['num_heads'])
        # Warm-start the GREAT side from B0 (vlm.* / visual_mhacot keys absent ->
        # strict=False). LoRA delta + GREAT downstream train; frozen Qwen base does not.
        if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
            ckpt = torch.load(opt.checkpoint_path, map_location='cpu')
            state = ckpt['model'] if 'model' in ckpt else ckpt
            if any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            log_string(f'[L3-LoRA] warm-started GREAT side from B0 '
                       f'(missing={len(missing)} unexpected={len(unexpected)})')
        n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n)
        n_great = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith('vlm.'))
        log_string(f'[L3-LoRA] trainable params: LoRA={n_lora} GREAT={n_great}')
    else:
        model = get_GREAT(img_model_path=dict['res18_pre'], N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                           proj_dim=dict['proj_dim'], num_heads=dict['num_heads'])

    model = model.to(device)
    criterion_hm = HM_Loss()
    criterion_ce = nn.CrossEntropyLoss()
    '''
    param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "img_encoder" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "img_encoder" in n and p.requires_grad], "lr": 1e-5}]
    '''
    if is_l3_lora:
        # LoRA adapter at its own lr; GREAT downstream at the base lr.
        vlm_lr = dict.get('vlm_lr', opt.vlm_lr)
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
        great_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('vlm.')]
        optimizer = torch.optim.Adam(
            [{'params': lora_params, 'lr': vlm_lr},
             {'params': great_params, 'lr': dict['lr']}],
            betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.decay_rate)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=dict['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['Epoch'], eta_min=1e-6)

    if opt.resume:
        map_location = device if opt.use_gpu else torch.device("cpu")
        model_checkpoint = torch.load(opt.checkpoint_path, map_location=map_location)
        model_state = model_checkpoint['model']
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {key.replace('module.', '', 1): value for key, value in model_state.items()}
        model.load_state_dict(model_state)
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['Epoch']
    else:
        start_epoch = -1

   
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)

    #pdb.set_trace()
    criterion_hm = criterion_hm.to(device)
    criterion_ce = criterion_ce.to(device)
    
    best_IOU = 0
    '''
    Training
    '''
    for epoch in range(start_epoch+1, dict['Epoch']):
        if distributed:
            train_sampler.set_epoch(epoch)
        log_string(f'Epoch:{epoch} strat-------')
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        log_string(f'lr_rate:{learning_rate}')

        num_batches = len(train_loader)
        loss_sum = 0
        total_point = 0
        model = model.train()
        model = model.to(device)
        if is_l2:
            # keep frozen backbone in eval() so its BatchNorm running stats are
            # not corrupted; only the trainable intent_proj stays in train mode.
            base = model.module if distributed else model
            base.eval()
            base.intent_proj.train()
        for i,(img, text_hd, text_od, points, labels, logits_labels) in enumerate(train_loader):

            optimizer.zero_grad()
            temp_loss = 0
            # l3_lora: run the (LoRA) Qwen ONCE per image, reuse V across pairings.
            if is_l3_lora:
                if opt.use_gpu:
                    img = img.to(device)
                qin = build_qwen_inputs(processor, list(text_hd), list(text_od), device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    V, vmask = model.encode_vlm(qin)
                V = V.float()
            for point, label, logits_label in zip(points, labels, logits_labels):

                point, label = point.float(), label.float()

                if(opt.use_gpu):
                    img = img.to(device)
                    point = point.to(device)
                    label = label.to(device)
                    logits_label = logits_label.to(device)


                if is_l2:
                    vis_emb = text_hd.float().to(device)
                    _3d = model(img, point, vis_emb)
                    loss_hm = criterion_hm(_3d, label)
                elif is_l3:
                    vis_feat = text_hd.float().to(device)
                    _3d, e_hk, h_aff = model(img, point, vis_feat)
                    loss_hm = criterion_hm(_3d, label)
                    loss_cons = (1 - F.cosine_similarity(e_hk, h_aff, dim=-1)).mean()
                    loss_hm = loss_hm + opt.lambda_cons * loss_cons
                elif is_l3_lora:
                    _3d, e_hk, h_aff = model.decode(img, point, V, vmask)
                    loss_hm = criterion_hm(_3d, label)
                    loss_cons = (1 - F.cosine_similarity(e_hk, h_aff, dim=-1)).mean()
                    loss_hm = loss_hm + opt.lambda_cons * loss_cons
                else:
                    _3d = model(img, point, text_hd, text_od)
                    loss_hm = criterion_hm(_3d, label)

                temp_loss += loss_hm

            if rank == 0:
                print(f'Epoch:{epoch} | iteration:{i} | loss:{temp_loss.item()}')
            temp_loss.backward() 
            optimizer.step()   
            loss_sum += temp_loss.item()

        mean_loss = loss_sum / (num_batches*dict['pairing_num'])
        log_string(f'Epoch:{epoch} | mean_loss:{mean_loss}')

        if rank == 0 and opt.storage == True:
            if((epoch+1) % 1==0):
                model_path = save_path + '/Epoch_' + str(epoch+1) + '.pt'
                sd = model.module.state_dict() if distributed else model.state_dict()
                if is_l3_lora:
                    sd = slim_l3_lora_state(sd)
                checkpoint = {
                    'model': sd,
                    'optimizer': optimizer.state_dict(),
                    'Epoch': epoch
                }
                torch.save(checkpoint, model_path)
                log_string(f'model saved at {model_path}')

        '''
        Evalization
        '''
        if((epoch+1)%1 == 0):
            if distributed:
                dist.barrier()
            if rank == 0:
                results = torch.zeros((len(val_dataset), 2048, 1))
                targets = torch.zeros((len(val_dataset), 2048, 1))
                num = 0
                with torch.no_grad():
                    log_string(f'EVALUATION strat-------')
                    num_batches = len(val_loader)
                    val_loss_sum = 0
                    total_MAE = 0
                    total_point = 0
                    model = model.eval()
                    for i,(img, text_hd, text_od, point, label,_,_) in enumerate(val_loader):
                        print(f'iteration: {i} start----')
                        point, label = point.float(), label.float()

                        if(opt.use_gpu):
                            img = img.to(device)
                            point = point.to(device)
                            label = label.to(device)


                        if is_l2:
                            vis_emb = text_hd.float().to(device)
                            _3d = model(img, point, vis_emb)
                        elif is_l3:
                            vis_feat = text_hd.float().to(device)
                            _3d = model(img, point, vis_feat)[0]
                        elif is_l3_lora:
                            qin = build_qwen_inputs(processor, list(text_hd), list(text_od), device)
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                V, vmask = model.encode_vlm(qin)
                            _3d = model.decode(img, point, V.float(), vmask)[0]
                        else:
                            _3d = model(img, point, text_hd, text_od)

                        val_loss = criterion_hm(_3d, label)


                        mae, point_nums = evaluating(_3d, label)
                        total_point += point_nums
                        val_loss_sum += val_loss.item()
                        total_MAE += mae.item()
                        pred_num = _3d.shape[0]
                        print(f'---val_loss | {val_loss.item()}')
                        results[num : num+pred_num, :, :] = _3d.cpu()
                        targets[num : num+pred_num, :, :] = label.cpu()
                        num += pred_num

                    val_mean_loss = val_loss_sum / num_batches
                    log_string(f'Epoch_{epoch} | val_loss | {val_mean_loss}')
                    mean_mae = total_MAE / total_point
                    results = results.detach().numpy()
                    targets = targets.detach().numpy()
                    SIM_matrix = np.zeros(targets.shape[0])
                    for i in range(targets.shape[0]):
                        SIM_matrix[i] = SIM(results[i], targets[i])

                    sim = np.mean(SIM_matrix)
                    AUC = np.zeros((targets.shape[0], targets.shape[2]))
                    IOU = np.zeros((targets.shape[0], targets.shape[2]))
                    IOU_thres = np.linspace(0, 1, 20)
                    targets = targets >= 0.5
                    targets = targets.astype(int)
                    for i in range(AUC.shape[0]):
                        t_true = targets[i]
                        p_score = results[i]

                        if np.sum(t_true) == 0:
                            AUC[i] = np.nan
                            IOU[i] = np.nan
                        else:
                            auc = roc_auc_score(t_true, p_score)
                            AUC[i] = auc

                            p_mask = (p_score > 0.5).astype(int)
                            temp_iou = []
                            for thre in IOU_thres:
                                p_mask = (p_score >= thre).astype(int)
                                intersect = np.sum(p_mask & t_true)
                                union = np.sum(p_mask | t_true)
                                temp_iou.append(1.*intersect/union)
                            temp_iou = np.array(temp_iou)
                            aiou = np.mean(temp_iou)
                            IOU[i] = aiou

                    AUC = np.nanmean(AUC)
                    IOU = np.nanmean(IOU)

                    log_string(f'AUC:{AUC} | IOU:{IOU} | SIM:{sim} | MAE:{mean_mae}')

                    current_IOU = IOU
                    if(current_IOU > best_IOU):
                        best_IOU = current_IOU
                        best_model_path = save_path + '/best_seen.pt'
                        sd = model.module.state_dict() if distributed else model.state_dict()
                        if is_l3_lora:
                            sd = slim_l3_lora_state(sd)
                        checkpoint = {
                            'model': sd,
                            'optimizer': optimizer.state_dict(),
                            'Epoch': epoch
                        }
                        torch.save(checkpoint, best_model_path)
                        log_string(f'best model saved at {best_model_path}')
            if distributed:
                dist.barrier()
        scheduler.step()

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)  
	torch.manual_seed(seed) 
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False 
	torch.backends.cudnn.deterministic = True 


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu device id')
    parser.add_argument('--decay_rate', type=float, default=1e-3, help='weight decay [default: 1e-3]')
    parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
    parser.add_argument('--save_dir', type=str, default='./runs/', help='path to save .pt model while training')
    parser.add_argument('--name', type=str, default='GREAT1', help='training name to classify each training process')
    parser.add_argument('--resume', type=str, default=False, help='start training from previous epoch')
    parser.add_argument('--checkpoint_path', type=str, default='./runs/best_seen.pt', help='checkpoint path')
    parser.add_argument('--log_name', type=str, default='train_seen.log', help='the name of current training')
    parser.add_argument('--storage', type=bool, default=False, help='whether to storage the model during training')
    parser.add_argument('--yaml', type=str, default='config/config_seen_GREAT.yaml', help='yaml path')
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'l2', 'l3', 'l3_lora'], help='base = original GREAT; l2 = visual-intent embedding swap; l3 = offline visual-grounded MHACoT (cached feat); l3_lora = online Qwen2.5-VL with LoRA (end-to-end)')
    parser.add_argument('--lambda_cons', type=float, default=0.1, help='L3 consistency loss weight (1 - cos(e_hk, h_aff))')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # L3-LoRA (online Qwen2.5-VL) options
    parser.add_argument('--vlm_lr', type=float, default=2e-4, help='LoRA adapter learning rate (l3_lora); yaml vlm_lr overrides')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (l3_lora)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha (l3_lora)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout (l3_lora)')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,v_proj', help='comma-separated LoRA target modules (l3_lora)')

    opt = parser.parse_args()
    seed_torch(seed=opt.seed)
    # anomaly detection ~2x memory/time; with a 7B LoRA graph it is too costly.
    torch.autograd.set_detect_anomaly(opt.mode != 'l3_lora')
    dict = read_yaml(opt.yaml)
    main(opt, dict)