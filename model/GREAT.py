import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from einops import rearrange
from transformers import AutoModel, AutoTokenizer
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cross_Attention(nn.Module):    
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim 
        self.proj_dim = proj_dim 
        self.proj_hq = nn.Linear(self.emb_dim, proj_dim)
        self.proj_oq = nn.Linear(self.emb_dim, proj_dim)
        self.proj_hk = nn.Linear(self.emb_dim, proj_dim)
        self.proj_hv = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ok = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ov = nn.Linear(self.emb_dim, proj_dim)
        self.scale = self.proj_dim ** (-0.5) 

        self.layernorm = nn.LayerNorm(self.emb_dim)
    def forward(self, hk, ok):

        '''
        hk : human knowledge [B,N_hk,C]
        ok : object knowledge [B,N_ok,C]
        '''

        hk_q = self.proj_hq(hk)                                        
        ok_key = self.proj_ok(ok)                                       
        ok_value = self.proj_ov(ok)

        ok_key_ = torch.cat((hk_q,ok_key),dim=1)  
        ok_value_ = torch.cat((hk_q,ok_value),dim=1)

        ok_q = self.proj_oq(ok)
        hk_key = self.proj_hk(hk)
        hk_value = self.proj_hv(hk)

        hk_key_ = torch.cat((ok_q,hk_key),dim=1)  
        hk_value_ = torch.cat((ok_q,hk_value),dim=1)

        atten_I1 = torch.bmm(hk_q, ok_key_.permute(0, 2, 1))*self.scale                 
        atten_I1 = atten_I1.softmax(dim=-1)                        
        I_1 = torch.bmm(atten_I1, ok_value_)                                

        atten_I2 = torch.bmm(ok_q, hk_key_.permute(0, 2, 1))*self.scale                 
        atten_I2 = atten_I2.softmax(dim=-1)
        I_2 = torch.bmm(atten_I2, hk_value_)                              

        I_1 = self.layernorm(hk + I_1)                                 
        I_2 = self.layernorm(ok + I_2)    
        return I_1, I_2

class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):  
        super(Self_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads 
        assert self.head_dim * num_heads == hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()                             
        

        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, seq_len, head_dim)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)         # (batch_size, num_heads, seq_len, head_dim)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # (batch_size, num_heads, seq_len, head_dim)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_size ** 0.5)                  # (batch_size, num_heads, seq_len, seq_len)

        attention_weights = nn.functional.softmax(scores, dim=-1)                                           # (batch_size, num_heads, seq_len, seq_len)   

        out = torch.matmul(attention_weights, values)                                                       # (batch_size, num_heads, seq_len, head_dim)  

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)                         # (batch_size, seq_len, embed_dim)

        out = self.ln(out + x) 
        return out

class Cross_Modal_Feature_Fusion(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.cross_atten1 = Cross_Attention(emb_dim = self.emb_dim, proj_dim = self.proj_dim)


        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
    )
        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim//2), 
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 2),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim//2, self.emb_dim),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim),
            SwapAxes(),
        )

        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fusion = nn.Sequential(                                        
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),   
            nn.ReLU()        
        )
        
    def forward(self,f_t,f_p):
        _, N_P, _ = f_p.size()
        f_to, f_po = self.cross_atten1(f_t, f_p)            
        f_to = f_to + self.fc(f_to)                     
        f_po = f_po + self.fc(f_po)                    
        f_t_p = self.pool(f_to.permute(0,2,1))                 
        f_t_r = f_t_p.repeat(1, 1, N_P)               

        joint = torch.cat((f_po.permute(0,2,1), f_t_r), dim = 1)
        output = self.fusion(joint)   
        return output
        
class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()

        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]

class Img_Encoder(nn.Module):
    def __init__(self):
        super(Img_Encoder, self).__init__()

        self.model = models.resnet18(pretrained=False)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        B, _, _, _ = img.size()
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))

        out = self.model.maxpool(out) 
        out = self.model.layer1(out)   
        down_1 = self.model.layer2(out)         
        down_2 = self.model.layer3(down_1)       
        down_3 = self.model.layer4(down_2)
       
        return down_3


class Text_Encoder(nn.Module):
    def __init__(self, emb_dim = 512, freeze_text_encoder = True):
        super().__init__()
        self.emb_dim = emb_dim

        self.text_encoder = AutoModel.from_pretrained('/mnt/sdb/wyn/model/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('/mnt/sdb/wyn/model/roberta-base')
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)  
        self.text_resizer = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, emb_dim, bias=True),
                                          nn.LayerNorm(emb_dim, eps=1e-12))
    
    def forward(self, text_queries):
        """
        input: text_queries 
        output:
            text_embedding:  [batch_size, num_phrases, emb_dim] 
            attention_mask:  [batch_size, num_phrases, seq_len] 
        """
        # Separate each input sentence by comma to get multiple phrases
        split_queries = [query.split(',') for query in text_queries]
        
        all_encoded_text = []

        for phrases in split_queries:
            tokenized_phrases = self.tokenizer.batch_encode_plus(phrases, padding='longest', return_tensors='pt')  
            tokenized_phrases = tokenized_phrases.to(device)
            outputs = self.text_encoder(**tokenized_phrases)
            pooled_output = outputs.pooler_output
            resized_phrases = self.text_resizer(pooled_output)  # [n_phrases, emb_dim]
            # img_text_fusion.reshape has Linear(3, ...) — must be exactly 3 phrases.
            # Original data always has 3; CoT strings may have fewer or more commas.
            n, d = resized_phrases.shape
            if n < 3:
                pad = resized_phrases.new_zeros(3 - n, d)
                resized_phrases = torch.cat([resized_phrases, pad], dim=0)
            elif n > 3:
                resized_phrases = resized_phrases[:3]
            all_encoded_text.append(resized_phrases)

        text_embeddings = torch.stack(all_encoded_text)

        return text_embeddings


class Text_Encoder2(nn.Module):
    def __init__(self, emb_dim = 512, freeze_text_encoder = True):
        super().__init__()
        self.emb_dim = emb_dim

        self.text_encoder = AutoModel.from_pretrained('/mnt/sdb/wyn/model/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('/mnt/sdb/wyn/model/roberta-base')
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)   
        self.text_resizer = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, emb_dim, bias=True),
                                          nn.LayerNorm(emb_dim, eps=1e-12))
    
    def forward(self, text_queries):
        
        with torch.inference_mode(mode=self.freeze_text_encoder):
            tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', max_length=512, truncation=True, return_tensors='pt')  
        tokenized_queries = tokenized_queries.to(device)
        outputs = self.text_encoder(**tokenized_queries)
        pooled_output = outputs.pooler_output
        pooled_output = pooled_output.unsqueeze(1)
        return self.text_resizer(pooled_output)
    
class affordance_dictionary_fusion(nn.Module):
    def __init__(self, emb_dim = 512, proj_dim = 512, num_heads = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.cross_atten = Cross_Attention(emb_dim = self.emb_dim, proj_dim = self.proj_dim)
        self.h_atten = Self_Attention(self.emb_dim, self.num_heads)
        self.o_atten = Self_Attention(self.emb_dim, self.num_heads)

    def forward(self,f_hk,f_ok):
        H, O = self.cross_atten(f_hk, f_ok)              
        H_= self.h_atten(H)
        O_= self.o_atten(O)
        return H_, O_

class img_text_fusion(nn.Module):
    def __init__(self, emb_dim = 512, proj_dim = 512):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()

        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )         
        self.reshape = nn.Sequential(
            nn.Linear(3, 3 * 8),
            SwapAxes(),
            nn.BatchNorm1d(3 * 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(3 * 8, 49),
        )                  
    def forward(self,F_i,T_h_):    
        T_h_ = self.reshape(T_h_.permute(0,2,1))  
        I_ = torch.cat((F_i, T_h_),dim=1)
        I_ = self.fusion(I_)  
        return I_
    
class Decoder(nn.Module):
    def __init__(self, additional_channel, emb_dim, proj_dim):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        #upsample
        self.fp3 = PointNetFeaturePropagation(in_channel=512+self.emb_dim, mlp=[768, 512])   
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512])  
        self.fp1 = PointNetFeaturePropagation(in_channel=518+additional_channel, mlp=[512, 512]) 

        self.cmff = Cross_Modal_Feature_Fusion(emb_dim, proj_dim)
        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim // 8, 1),
        )
        self.reshape = nn.Sequential(
            nn.Linear(49, 49 * 8),
            SwapAxes(),
            nn.BatchNorm1d(49 * 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(49 * 8, 2048),
        )          
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )  
    def forward(self, T_o, I_h, encoder_p, return_feat=False):

        '''
        T_o --->object knowledge embedding
        I_h ---> [B, N_i, C]
        encoder_p  ---> [Hierarchy feature]
        return_feat ---> if True, also return the pooled mid-feature h_aff [B, emb]
                         (Level-3 consistency loss). Default False keeps GREAT /
                         GREAT_L2 output byte-identical.
        '''
        B, _, _ = I_h.shape

        p_0, p_1, p_2, p_3 = encoder_p

        p_3[1] = self.cmff(T_o, p_3[1].transpose(-2, -1))
        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], p_3[1])


        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)


        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]],1), up_sample)

        F_I = self.reshape(I_h.permute(0,2,1))

        F_j = torch.cat((F_I, up_sample),dim=1)
        F_j_fusion = self.fusion(F_j)

        _3daffordance = self.out_head(F_j_fusion.permute(0, 2, 1))
        _3daffordance = self.sigmoid(_3daffordance)

        if return_feat:
            h_aff = F_j_fusion.mean(dim=-1)          # [B, emb] pooled over points
            return _3daffordance, h_aff
        return _3daffordance

class GREAT(nn.Module):
    def __init__(self, img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
                N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, freeze_text_encoder = True):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel

        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        self.img_encoder = Img_Encoder()
        if pre_train:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)
        self.text_encoder = Text_Encoder(self.emb_dim, freeze_text_encoder = True)
        self.text_encoder2 = Text_Encoder2(self.emb_dim, freeze_text_encoder = True)

        self.affordance_dictionary_fusion  = affordance_dictionary_fusion(self.emb_dim, self.proj_dim, self.num_heads)
        self.img_text_fusion = img_text_fusion(self.emb_dim, self.proj_dim)
        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.proj_dim)


    def forward(self, img, xyz, text_human, text_object):

        '''
        img: [B, 3, H, W]
        xyz: [B, 3, 2048]
        '''

        B, C, N = xyz.size()
        F_I = self.img_encoder(img)     
        F_i = F_I.view(B, self.emb_dim, -1)         

        F_p_wise = self.point_encoder(xyz)
        T_h= self.text_encoder(text_human)
        T_o = self.text_encoder2(text_object)

        T_h_, T_o_ =self.affordance_dictionary_fusion(T_h, T_o)     
        I_h = self.img_text_fusion(F_i,T_h_)         

        _3daffordance = self.decoder(T_o_, I_h.permute(0,2,1), F_p_wise)

        return _3daffordance


def get_GREAT(img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
    N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, freeze_text_encoder = True):

    model = GREAT(img_model_path, pre_train, normal_channel, local_rank,
    N_p, emb_dim, proj_dim, num_heads, freeze_text_encoder)
    return model


# =============================================================================
# Level 2 — visual-intent embedding swap
#
# Replaces GREAT's text-encoder branch (Text_Encoder / Text_Encoder2 producing
# T_h [B,3,emb] and T_o [B,1,emb]) with a trainable projection MLP fed by a
# frozen Qwen2.5-VL "visual intent" embedding (one [B, vlm_dim] vector per
# image, cached offline by level2_extract_vis_emb.py).
#
# Everything downstream of the text branch (affordance_dictionary_fusion,
# img_text_fusion, decoder) is byte-identical to GREAT, so the B0 checkpoint
# loads into those submodules with strict=False and stays frozen during L2
# training — only `intent_proj` learns.
# =============================================================================

class IntentProj(nn.Module):
    """Frozen-VLM visual intent embedding -> GREAT's (T_h', T_o') text slots.

    Input : vis_emb [B, vlm_dim]
    Output: T_h' [B, 3, emb_dim]   (human/affordance knowledge, mirrors T_h)
            T_o' [B, 1, emb_dim]   (object knowledge,            mirrors T_o)

    The 3 / 1 token counts are mandatory: img_text_fusion.reshape expects
    exactly 3 phrase tokens (Linear(3, ...)) and the decoder consumes a single
    object-knowledge token.
    """
    def __init__(self, vlm_dim, emb_dim=512, n_hk=3, n_ok=1):
        super().__init__()
        self.vlm_dim = vlm_dim
        self.emb_dim = emb_dim
        self.n_hk = n_hk
        self.n_ok = n_ok

        self.trunk = nn.Sequential(
            nn.Linear(vlm_dim, emb_dim),
            nn.GELU(),
        )
        self.hk_head = nn.Linear(emb_dim, n_hk * emb_dim)
        self.ok_head = nn.Linear(emb_dim, n_ok * emb_dim)
        self.hk_norm = nn.LayerNorm(emb_dim)
        self.ok_norm = nn.LayerNorm(emb_dim)

    def forward(self, vis_emb):
        B = vis_emb.size(0)
        h = self.trunk(vis_emb)                                   # [B, emb]
        T_h = self.hk_head(h).view(B, self.n_hk, self.emb_dim)    # [B, 3, emb]
        T_o = self.ok_head(h).view(B, self.n_ok, self.emb_dim)    # [B, 1, emb]
        T_h = self.hk_norm(T_h)
        T_o = self.ok_norm(T_o)
        return T_h, T_o


class GREAT_L2(nn.Module):
    """GREAT with the text branch replaced by IntentProj.

    Submodules img_encoder / point_encoder / affordance_dictionary_fusion /
    img_text_fusion / decoder are named identically to GREAT so that a B0
    GREAT checkpoint loads into them with strict=False.
    """
    def __init__(self, vlm_dim, img_model_path=None, pre_train=False, normal_channel=False,
                 local_rank=None, N_p=64, emb_dim=512, proj_dim=512, num_heads=4):
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.additional_channel = 3 if self.normal_channel else 0

        self.img_encoder = Img_Encoder()
        if pre_train and img_model_path is not None:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)
        self.affordance_dictionary_fusion = affordance_dictionary_fusion(self.emb_dim, self.proj_dim, self.num_heads)
        self.img_text_fusion = img_text_fusion(self.emb_dim, self.proj_dim)
        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.proj_dim)

        # the only trainable branch in L2
        self.intent_proj = IntentProj(vlm_dim, emb_dim=self.emb_dim)

    def forward(self, img, xyz, vis_emb):
        '''
        img:     [B, 3, H, W]
        xyz:     [B, 3, 2048]
        vis_emb: [B, vlm_dim]  frozen Qwen2.5-VL visual intent embedding
        '''
        B, C, N = xyz.size()
        F_I = self.img_encoder(img)
        F_i = F_I.view(B, self.emb_dim, -1)

        F_p_wise = self.point_encoder(xyz)

        T_h, T_o = self.intent_proj(vis_emb)            # [B,3,emb], [B,1,emb]

        T_h_, T_o_ = self.affordance_dictionary_fusion(T_h, T_o)
        I_h = self.img_text_fusion(F_i, T_h_)

        _3daffordance = self.decoder(T_o_, I_h.permute(0, 2, 1), F_p_wise)
        return _3daffordance


def get_GREAT_L2(vlm_dim, img_model_path=None, pre_train=False, normal_channel=False,
                 local_rank=None, N_p=64, emb_dim=512, proj_dim=512, num_heads=4):
    model = GREAT_L2(vlm_dim, img_model_path, pre_train, normal_channel, local_rank,
                     N_p, emb_dim, proj_dim, num_heads)
    return model


# =============================================================================
# Level 3 — visual-grounded MHACoT + consistency loss (end-to-end)
#
# Unlike L2 (a single mean-pooled VLM vector -> tiny MLP, frozen backbone), L3
# queries a *visual feature map* V [B, M, vlm_dim] (the image-token subsequence
# of a frozen Qwen2.5-VL last hidden state under the MHACoT intent prompt,
# cached offline by level3_extract_vis_feat.py) with 4 learnable step tokens via
# cross-attention. The 4 steps mirror the MHACoT chain Q1-Q4 and are routed to
# GREAT's two knowledge slots:
#     Q1,Q2 -> ok_head -> T_o [B,1,emb]  (object knowledge)
#     Q3,Q4 -> hk_head -> T_h [B,3,emb]  (human/affordance knowledge)
# Everything downstream (CMAFM, img_text_fusion, decoder) is named identically
# to GREAT so the B0 checkpoint warm-starts those submodules (strict=False);
# in L3 ALL weights then train end-to-end. The decoder additionally exposes a
# pooled mid-feature h_aff for L_consistency = 1 - cos(e_hk, h_aff).
# =============================================================================

class VisualMHACoT(nn.Module):
    """4-step visual reasoning chain over the cached VLM feature map.

    Input : V     [B, M, vlm_dim]   frozen Qwen2.5-VL image-token features
    Output: T_h   [B, n_hk, emb]    human/affordance knowledge (mirrors GREAT T_h)
            T_o   [B, n_ok, emb]    object knowledge            (mirrors GREAT T_o)
            e_hk  [B, emb]          pooled affordance/human embedding (consistency)

    The 3/1 token counts are mandatory: img_text_fusion.reshape expects exactly
    3 phrase tokens (Linear(3,...)) and the decoder consumes one object token.
    """
    def __init__(self, vlm_dim, emb_dim=512, n_steps=4, num_heads=4, n_hk=3, n_ok=1):
        super().__init__()
        self.vlm_dim = vlm_dim
        self.emb_dim = emb_dim
        self.n_steps = n_steps
        self.n_hk = n_hk
        self.n_ok = n_ok

        self.v_proj = nn.Linear(vlm_dim, emb_dim)
        self.v_norm = nn.LayerNorm(emb_dim)

        # one learnable query token per MHACoT step (Q1 part / Q2 geom /
        # Q3 current interaction / Q4 additional interactions)
        self.step_tokens = nn.Parameter(torch.randn(n_steps, emb_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

        # route pooled (Q1,Q2)->object and (Q3,Q4)->human/affordance
        self.ok_head = nn.Linear(2 * emb_dim, n_ok * emb_dim)
        self.hk_head = nn.Linear(2 * emb_dim, n_hk * emb_dim)
        self.ok_norm = nn.LayerNorm(emb_dim)
        self.hk_norm = nn.LayerNorm(emb_dim)

    def forward(self, V, key_padding_mask=None):
        '''
        V                : [B, M, vlm_dim]
        key_padding_mask : [B, M] bool, True = padded (ignored) key position.
                           None -> all keys valid (offline L3 behaviour unchanged).
        '''
        B = V.size(0)
        Vp = self.v_norm(self.v_proj(V))                          # [B, M, emb]

        q = self.step_tokens.unsqueeze(0).expand(B, -1, -1)       # [B, 4, emb]
        attn_out, _ = self.cross_attn(q, Vp, Vp,
                                      key_padding_mask=key_padding_mask)  # [B, 4, emb]
        s = self.norm1(q + attn_out)
        s = self.norm2(s + self.ffn(s))                           # [B, 4, emb]

        ok_in = torch.cat([s[:, 0], s[:, 1]], dim=-1)            # [B, 2*emb]
        hk_in = torch.cat([s[:, 2], s[:, 3]], dim=-1)            # [B, 2*emb]

        T_o = self.ok_norm(self.ok_head(ok_in).view(B, self.n_ok, self.emb_dim))
        T_h = self.hk_norm(self.hk_head(hk_in).view(B, self.n_hk, self.emb_dim))

        e_hk = s[:, 2:4].mean(dim=1)                              # [B, emb]
        return T_h, T_o, e_hk


class GREAT_L3(nn.Module):
    """GREAT with the text branch replaced by VisualMHACoT, trained end-to-end.

    Submodules img_encoder / point_encoder / affordance_dictionary_fusion /
    img_text_fusion / decoder are named identically to GREAT so a B0 checkpoint
    warm-starts them with strict=False. Unlike L2 nothing is frozen here.
    """
    def __init__(self, vlm_dim, img_model_path=None, pre_train=False, normal_channel=False,
                 local_rank=None, N_p=64, emb_dim=512, proj_dim=512, num_heads=4):
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.additional_channel = 3 if self.normal_channel else 0

        self.img_encoder = Img_Encoder()
        if pre_train and img_model_path is not None:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)
        self.affordance_dictionary_fusion = affordance_dictionary_fusion(self.emb_dim, self.proj_dim, self.num_heads)
        self.img_text_fusion = img_text_fusion(self.emb_dim, self.proj_dim)
        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.proj_dim)

        # visual-grounded reasoning chain (new in L3)
        self.visual_mhacot = VisualMHACoT(vlm_dim, emb_dim=self.emb_dim, num_heads=self.num_heads)

    def forward(self, img, xyz, vis_feat):
        '''
        img:      [B, 3, H, W]
        xyz:      [B, 3, 2048]
        vis_feat: [B, M, vlm_dim]  frozen Qwen2.5-VL visual feature map
        returns:  (pred [B,2048,1], e_hk [B,emb], h_aff [B,emb])
        '''
        B, C, N = xyz.size()
        F_I = self.img_encoder(img)
        F_i = F_I.view(B, self.emb_dim, -1)

        F_p_wise = self.point_encoder(xyz)

        T_h, T_o, e_hk = self.visual_mhacot(vis_feat)

        T_h_, T_o_ = self.affordance_dictionary_fusion(T_h, T_o)
        I_h = self.img_text_fusion(F_i, T_h_)

        pred, h_aff = self.decoder(T_o_, I_h.permute(0, 2, 1), F_p_wise, return_feat=True)
        return pred, e_hk, h_aff


def get_GREAT_L3(vlm_dim, img_model_path=None, pre_train=False, normal_channel=False,
                 local_rank=None, N_p=64, emb_dim=512, proj_dim=512, num_heads=4):
    model = GREAT_L3(vlm_dim, img_model_path, pre_train, normal_channel, local_rank,
                     N_p, emb_dim, proj_dim, num_heads)
    return model


# =============================================================================
# Level 3 (LoRA) — online Qwen2.5-VL in the training graph
#
# Unlike offline L3 (frozen Qwen, cached vis_feat.npy), here the (LoRA-wrapped)
# Qwen runs inside forward so gradients reach the adapters. We take the
# image-token subsequence of the last hidden state as the visual feature map V,
# pad it to the batch max with a key_padding_mask, and feed VisualMHACoT exactly
# as offline L3 does. Everything downstream is the same GREAT stack (warm-started
# from B0); the Qwen base stays frozen, only its LoRA delta + the GREAT side
# (visual_mhacot + img/point/fusion/decoder) train.
#
# The Qwen forward and the GREAT decode are split into two methods (encode_vlm /
# decode) so the loop can run Qwen ONCE per image and reuse V across the
# pairing_num point clouds. Runs single-GPU (no DDP) — a 7B base barely fits one
# 24GB card.
# =============================================================================

def _gather_image_tokens(hidden, input_ids, image_token_id):
    """Per-sample, select the image-token positions of `hidden` and pad to the
    batch max.

    hidden        : [B, T, D]   last hidden state
    input_ids     : [B, T]      token ids (same T, padded by the processor)
    image_token_id: int

    Returns
        V    : [B, Mmax, D]  image-token features, zero-padded
        mask : [B, Mmax]     bool, True = padded (ignored) position
    If a sample has no image token (unexpected), falls back to all its tokens so
    the map is never empty.
    """
    B, T, D = hidden.shape
    sel = (input_ids == image_token_id)                  # [B, T]
    rows = []
    counts = []
    for b in range(B):
        s = sel[b]
        if not bool(s.any()):
            s = torch.ones(T, dtype=torch.bool, device=hidden.device)
        rows.append(hidden[b][s])                        # [m_b, D]
        counts.append(rows[-1].shape[0])
    Mmax = max(counts)
    V = hidden.new_zeros(B, Mmax, D)
    mask = torch.ones(B, Mmax, dtype=torch.bool, device=hidden.device)  # True=pad
    for b, (r, m) in enumerate(zip(rows, counts)):
        V[b, :m] = r
        mask[b, :m] = False
    return V, mask


class GREAT_L3_LoRA(nn.Module):
    """GREAT_L3 with an online (LoRA) Qwen2.5-VL instead of a cached feature map.

    `vlm` is the already-LoRA-wrapped Qwen model, built and frozen-except-LoRA in
    train.py (mirrors Hammer). The GREAT submodules are named identically to
    GREAT so a B0 checkpoint warm-starts them (strict=False); the vlm.* keys are
    simply absent from B0.
    """
    def __init__(self, vlm, vlm_dim, image_token_id, img_model_path=None, pre_train=False,
                 normal_channel=False, local_rank=None, N_p=64, emb_dim=512,
                 proj_dim=512, num_heads=4):
        super().__init__()

        self.vlm = vlm
        self.image_token_id = image_token_id

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.additional_channel = 3 if self.normal_channel else 0

        self.img_encoder = Img_Encoder()
        if pre_train and img_model_path is not None:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)
        self.affordance_dictionary_fusion = affordance_dictionary_fusion(self.emb_dim, self.proj_dim, self.num_heads)
        self.img_text_fusion = img_text_fusion(self.emb_dim, self.proj_dim)
        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.proj_dim)

        # visual-grounded reasoning chain (shared with offline L3)
        self.visual_mhacot = VisualMHACoT(vlm_dim, emb_dim=self.emb_dim, num_heads=self.num_heads)

    def encode_vlm(self, qwen_inputs):
        '''
        qwen_inputs: dict from the Qwen processor (input_ids, attention_mask,
                     pixel_values, image_grid_thw, ...), already on device.
        returns: V [B, Mmax, vlm_dim] (fp32), key_padding_mask [B, Mmax]
        '''
        out = self.vlm(**qwen_inputs, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]                       # [B, T, D]
        V, mask = _gather_image_tokens(hidden, qwen_inputs['input_ids'], self.image_token_id)
        return V.float(), mask

    def decode(self, img, xyz, V, vfeat_mask):
        '''
        img       : [B, 3, H, W]
        xyz       : [B, 3, 2048]
        V         : [B, M, vlm_dim]   image-token feature map
        vfeat_mask: [B, M] bool, True = padded key position
        returns:    (pred [B,2048,1], e_hk [B,emb], h_aff [B,emb])
        '''
        B, C, N = xyz.size()
        F_I = self.img_encoder(img)
        F_i = F_I.view(B, self.emb_dim, -1)

        F_p_wise = self.point_encoder(xyz)

        T_h, T_o, e_hk = self.visual_mhacot(V, key_padding_mask=vfeat_mask)

        T_h_, T_o_ = self.affordance_dictionary_fusion(T_h, T_o)
        I_h = self.img_text_fusion(F_i, T_h_)

        pred, h_aff = self.decoder(T_o_, I_h.permute(0, 2, 1), F_p_wise, return_feat=True)
        return pred, e_hk, h_aff

    def forward(self, img, xyz, qwen_inputs):
        '''Single-call convenience (eval): runs the vlm then decodes.'''
        V, vfeat_mask = self.encode_vlm(qwen_inputs)
        return self.decode(img, xyz, V, vfeat_mask)


def get_GREAT_L3_LoRA(vlm, vlm_dim, image_token_id, img_model_path=None, pre_train=False,
                      normal_channel=False, local_rank=None, N_p=64, emb_dim=512,
                      proj_dim=512, num_heads=4):
    model = GREAT_L3_LoRA(vlm, vlm_dim, image_token_id, img_model_path, pre_train,
                          normal_channel, local_rank, N_p, emb_dim, proj_dim, num_heads)
    return model


