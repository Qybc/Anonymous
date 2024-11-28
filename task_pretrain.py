import torch 
import numpy as np 
from models.nnformer.nnFormer import nnFormer
from models.losses import Med3DWith2DLoss
from mydatasets.dataset import get_pretrain_dataloader
import torch.nn as nn 
from medpy.metric.binary import dc
import argparse
import os


import clip
import open_clip


from PIL import Image


def build_model(num_classes=1, args=None):
    model = nnFormer(
        crop_size=args.spatial_size,
        embedding_dim=96,
        input_channels=1, 
        num_classes=num_classes, 
        conv_op=nn.Conv3d, 
        depths=[2,2,2,2],
        num_heads=[3, 6, 12, 24],
        patch_size=[4,4,4],
        window_size=[4,4,8,4],
        deep_supervision=False,
    )

    return model

def build_vlp_model():
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.eval()
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
    # model.eval()
    # tokenizer = open_clip.get_tokenizer('ViT-B/32')

    return model, preprocess, tokenizer

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    # 注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epoches", type=int,default=100)
    parser.add_argument("--num_classes", type=int,default=4)
    parser.add_argument("--batch_size", type=int,default=1)
    parser.add_argument("--spatial_size", type=list, nargs='+',default=[128,128,128])
    args = parser.parse_args()

    model = build_model(num_classes=args.num_classes,args=args)
    model.cuda()
    import pdb;pdb.set_trace()

    if os.path.exists('/data-pool/data/data2/qiuhui/code/Med3DInsight/ckpts_pretrain/latest_model_biomedclip_huatuo.pth'):
        model.load_state_dict(torch.load('/data-pool/data/data2/qiuhui/code/Med3DInsight/ckpts_pretrain/latest_model_biomedclip_huatuo.pth'),strict=False)

    vlp_model, vlp_preprocess, vlp_tokenizer = build_vlp_model()
    vlp_model.cuda()

    train_datalist = [
    'pretrain_huatuo',
    ]
    val_datalist = [
    'pretrain_huatuo',
    ]

    trainloader = get_pretrain_dataloader(train_datalist, batch_size=args.batch_size,shuffle=True,num_workers=0, drop_last=True)
    valloader = get_pretrain_dataloader(val_datalist, batch_size=args.batch_size,shuffle=False,num_workers=0, drop_last=False)


    loss_func = Med3DWith2DLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    
    steps_per_epoch = len(trainloader)
    steps_per_val_epoch = len(valloader)
    for epoch in range(args.epoches):
        print("epoch is " + str(epoch))
        model.train()
        data_iterator = iter(trainloader)
        step = 0
        report_loss = 0.0
        for train_iter in range(steps_per_epoch):
            optimizer.zero_grad()
            data = next(data_iterator)
            step += 1

            
            med_img = data[0].unsqueeze(1).cuda()
            img2dpath, plane, idx_slice, text2d = data[1:]
            

            image = torch.stack([vlp_preprocess(Image.open(item)) for item in img2dpath],dim=0).cuda()
            text = vlp_tokenizer(text2d).cuda() # ,truncate=True

            with torch.no_grad():
                image_embedding = vlp_model.encode_image(image)
                text_embedding = vlp_model.encode_text(text)

            outputs = model.forward_pretrain(med_img, plane, idx_slice) ## bs 512

            loss = loss_func(outputs, image_embedding, text_embedding, model.L)

            report_loss += loss.item()
            print("{}/{} step loss is {} ".format(train_iter, steps_per_epoch, str(loss.item())))
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)
            # import pdb;pdb.set_trace()
            optimizer.step()
        mean_loss = report_loss / step
        print("mean loss is " + str(mean_loss))
        
        torch.save(model.state_dict(), "./ckpts_pretrain/latest_model_biomedclip_huatuo.pth")