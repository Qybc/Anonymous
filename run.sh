# stage1
export CUDA_VISIBLE_DEVICES=3;python3 task_pretrain.py --batch_size 10


# stage2
# export CUDA_VISIBLE_DEVICES=0;python3 task_seg_nnformer2.py --batch_size 1 --dataset MM-WHS
# export CUDA_VISIBLE_DEVICES=1;python3 task_seg_nnformer2.py --batch_size 4 --dataset CHAOS
# export CUDA_VISIBLE_DEVICES=0;python3 task_seg_nnformer2.py --batch_size 6 --dataset OASIS1
# # 以上3个是原先的

# export CUDA_VISIBLE_DEVICES=0;python3 task_seg_nnformer2.py --batch_size 6 --dataset VS
# export CUDA_VISIBLE_DEVICES=1;python3 task_seg_nnformer2.py --batch_size 6 --dataset LITS
# export CUDA_VISIBLE_DEVICES=2;python3 task_seg_nnformer2.py --batch_size 6 --dataset ADNI4
# export CUDA_VISIBLE_DEVICES=0;python3 task_seg_nnformer2.py --dataset ABDOMENCT-1K
# export CUDA_VISIBLE_DEVICES=1;python3 task_seg_nnformer2.py --dataset MSD-HEART

# export CUDA_VISIBLE_DEVICES=2;python3 task_seg_nnformer2.py --batch_size 6 --dataset ACDC # 没效果，不跑这个
# export CUDA_VISIBLE_DEVICES=0;python3 task_seg_nnformer2.py --batch_size 4 --dataset ADNI35 # bs大不起来，还是4分类吧
# export CUDA_VISIBLE_DEVICES=0;python3 task_seg_nnformer2.py --batch_size 6 --dataset SCGM # 有一个plane才3张，不好建模


# visualization
# export CUDA_VISIBLE_DEVICES=2;python3 task_vis.py --batch_size 16
