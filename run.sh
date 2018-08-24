CUDA_VISIBLE_DEVICES=0,1,2,3   python main.py --gpu 4 --dataroot './dataset' --result './result'  --h 448 --w 448  --nclass 200 --train_batch_size 16 --test_batch_size 32 --lr 0.1 --epochs 1000
