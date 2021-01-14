CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --data_path data/FB15k-237 \
    --model TransE \
    -n 128 -b 1024 -d 1000 \
    -g 12.0 -a 1.0 \
    -lr 0.0001 --max_steps 99999 \
    --valid_steps 10000 \
    -save models/TransE_FB15k-237_5_4000 --test_batch_size 16 \
    -khop 5 -nrw 4000 \
