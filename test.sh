
# python -u codes/test.py
# python -u codes/model.py

CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_test --do_valid \
    --data_path data/small \
    --model TransE \
    -n 2 -b 3 -d 10 \
    -g 12.0 -a 1.0 \
    -lr 1 --max_steps 15000 \
    -save models/TransE_small_0 \
    -khop 1 -nrw 3





CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train \
    --data_path data/FB15k-237 \
    --model TransE \
    -n 256 -b 1024 -d 1000 \
    -g 24.0 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save models/TransE_wn18rr_0 --test_batch_size 16 \
    -khop 3 -nrw 1000

CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test \
    --init_checkpoint models/TransE_FB15k-237_0 \
    -save models/TransE_wn18rr_0 \
    -lr 0.0001 --max_steps 100000 \
    --test_batch_size 16 \
    -khop 3 -nrw 1000
