CUDA_VISIBLE_DEVICES=1 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test \
    --data_path data/wn18rr \
    --model TransE \
    -n 512 -b 1024 -d 500 \
    -g 6.0 -a 0.5 \
    -lr 0.00005 --max_steps 60000 \
    --changing_weight 40000 \
    -save models/TransE_wn18rr_2_1000 --test_batch_size 8 \
    -khop 2 -nrw 1000

CUDA_VISIBLE_DEVICES=1 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test \
    --data_path data/wn18rr \
    --model TransE \
    -n 512 -b 1024 -d 500 \
    -g 6.0 -a 0.5 \
    -lr 0.00005 --max_steps 60000 \
    --changing_weight 40000 \
    -save models/TransE_wn18rr_3_2000 --test_batch_size 8 \
    -khop 3 -nrw 2000