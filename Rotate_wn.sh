CUDA_VISIBLE_DEVICES=3 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --data_path data/wn18rr \
    --model RotatE \
    -n 512 -b 1024 -d 500 \
    -g 6.0 -a 0.5 \
    -lr 0.00005 --max_steps 80000 \
    --changing_weight 60000 \
    --valid_steps 10000 \
    -save models/RotatE_wn18rr_2_1000 --test_batch_size 8 \
    -khop 2 -nrw 1000 \
    -de

CUDA_VISIBLE_DEVICES=2 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --data_path data/wn18rr \
    --model RotatE \
    -n 1024 -b 256 -d 1000 \
    -g 6.0 -a 0.5 \
    -lr 0.00005 --max_steps 70000 \
    --changing_weight 60000 \
    --valid_steps 10000 \
    -save models/RotatE_wn18rr_3_1500 --test_batch_size 8 \
    -khop 3 -nrw 1500 \
    -de
