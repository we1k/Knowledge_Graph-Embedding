CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --data_path data/wn18rr \
    --model TransE \
    -n 256 -b 1024 -d 1000 \
    -g 24.0 -a 1.0 \
    -lr 0.0001 --max_steps 99999 \
    --changing_weight 50000\
    --valid_steps 5000\
    -save models/TransE_wn18rr_3 --test_batch_size 16 \
    -khop 2 -nrw 3000

CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --data_path data/FB15k-237 \
    --model DistMult \
    -n 512 -b 1024 -d 1000 \
    -g 24.0 -a 1.0 \
    -lr 0.000004 --max_steps 80000 \
    --changing_weight 40000\
    --valid_steps 5000\
    -save models/Dist_FB15k-237_2_1000  --test_batch_size 16 \
    -khop 2 -nrw 1000
# CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
#     --do_train --do_valid --do_test \
#     --data_path data/FB15k-237 \
#     --model TransE \
#     -n 256 -b 1024 -d 1000 \
#     -g 9.0 -a 1.0 \
#     -lr 0.00005 --max_steps 99999 \
#     --changing_weight 20000\
#     --valid_steps 5000\
#     -save models/TransE_FB15k-237_3  --test_batch_size 16 \
#     -khop 2 -nrw 20
