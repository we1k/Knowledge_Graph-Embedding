CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --init_checkpoint models/Dist_FB15k-237_4_3000 \
    --changing_weight 50000\
    --valid_steps 10000\
    -save models/Dist_FB15k-237_4_3000  --test_batch_size 16 \
    -khop 4 -nrw 3000 \
    -r 0.00001

CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid --do_test\
    --init_checkpoint models/Dist_FB15k-237_5_4000 \
    --changing_weight 50000\
    --valid_steps 10000\
    -save models/Dist_FB15k-237_5_4000  --test_batch_size 16 \
    -khop 5 -nrw 4000 \
    -r 0.0001