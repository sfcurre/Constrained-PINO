python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 -Ds ds_2x64_2x128 -Dt dt_2x64_2x128 \
    -d 2 -t 64 -u 1 -p 0.02 -l 800 -c -a -b 64 -e 1000 -r 0.0002 \
    -o ./output.npy -m ./models/saved_models/

python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 -Ds ds_2x64_2x128 -Dt dt_2x64_2x128 \
    --dim 2 --tile-size 64 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 64 --epochs 1000 --learning-rate 0.0002 \
    --output ./output.npy --model-path ./models/saved_models/