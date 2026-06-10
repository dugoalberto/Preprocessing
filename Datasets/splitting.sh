mkdir -p /tmp/datasets/frames/train /tmp/datasets/frames/test

# Train
while read scene; do
    if [ -d "/tmp/datasets/frames/${scene}" ]; then
        mv "/tmp/datasets/frames/${scene}" /tmp/datasets/frames/train/
        echo "train: ${scene}"
    else
        echo "MISSING: ${scene}"
    fi
done < /mnt/home/albertodugo/Projects/Preproccessing/Datasets/splits/splits/nvs_sem_train.txt

# Test
while read scene; do
    if [ -d "/tmp/datasets/frames/${scene}" ]; then
        mv "/tmp/datasets/frames/${scene}" /tmp/datasets/frames/test/
        echo "test: ${scene}"
    else
        echo "MISSING: ${scene}"
    fi
done < /mnt/home/albertodugo/Projects/Preproccessing/Datasets/splits/splits/sem_test.txt