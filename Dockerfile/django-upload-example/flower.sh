
echo $1

#cd /mnt/edward2/training/tensorflow-for-poets-2/
#python -m scripts.label_image     --graph=tf_files/retrained_graph.pb      --image=$1

cd /mnt/edward2/training/tensorflow-for-poets-2; python -m /mnt/edward2/training/tensorflow-for-poets-2/scripts/label_image.py     --graph=tf_files/retrained_graph.pb      --image=/mnt/edward2/datasets/flower_photos/roses/14001990976_bd2da42dbc.jpg

#--image=/mnt/edward2/datasets/flower_photos/roses/10090824183_d02c613f10_m.jpg
