data_path="/home/trangndp/projects/trading_bot/dataset"
device="cuda"

python -u run.py \
    --training 0 \
    --data_path /data/dataset/ \
    --data_name XAUUSD_M5 \
    --checkpoint /data/model/ \
    --log_path /data/log/ \
    --result_path /data/export/img/ \
    --device $device
