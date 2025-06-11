
cd TOV_mmdetection && source activate open-mmlab-hui


# clear invalid log_dir and copy valid log to ./log dir
python exp/tools/sync_log.py

# clear pth
python exp/tools/clear_tmp_pth.py ../TOV_mmdetection_cache/work_dir/
