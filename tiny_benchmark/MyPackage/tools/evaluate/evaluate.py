import os


def re_evaluate_all(model_dir, excute_cmd):
    log_file = os.path.join(model_dir, 'last_checkpoint')
    log_data = open(log_file).read()

    iter_ids = []
    for model_file in os.listdir(model_dir):
        if model_file.startswith('model_'):
            iter_id = model_file.split('.')[0].split('_')[1]
            if iter_id != 'final':
                iter_str_len = len(iter_id)
                iter_ids.append(int(iter_id))

    iter_ids = sorted(iter_ids)
    if os.path.exists(os.path.join(model_dir, 'model_final.pth')):
        iter_ids.append('final')

    for iter_id in iter_ids:
        f = os.path.join('.', model_dir, 'model_{}.pth'.format(str(iter_id).zfill(iter_str_len)))
        fp = open(log_file, 'w')
        fp.write(f)
        fp.close()
        if iter_id != 'final':
            return_id = os.system('{} SOLVER.TEST_ITER {} SOLVER.MAX_ITER {}'.format(excute_cmd, iter_id+2, iter_id+2))
        else:
            return_id = os.system('{}'.format(excute_cmd))
        if return_id != 0:
                break

    fp = open(log_file, 'w')
    fp.write(log_data)
    fp.close()


if __name__ == '__main__':
    os.chdir('../')
    re_evaluate_all('outputs/cityperson/FPN/base',
                    'export LD_LIBRARY_PATH=/home/hui/ide/miniconda3/envs/torch100/lib/:$LD_LIBRARY_PATH && '
                    'CUDA_VISIBLE_DEVICES=1 /home/hui/ide/miniconda3/envs/torch100/bin/python'
                    ' tools/train_test_net.py --config-file configs/cityperson/e2e_faster_rcnn_R_50_FPN.yaml')
                    # ' log/citypersons/log_citypersons_FPN_result.log')
