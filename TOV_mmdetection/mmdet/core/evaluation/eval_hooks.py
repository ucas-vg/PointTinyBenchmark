import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    # add by hui ##########################################################
    def __init__(self, *args, **eval_kwargs):
        self.do_first_eval = eval_kwargs.pop('do_first_eval', False)
        self.is_run_first = True
        self.do_final_eval = eval_kwargs.pop('do_final_eval', False)
        self.do_eval = False
        super(EvalHook, self).__init__(*args, **eval_kwargs)

    def before_run(self, runner):
        if self.do_first_eval and self.is_run_first:
            self.is_run_first = False
            self.do_eval = True
            self._do_evaluate(runner)
        super(EvalHook, self).before_run(runner)

    def _should_evaluate(self, runner):
        if self.do_eval:
            self.do_eval = False
            return True
        return super(EvalHook, self)._should_evaluate(runner)

    def after_run(self, runner):
        if self.do_final_eval:
            self.do_eval = True
            self._do_evaluate(runner)
        super(EvalHook, self).after_run(runner)
    #########################################################################

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(BaseDistEvalHook):
    # add by hui ##########################################################
    def __init__(self, *args, **eval_kwargs):
        self.do_first_eval = eval_kwargs.pop('do_first_eval', False)
        self.is_run_first = True
        self.do_final_eval = eval_kwargs.pop('do_final_eval', False)
        self.do_eval = False
        super(DistEvalHook, self).__init__(*args, **eval_kwargs)

    def before_run(self, runner):
        if self.do_first_eval and self.is_run_first:
            self.is_run_first = False
            self.do_eval = True
            self._do_evaluate(runner)
        super(DistEvalHook, self).before_run(runner)

    def _should_evaluate(self, runner):
        if self.do_eval:
            self.do_eval = False
            return True
        return super(DistEvalHook, self)._should_evaluate(runner)

    def after_run(self, runner):
        if self.do_final_eval:
            self.do_eval = True
            self._do_evaluate(runner)
        super(DistEvalHook, self).after_run(runner)
    #########################################################################

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
