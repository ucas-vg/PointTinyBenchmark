from mmcv.utils import Registry, build_from_cfg

POINT_SAMPLERS = Registry('point_sampler')


def build_point_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, POINT_SAMPLERS, default_args)


class BasePointSampler(object):
    pass


class CirclePointSampler(object):
    def __call__(self, gts):
        pass
