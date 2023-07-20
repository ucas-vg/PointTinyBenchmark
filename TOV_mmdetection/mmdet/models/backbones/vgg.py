from mmcv.cnn import VGG
from ..builder import BACKBONES


@BACKBONES.register_module()
class MyVGG(VGG):
    def __init__(self,
                 pretrained=None,
                 *args,
                 **kwargs):
        self.pretrained = pretrained
        super().__init__(*args, **kwargs)

    def init_weights(self, pretrained=None):
        super().init_weights(self.pretrained)

    def forward(self, x):

        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)

        return tuple(outs)