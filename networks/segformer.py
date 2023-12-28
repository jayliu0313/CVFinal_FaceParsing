from transformers import TFSegformerForSemanticSegmentation, SegformerConfig
import torch

def Segformer(num_classes=19):
    configuration = SegformerConfig(num_labels=num_classes)
    model = TFSegformerForSemanticSegmentation(configuration)
    return model

if __name__ == '__main__':

    net = Segformer()
    data = torch.rand((1, 3, 256, 256))
    out = net(data)
    print(out[0][-1].shape)
