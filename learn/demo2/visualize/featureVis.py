# 3.选择需要可视化的层和通道
layer_name = 'layer3'
channel_index = 32

# 4.我们定义了一个特征可视化类，其中包括了一个损失函数，使得输出特定通道的特征图时，该通道的响应最大。
class FeatureVisualization():
    def __init__(self, model, layer_name, channel_index):
        self.model = model
        self.layer_name = layer_name
        self.channel_index = channel_index

        self.model.eval()

        self.features = None
        self.hook = self.model._modules[self.layer_name].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output[:, self.channel_index, :, :]

    def close(self):
        self.hook.remove()

    def get_activation(self, input_image):
        self.model(input_image)
        return self.features

fv = FeatureVisualization(model, layer_name, channel_index)

def loss(input_image):
    feature_map = fv.get_activation(input_image)
    return -torch.mean(feature_map)

# 5.计算梯度
def get_grads(input_image):
    input_image.requires_grad = True
    model.zero_grad()
    loss_value = loss(input_image)
    loss_value.backward()
    grads = input_image.grad.clone()
    grads /= (torch.sqrt(torch.mean(torch.square(grads))) + 1e-5)
    return loss_value, grads

# 6.进行可视化：
input_image = torch.rand((1, 3, 224, 224)).mul(20).add(128).requires_grad_(True)
step = 1.
for i in range(40):
    loss_value, grads_value = get_grads(input_image)
    input_image += grads_value * step
    if loss_value <= 0.:
        break

plt.imshow(input_image[0].permute(1, 2, 0).detach().numpy().astype('uint8'))
plt.show()