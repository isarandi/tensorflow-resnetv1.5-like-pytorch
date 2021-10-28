# Porting ResNetv1.5 from PyTorch to TensorFlow

"ResNetV1.5" is a slight modification of ResNetV1. In V1, the striding in the bottleneck blocks happens on the 1x1 conv before the 3x3, while in V1.5 the stride is on the 3x3 conv. When PyTorch talks about "ResNet" as such, they mean V1.5, while TensorFlow means V1.

Run `python convert_weights.py` to generate Keras-compatible HDF5 checkpoint files with the weights from the corresponding PyTorch checkpoints.
