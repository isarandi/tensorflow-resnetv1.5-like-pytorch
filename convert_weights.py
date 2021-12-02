#!/usr/bin/env python3
import hashlib
import os
import re
import shutil
import numpy as np
import h5py
import tensorflow as tf
import torch


def main():
    pytorch_root = 'https://download.pytorch.org/models'
    tf_root = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet'
    model_dir = os.environ['HOME'] + '/.keras/models'

    # ResNets with no top layer (no fully-connected classifier at the end)
    convert(
        url_pytorch=f'{pytorch_root}/resnet18-5c106cde.pth',
        url_tf=f'{tf_root}/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        out_path=f'{model_dir}/resnet18_weights_tf_dim_ordering_tf_kernels_notop.h5',
        hash_pytorch='5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8',
        hash_tf_in='4d473c1dd8becc155b73f8504c6f6626',
        hash_tf_out_expected='cd9aca5b625298765956a04230be071a')

    convert(
        url_pytorch=f'{pytorch_root}/resnet34-333f7ec4.pth',
        url_tf=f'{tf_root}/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        out_path=f'{model_dir}/resnet34_weights_tf_dim_ordering_tf_kernels_notop.h5',
        hash_pytorch='333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef',
        hash_tf_in='4d473c1dd8becc155b73f8504c6f6626',
        hash_tf_out_expected='5d0432fa0b4d5bf5fd88f04151f590a4')

    convert(
        url_pytorch=f'{pytorch_root}/resnet50-0676ba61.pth',
        url_tf=f'{tf_root}/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        out_path=f'{model_dir}/resnet50v1_5_weights_tf_dim_ordering_tf_kernels_notop.h5',
        hash_pytorch='0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a',
        hash_tf_in='4d473c1dd8becc155b73f8504c6f6626',
        hash_tf_out_expected='315b92000a86ce737f460441071d7579')

    convert(
        url_pytorch=f'{pytorch_root}/resnet101-63fe2227.pth',
        url_tf=f'{tf_root}/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5',
        out_path=f'{model_dir}/resnet101v1_5_weights_tf_dim_ordering_tf_kernels_notop.h5',
        hash_pytorch='63fe2227b86e8f1f2063f43a75c84d195911b6a0eace650907dd3dc62dd49a0a',
        hash_tf_in='88cf7a10940856eca736dc7b7e228a21',
        hash_tf_out_expected='0b87f84107ae1a0616f76d028781b6a6')

    convert(
        url_pytorch=f'{pytorch_root}/resnet152-394f9c45.pth',
        url_tf=f'{tf_root}/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5',
        out_path=f'{model_dir}/resnet152v1_5_weights_tf_dim_ordering_tf_kernels_notop.h5',
        hash_pytorch='394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58',
        hash_tf_in='ee4c566cf9a93f14d82f913c2dc6dd0c',
        hash_tf_out_expected='471a7a36f82f50879a64731f1615f2df')

    # ResNets with the top layer (fully-connected classifier at the end)
    convert(
        url_pytorch=f'{pytorch_root}/resnet18-5c106cde.pth',
        url_tf=f'{tf_root}/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        out_path=f'{model_dir}/resnet18_weights_tf_dim_ordering_tf_kernels.h5',
        hash_pytorch='5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8',
        hash_tf_in='2cb95161c43110f7111970584f804107',
        hash_tf_out_expected='a04f614a6c28f19f9e766a22a65d87d7')

    convert(
        url_pytorch=f'{pytorch_root}/resnet34-333f7ec4.pth',
        url_tf=f'{tf_root}/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        out_path=f'{model_dir}/resnet34_weights_tf_dim_ordering_tf_kernels.h5',
        hash_pytorch='333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef',
        hash_tf_in='2cb95161c43110f7111970584f804107',
        hash_tf_out_expected='25351c4102513ba73866398dfda04546')

    convert(
        url_pytorch=f'{pytorch_root}/resnet50-0676ba61.pth',
        url_tf=f'{tf_root}/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        out_path=f'{model_dir}/resnet50v1_5_weights_tf_dim_ordering_tf_kernels.h5',
        hash_pytorch='0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a',
        hash_tf_in='2cb95161c43110f7111970584f804107',
        hash_tf_out_expected='595763ceca1995bf6e34ccd730b81741')

    convert(
        url_pytorch=f'{pytorch_root}/resnet101-63fe2227.pth',
        url_tf=f'{tf_root}/resnet101_weights_tf_dim_ordering_tf_kernels.h5',
        out_path=f'{model_dir}/resnet101v1_5_weights_tf_dim_ordering_tf_kernels.h5',
        hash_pytorch='63fe2227b86e8f1f2063f43a75c84d195911b6a0eace650907dd3dc62dd49a0a',
        hash_tf_in='f1aeb4b969a6efcfb50fad2f0c20cfc5',
        hash_tf_out_expected='b16e80439827b6abfb2c378ac434fd45')

    convert(
        url_pytorch=f'{pytorch_root}/resnet152-394f9c45.pth',
        url_tf=f'{tf_root}/resnet152_weights_tf_dim_ordering_tf_kernels.h5',
        out_path=f'{model_dir}/resnet152v1_5_weights_tf_dim_ordering_tf_kernels.h5',
        hash_pytorch='394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58',
        hash_tf_in='100835be76be38e30d865e96f2aaae62',
        hash_tf_out_expected='2e445ecb46e5d72aa0004b51f668623c')


def convert(url_pytorch, url_tf, out_path, hash_pytorch, hash_tf_in,
            hash_tf_out_expected=None):
    tf_path_in = tf.keras.utils.get_file(origin=url_tf, file_hash=hash_tf_in, cache_dir='/tmp')
    shutil.copy2(tf_path_in, out_path)

    torch_path = tf.keras.utils.get_file(
        origin=url_pytorch, file_hash=hash_pytorch, cache_dir='/tmp')
    ckpt_torch = torch.load(torch_path)

    with h5py.File(out_path, mode='r+') as out_h5:
        needed_conv_prefixes = set()

        for name, value in ckpt_torch.items():
            if name.startswith('fc.') and 'notop' in url_tf:
                continue
            value = value.detach().numpy()
            h5_name = name_pytorch_to_h5(name)
            if h5_name.endswith('kernel:0'):
                if h5_name.startswith('probs'):
                    value = value.transpose([1, 0])  # c_out c_in -> c_in c_out
                else:
                    value = value.transpose([2, 3, 1, 0])  # c_out c_in w h -> w h c_in c_out
            set_h5_value(out_h5, h5_name, value)

            if h5_name.startswith('conv'):
                main_part = h5_name.split('/')[0]
                subparts = main_part.split('_')
                layer_prefix = '_'.join(subparts[:-1])
                needed_conv_prefixes.add(layer_prefix)
                if len(subparts) > 2:
                    if subparts[-2].isdigit():
                        other_prefix = '_'.join(subparts[:-2])
                        needed_conv_prefixes.add(other_prefix)

        if 'resnet34' in url_pytorch or 'resnet18' in url_pytorch:
            # Delete unneeded layers
            layer_names_old = [x.decode('ascii') for x in out_h5.attrs['layer_names']]
            layer_names_new = [n for n in layer_names_old if
                               is_layer_needed(n, needed_conv_prefixes)
                               and not n.endswith('_2_relu')]
            for unneeded_layer in sorted(set(layer_names_old) - set(layer_names_new)):
                del out_h5[unneeded_layer]

            # Rewrite layer_names attribute
            del out_h5.attrs['layer_names']
            layer_names_new = resort_layer_names(layer_names_new)
            out_h5.attrs['layer_names'] = np.array([x.encode('ascii') for x in layer_names_new])

            # Delete bias values
            for layer_name in layer_names_new:
                if layer_name.startswith('conv'):
                    # Rewrite weight_names attribute
                    out_h5[layer_name].attrs['weight_names'] = [
                        w for w in out_h5[layer_name].attrs['weight_names']
                        if not w.endswith(b'bias:0')]
                    bias_name = f'{layer_name}/{layer_name}/bias:0'
                    if bias_name in out_h5:
                        del out_h5[bias_name]

    hash_tf_out = get_md5(out_path)
    if hash_tf_out_expected is not None:
        print('Hash OK' if hash_tf_out == hash_tf_out_expected else 'Hash error')
    print(out_path, hash_tf_out)


def get_prefix(name):
    main_part = name.split('/')[0]
    subparts = main_part.split('_')
    return '_'.join(subparts[:-1])


def is_layer_needed(layer_name, needed_conv_prefixes):
    return (not layer_name.startswith('conv')
            or get_prefix(layer_name) in needed_conv_prefixes)


def resort_layer_names(layer_names):
    # The order of layer names must be reordered, else we can't load the checkpoint
    # Keras' h5 checkpoint is based on the order of the layers
    pattern = r'^(?P<prefix>.+)_(?P<suffix>(0|2)_(conv|bn))$'
    ends = ['0_conv', '2_conv', '0_bn', '2_bn']
    result = []
    i = 0
    while i <= len(layer_names) - 4:
        name_group = [layer_names[i + j] for j in range(4)]
        matches = [re.match(pattern, name) for name in name_group]
        if all(matches) and all(m['prefix'] == matches[0]['prefix'] for m in matches):
            prefix = matches[0]['prefix']
            result += [f'{prefix}_{e}' for e in ends]
            i += 4
        else:
            result.append(layer_names[i])
            i += 1
    result += layer_names[-3:]
    return result


def set_h5_value(out_h5, h5_name, value):
    if h5_name in out_h5 and out_h5[h5_name][:].shape == value.shape:
        out_h5[h5_name][:] = value
    else:
        if h5_name in out_h5:
            del out_h5[h5_name]
        out_h5.create_dataset(h5_name, data=value, track_times=False)


def name_pytorch_to_h5(name):
    bn_dict = dict(
        weight='gamma', bias='beta', running_mean='moving_mean', running_var='moving_variance')
    conv_dict = dict(weight='kernel', bias='bias')

    parts = name.split('.')
    out = []
    part = parts.pop(0)
    if part == 'fc':
        out.append('probs')
        name_dict = conv_dict
    elif part == 'conv1':
        out += ['conv1', 'conv']
        name_dict = conv_dict
    elif part == 'bn1':
        out += ['conv1', 'bn']
        name_dict = bn_dict
    else:
        tf_block = int(part[5])
        tf_conv = int(parts.pop(0))
        out += [f'conv{tf_block + 1}', f'block{tf_conv + 1}']
        part = parts.pop(0)
        if part == 'downsample':
            out.append('0')
            part = parts.pop(0)
            if part == '0':
                out.append('conv')
                name_dict = conv_dict
            else:
                out.append('bn')
                name_dict = bn_dict
        else:
            out.append(str(int(part[-1])))
            if part.startswith('bn'):
                out.append('bn')
                name_dict = bn_dict
            else:
                out.append('conv')
                name_dict = conv_dict

    folder = '_'.join(out)
    lastpart = name_dict[parts.pop(0)]
    return f'{folder}/{folder}/{lastpart}:0'


def get_md5(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


if __name__ == '__main__':
    main()
