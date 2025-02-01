import argparse
import sys
from natsort import natsorted
from create_figs.multi_samples_figs import run_multi_samples_figs
from create_figs.single_sample_figs import run_single_sample_figs

single_samples_figs = [f'fig{e}' for e in
                       [1, '2_weights_norm', '2_lipschitz_norm',
                        5, 7, 8, 9, '10_4samples', '10_32samples', 11, 12,
                        '14_weights_norm', '14_lipschitz_norm',
                        '15_weights_norm', '15_lipschitz_norm',
                        '16_weights_norm', '16_lipschitz_norm',
                        '17_weights_norm', '17_lipschitz_norm',
                        '18_weights_norm', '18_lipschitz_norm',
                        '26_4samples', '26_32samples',
                        '27_mnist', '27_cifar']]

multi_samples_figs = [f'fig{e}' for e in
                      ['3_lenet_mnist', '3_mlp_mnist', '3_resnet_mnist', '3_lenet_cifar', '3_mlp_cifar',
                       '3_resnet_cifar',
                       '4_lenet_mnist', '4_mlp_mnist', '4_resnet_mnist', '4_lenet_cifar', '4_mlp_cifar',
                       '4_resnet_cifar',
                       '19_lenet_mnist_2vs5', '19_lenet_mnist_2vs8', '19_lenet_mnist_0vs7', '19_lenet_mnist_0vs8',
                       '19_lenet_mnist_3vs5',
                       '20_lenet_cifar_bird_vs_ship', '20_lenet_cifar_deer_vs_truck', '20_lenet_cifar_plane_vs_frog',
                       '20_lenet_cifar_car_vs_ship', '20_lenet_cifar_horse_vs_truck',
                       '21_mlp_mnist_3vs5', '21_mlp_cifar_plane_vs_frog',
                       '22_lenet_mnist',
                       '23_lenet_mnist_0vs7', '23_lenet_mnist_0vs8', '23_lenet_mnist_3vs5',
                       '23_lenet_cifar_bird_vs_ship', '23_lenet_cifar_deer_vs_truck', '23_lenet_cifar_plane_vs_frog',
                       '24_mlp_mnist_3vs5', '24_mlp_cifar_plane_vs_frog',
                       '25_lenet_mnist']]
parser = argparse.ArgumentParser()
# Set argument with list of options, sort by natural order
parser.add_argument('fig_name', type=str, choices=natsorted(single_samples_figs + multi_samples_figs))

if __name__ == '__main__':
    fig_name = parser.parse_args().fig_name
    sys.argv = [sys.argv[0]]

    if fig_name in single_samples_figs:
        print(f'running single samples - {fig_name}')
        run_single_sample_figs(fig_name)
    elif fig_name in multi_samples_figs:
        print(f'running multi samples - {fig_name}')
        run_multi_samples_figs(fig_name)
    else:
        raise ValueError(f'fig {fig_name} not found')
