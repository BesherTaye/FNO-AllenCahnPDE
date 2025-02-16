import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the architecture of FNO2_1d
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input layer (2 channels: u0(x), x)
    to_input("input.png", width=8, height=8, name="input"),

    # Initial linear layer to lift input dimensions
    to_Conv("linear_p", 2, 64, offset="(0,0,0)", to="(input-east)", width=3, height=40, depth=40, caption="LinProj"),
    to_connection("input", "linear_p"),

    # Fourier Convolution Layer 1
    to_Conv("spect1", 64, 64, offset="(1.5,0,0)", to="(linear_p-east)", width=5, height=40, depth=40, caption="SpectralConv"),
    to_connection("linear_p", "spect1"),
    to_Conv("lin0", 64, 64, offset="(1.5,0,0)", to="(spect1-east)", width=3, height=40, depth=40, caption="Conv1d"),
    to_connection("spect1", "lin0"),
    to_SoftMax("tanh1", 64, offset="(1.5,0,0)", to="(lin0-east)", width=2, caption="Tanh"),
    to_connection("lin0", "tanh1"),

    # Fourier Convolution Layer 2
    to_Conv("spect2", 64, 64, offset="(2,0,0)", to="(tanh1-east)", width=5, height=35, depth=35, caption="SpectralConv"),
    to_connection("tanh1", "spect2"),
    to_Conv("lin1", 64, 64, offset="(1.5,0,0)", to="(spect2-east)", width=3, height=35, depth=35, caption="Conv1d"),
    to_connection("spect2", "lin1"),
    to_SoftMax("tanh2", 64, offset="(1.5,0,0)", to="(lin1-east)", width=2, caption="Tanh"),
    to_connection("lin1", "tanh2"),

    # Fourier Convolution Layer 3
    to_Conv("spect3", 64, 64, offset="(2,0,0)", to="(tanh2-east)", width=5, height=30, depth=30, caption="SpectralConv"),
    to_connection("tanh2", "spect3"),
    to_Conv("lin2", 64, 64, offset="(1.5,0,0)", to="(spect3-east)", width=3, height=30, depth=30, caption="Conv1d"),
    to_connection("spect3", "lin2"),
    to_SoftMax("tanh3", 64, offset="(1.5,0,0)", to="(lin2-east)", width=2, caption="Tanh"),
    to_connection("lin2", "tanh3"),

    # Fourier Convolution Layer 4
    to_Conv("spect4", 64, 64, offset="(2,0,0)", to="(tanh3-east)", width=5, height=25, depth=25, caption="SpectralConv"),
    to_connection("tanh3", "spect4"),
    to_Conv("lin3", 64, 64, offset="(1.5,0,0)", to="(spect4-east)", width=3, height=25, depth=25, caption="Conv1d"),
    to_connection("spect4", "lin3"),
    to_SoftMax("tanh4", 64, offset="(1.5,0,0)", to="(lin3-east)", width=2, caption="Tanh"),
    to_connection("lin3", "tanh4"),

    # Fourier Convolution Layer 5
    to_Conv("spect5", 64, 64, offset="(2,0,0)", to="(tanh4-east)", width=5, height=20, depth=20, caption="SpectralConv"),
    to_connection("tanh4", "spect5"),
    to_Conv("lin4", 64, 64, offset="(1.5,0,0)", to="(spect5-east)", width=3, height=20, depth=20, caption="Conv1d"),
    to_connection("spect5", "lin4"),
    to_SoftMax("tanh5", 64, offset="(1.5,0,0)", to="(lin4-east)", width=2, caption="Tanh"),
    to_connection("lin4", "tanh5"),

    # Final linear transformations
    to_Conv("linear_q", 64, 32, offset="(2,0,0)", to="(tanh5-east)", width=3, height=15, depth=15, caption="Linear Layer"),
    to_connection("tanh5", "linear_q"),
    to_Conv("output_layer", 32, 1, offset="(2,0,0)", to="(linear_q-east)", width=2, height=10, depth=10, caption="Output Layer"),
    to_connection("linear_q", "output_layer"),

    # Output
    to_SoftMax("output", 1, "(3,0,0)", "(output_layer-east)", width=2, caption="Solution"),
    to_connection("output_layer", "output"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()

