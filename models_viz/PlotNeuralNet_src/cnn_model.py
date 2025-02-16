import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define the architecture of the CNNModel
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input layer
    to_input("input.png", width=8, height=8, name="input"),

    # First Convolutional Layer
    to_Conv("conv1", 1, 32, offset="(0,0,0)", to="(input-east)", width=4, height=40, depth=40, caption="Conv1d 32"),
    to_connection("input", "conv1"),
    
    # Second Convolutional Layer
    to_Conv("conv2", 32, 64, offset="(1.5,0,0)", to="(conv1-east)", width=5, height=40, depth=40, caption="Conv1d 64"),
    to_connection("conv1", "conv2"),

    # Third Convolutional Layer
    to_Conv("conv3", 64, 128, offset="(1.5,0,0)", to="(conv2-east)", width=6, height=35, depth=35, caption="Conv1d 128"),
    to_connection("conv2", "conv3"),

    # Fourth Convolutional Layer
    to_Conv("conv4", 128, 128, offset="(1.5,0,0)", to="(conv3-east)", width=6, height=35, depth=35, caption="Conv1d 128"),
    to_connection("conv3", "conv4"),

    # Fifth Convolutional Layer
    to_Conv("conv5", 128, 256, offset="(2,0,0)", to="(conv4-east)", width=7, height=30, depth=30, caption="Conv1d 256"),
    to_connection("conv4", "conv5"),

    # Sixth Convolutional Layer
    to_Conv("conv6", 256, 512, offset="(2,0,0)", to="(conv5-east)", width=8, height=25, depth=25, caption="Conv1d 512"),
    to_connection("conv5", "conv6"),

    # Final Convolutional Layer (Output)
    to_Conv("conv7", 512, 1, offset="(2,0,0)", to="(conv6-east)", width=2, height=20, depth=20, caption="Conv1d 1"),
    to_connection("conv6", "conv7"),

    # Adaptive Average Pooling
    to_Pool("pool", offset="(1.5,0,0)", to="(conv7-east)", width=2, height=15, depth=15, opacity=0.5, caption="Adaptive Pool"),
    to_connection("conv7", "pool"),

    # Output
    to_SoftMax("output", 1, "(3,0,0)", "(pool-east)", width=2, caption="Output"),
    to_connection("pool", "output"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()

