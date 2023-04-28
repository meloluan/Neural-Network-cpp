neural-network-cpp
==================
A rock solid project with enforced style, testing and static analysis

[![Build Status]
    (https://travis-ci.org/meloluan/neural-network-cpp.svg)]
    (https://travis-ci.org/meloluan/neural-network-cpp)

Open a Terminal and make sure that your system is updated:

1 sudo apt update
2 sudo apt upgrade

Next, we’ll install Armadillo prerequisites:

1 sudo apt install cmake libopenblas-dev liblapack-dev

Theoretically, you can install Armadillo using the apt package manager, but this is not recommended because the version provided by apt is really old. I suggest to download and extract the latest stable release of Armadillo. Once you have Armadillo extracted in your Downloads folder, you can build and install the library with:

1 cd Downloads/
2 cd arma*
3 cmake .
4 make
5 sudo make install