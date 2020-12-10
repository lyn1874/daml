#!/bin/bash
download_ckpt=${1?Error: Do you want to download the checkpoints?}
trap "exit" INT
pip install tensorflow-gpu==1.13.1
yes Y | conda install cudatoolkit=10.0

if [ $download_ckpt = true ]; then
    checkpath=checkpoints/
    if [ -d "${checkpath}" ]; then
        echo "next, download model ckpts"
    else
        mkdir $checkpath
    fi
    cd $checkpath
    echo "Download the checkpoint for ucsd1"
    wget https://cloud.ilabt.imec.be/index.php/s/KE6THz68drZdPgF/download -O ucsd1.zip
    mkdir ano_ucsd1_motion_end2end
    unzip -d ano_ucsd1_motion_end2end/ ucsd1.zip
    rm ucsd1.zip
    echo "Download the checkpoint for ucsd2"
    wget https://cloud.ilabt.imec.be/index.php/s/4gwEd6aRTMf3cLx/download -O ucsd2.zip 
    mkdir ano_ucsd2_motion_end2end
    unzip -d ano_ucsd2_motion_end2end/ ucsd2.zip
    rm ucsd2.zip
    cd ..
fi

