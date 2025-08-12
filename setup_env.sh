# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, installing Miniconda..."
    # Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p $HOME/miniconda && echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
    conda init
    source ~/.bashrc
fi

# Create and activate conda environment
conda create -y --name llmcompass_ae python=3.9
conda activate llmcompass_ae

# Install Python Dpendencies
pip3 install scalesim
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
pip3 install matplotlib
pip3 install seaborn
pip3 install scipy
pip3 install pynvml

# Clone the repository
git clone https://github.com/fangzhonglyu/LLMCompass.git
cd LLMCompass
git submodule init
git submodule update --recursive
git checkout benchmark
git pull

# Enable persistent GPU mode
nvidia-smi -pm 1