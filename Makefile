install-dependencies:
	conda install -y -c anaconda click scipy numpy pymongo scikit-learn pandas
	conda install -y -c conda-forge tqdm matplotlib
	pip install sacred seqdataloader tables keras==2.2.5 tensorflow-gpu==1.14.0
	conda install -y -c bioconda pyfaidx pybigwig

install-interpet:
	mkdir -p ~/lib
	cd ~/lib && git clone https://github.com/AvantiShri/shap.git && cd shap && pip install -e .
	cd ~/lib && git clone https://github.com/kundajelab/deeplift.git && cd deeplift && pip install -e .
	cd ~/lib && git clone https://github.com/kundajelab/tfmodisco.git && cd tfmodisco && pip install -e .
	pip install psutil
