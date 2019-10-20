install-dependencies:
	conda install -c anaconda click scipy numpy pymongo scikit-learn pandas
	conda install -c conda-forge tqdm matplotlib tiledb
	pip install sacred seqdataloader tables keras==2.2.*
	conda install pytorch torchvision cudatoolkit=10.0 -c pytorch h5py
	conda install -c bioconda pyfaidx pybigwig
