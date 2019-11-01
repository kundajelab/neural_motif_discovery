install-dependencies:
	conda install -y -c anaconda click scipy numpy pymongo scikit-learn pandas
	conda install -y -c conda-forge tqdm matplotlib tiledb
	pip install sacred seqdataloader tables keras==2.2.* tensorflow-gpu==1.14
	conda install -y -c bioconda pyfaidx pybigwig
