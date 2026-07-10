build:
	conda install -c conda-forge cuda-toolkit -y
	conda install conda-forge::cudnn -y
	conda install conda-forge::tensorflow-gpu -y
	conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
	pip install -r requirements.txt
	pip install -e .