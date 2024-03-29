# Beyond Active Learning: Leveraging the Full Potential of Human interaction via Auto-Labeling, Human Correction, and Human Verification

This repository contains the code artifacts for the experiments contained within the paper ["Beyond Active Learning: Leveraging the Full Potential of Human Interaction via Auto-Labeling, Human Correction, and Human Verification"](https://arxiv.org/abs/2306.01277).

# Setup

The [experiment script](new_smi_autolabeling.py) requires the use of DISTIL and its dependencies, which is detailed on [DISTIL's homepage](https://github.com/decile-team/distil). We alternatively use a different version for one of DISTIL's dependencies, which can be installed as follows:

	git clone https://github.com/decile-team/submodlib.git
	git checkout oom_fix
	pip install -r requirements.txt
	python setup.py bdist_wheel
	pip install .

# Running

The arguments required by the script are detailed in the supplementary material of the paper. An example is given here:

	python new_smi_autolabeling.py --dataset=cifar100 --al_strategy=badge --human_correct_strategy=logdetmi --auto_assign_strategy=highest_confidence --b1=1000 --b2=2000 --b3=3000 --seed_size=10000 --rounds=7 --runs=3 --device=0

# Labeling Experiments

Please refer to our [modified labeling tool](https://github.com/krishnatejakk/labelImg) for its code artifacts.