# LiSum
The official repository of LiSum

## Questionnaires and responses
You can see the contents of the quesionnaires and the responses in the folder '/Questionnaires and answers'. As the contents of two questionniares are consistent across languages, we simply upload the survey questions of English version.

## Datasets
You can find the referenced dataset of summarization and classification in '/Dataset/datasets.json'. The format of the .json file is illustrated as follows:
```
{
	"license_title": 'The title of the license',
	"quick_summary": 'The quick summary of the license',
	"can_list": [{
			"name": 'An action item',
			"description": 'An explanation of the action item'
		  	},
		...],
	"cannot_list": [],
	"must_list": [],
	"fulltext": 'The fulltext of the license"
}
```

## Requirements
We provide a .yaml file in the 'Code' folder to help you create the environment needed to train and evaluate LiSum. You can run the following command to create a new conda environment and automatically install required dependencies.
```
conda env create -f lisum.yaml
```

## The replication package
We provide our code in the '/Code' folder. To reimplement our results illustrated in the paper, you can run the following command.
```
python FinetuneUnifiedTEConcat.py --classification_loss_weight 0.7
```

We also provide LiSum-Base's checkpoints of three folds respectively. You can download these checkpoints from https://drive.google.com/file/d/19TA1PEP-bZYhmTBBHlflc0HU7JbyHvTn/view?usp=sharing.


## License
Our codes follow MIT License.
Our datasets follow Computational Use of Data Agreement (C-UDA) License.