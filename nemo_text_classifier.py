import typer

from nemo.collections.nlp.models.text_classification import TextClassificationModel
import wget
import os
from omegaconf import OmegaConf

def get_config():
    MODEL_CONFIG = "text_classification_config.yaml"
    CONFIG_DIR =  './configs/'
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_DIR + MODEL_CONFIG):
        print('Downloading config file...')
        wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/nlp/text_classification/conf/' + MODEL_CONFIG, CONFIG_DIR)
        print('Config file downloaded!')
    else:
        print ('config file already exists')
        
    return f'./configs/{MODEL_CONFIG}'

def main():
    config_path = get_config()
    config = OmegaConf.load(config_path)
    config.model.language_model.pretrained_model_name = "bert-base-uncased"
    config.model.dataset.num_classes=2

    model = TextClassificationModel(cfg=config.model)
    # define the list of queries for inference
    queries = ['by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .', 
            'director rob marshall went out gunning to make a great one .', 
            'uneasy mishmash of styles and genres .']
    
    labels = ['negative', 'positive']
    # max_seq_length=512 is the maximum length BERT supports.       
    results = model.classifytext(queries=queries, batch_size=3, max_seq_length=512)
    for res, qu in zip(results, queries):
        print('Query: {}, Label: {}'.format(qu, labels[res]))

if __name__ == "__main__":
    typer.run(main)