import tritonclient.http as http_client
import numpy as np
import typer
from nemo.collections import nlp as nemo_nlp

def main(
    host:str='localhost:8000', verbose:bool=False
):
    client = http_client.InferenceServerClient(
        host, verbose=verbose
    )

    tokenizer = nemo_nlp.modules.get_tokenizer("bert-base-cased")
    
    sample_sent = 'we bought four shirts from the nvidia gear store in santa clara.'
    tokenzed_input = tokenizer.tokenizer(
        sample_sent, return_tensors='pt',
        padding='max_length', truncation=True
    )

if __name__ == "__main__":
    typer.run(main)