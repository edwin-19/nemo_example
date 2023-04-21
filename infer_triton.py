import tritonclient.http as http_client
from tritonclient.utils import triton_to_np_dtype
import numpy as np
import typer
from nemo.collections import nlp as nemo_nlp

def main(
    host:str='localhost:8000', verbose:bool=False,
    model_name:str=typer.Option('ner_bert'), model_version:str=typer.Option('-1')
):
    client = http_client.InferenceServerClient(
        host, verbose=verbose
    )

    tokenizer = nemo_nlp.modules.get_tokenizer("bert-base-cased")
    
    sample_sent = 'we bought four shirts from the nvidia gear store in santa clara.'
    tokenzed_input = tokenizer.tokenizer(
        sample_sent, return_tensors='np',
        padding='max_length', truncation=True
    )
    
    is_model_live = client.is_model_ready(model_name)
    if is_model_live:
        metadata = client.get_model_config(model_name)
        inputs = {data['name']: {'shape': data['dims'], 'dtype': data['data_type'][5:]} for data in metadata['input']}
        
        triton_inputs = [http_client.InferInput(key, tokenzed_input[key].shape, value['dtype']) for key, value in inputs.items()]
        [data_input.set_data_from_numpy(tokenzed_input[data_input.name()], binary_data=True) for data_input in triton_inputs]
        
        output_names = [data['name'] for data in metadata['output']]
        triton_outputs = [http_client.InferRequestedOutput(name, binary_data=True) for name in output_names]
        
        results = client.infer(model_name, inputs=triton_inputs, outputs=triton_outputs)
        output = results.as_numpy(output_names[0])
        
        print(output.argmax(axis=-1)[0])

if __name__ == "__main__":
    typer.run(main)