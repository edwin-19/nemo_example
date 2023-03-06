import onnxruntime as ort
import typer

def main(
    onnx_model_path:str=typer.Option('./triton_models/ner_bert/1/model.onnx'),
):
    onnx_sess = ort.InferenceSession(onnx_model_path)
    input_data = onnx_sess.get_inputs()
    output_data = onnx_sess.get_outputs()
    
    print('Input Data')
    for data in input_data:
        print('Name: {}, Shape: {}, Dtype: {}'.format(data.name, data.shape, data.type))

    print('Output data')        
    for data in output_data:
        print('Name: {}, Shape: {}, Dtype: {}'.format(data.name, data.shape, data.type))
    
if __name__ == '__main__':
    typer.run(main)