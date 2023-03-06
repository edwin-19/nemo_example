import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import typer
import os

def main(
    outdir:str=typer.Option('onnx_models/'),
    quart_name:str=typer.Option('quartz.onnx'),
    bert_name:str=typer.Option('bert_ner.onnx')
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    print('Exporting ONNX model')
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    quartz_export_path = os.path.join(outdir, quart_name)
    quartznet.export(quartz_export_path)
    
    bert_export_path = os.path.join(outdir, bert_name)
    pretrained_ner_model = nemo_nlp.models.TokenClassificationModel.from_pretrained(model_name="ner_en_bert")
    pretrained_ner_model.export(bert_export_path)

if __name__ == '__main__':
    typer.run(main)