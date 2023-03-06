import nemo.collections.asr as nemo_asr
import jiwer

if __name__ == '__main__':
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    gt_text = 'complain that all we have done is unnecessary and subject to great risks'
    
    transribe_text = quartznet.transcribe(['sample/sample.wav'])
    
    cer = jiwer.cer(gt_text, transribe_text)
    print('Error Rate: {}'.format(cer))
    print(transribe_text)