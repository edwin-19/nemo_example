import typer

from nemo.collections.tts.models import Tacotron2Model
from nemo.collections.tts.models import HifiGanModel
import torch
import soundfile as sf

# Load Tacotron2 & vocoder
def main(
    sent:str=typer.Option("You can type your sentence here to get nemo to produce speech.")
):
    vocoder = HifiGanModel.from_pretrained(model_name="tts_hifigan")
    spec_generator = Tacotron2Model.from_pretrained("tts_en_tacotron2")
    
    with torch.no_grad():
        parsed = spec_generator.parse(sent)
        spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
        
    sf.write('sample.wav', audio.T, 22050, "PCM_16")
    
if __name__ == "__main__":
    typer.run(main)