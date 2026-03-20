from wer import jiwer_wer
from encoder.audioEncoder import AudioEncoder

class Evaluate():
    def __init__(self, model, tokenizer, dataloader):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader

    def evaluate(self, audio_input):
        WERs = []
        for batch in self.dataloader:
            audio_input = self.AudioEncoder(batch['audio'])
            original_transcriptions = batch['transcription']
            output = self.model(audio_input)
            output_transcriptions = self.postprocess_output(output)
            # Compute WER for each pair of original and output transcriptions
            for orig, out in zip(original_transcriptions, output_transcriptions):
                wer = jiwer_wer(orig, out)
                WERs.append(wer)
        average_WER = sum(WERs) / len(WERs) if WERs else 0
        return average_WER