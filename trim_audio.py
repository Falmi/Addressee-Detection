import os
from scipy.io import wavfile

if __name__=="__main__":
    
    csv_path="data/E_MUMMER_csv/all_E_MUMMER_dataset.csv"
    audio_path="data/audio/"
    out_put_audio="data/croped_E_MUMMER_audio"

    if not os.path.exists(out_put_audio):
         os.makedirs(out_put_audio, exist_ok=True)

    count=0
    with open(csv_path, "r") as f:
        for line in f:
            line= (line).split('\t')
            audio_file_name = os.path.join(audio_path,line[0].split('_')[0],f"{line[0].split('_')[1].split(':')[0]}.wav")
            print(audio_file_name)
            try:
                sr, in_audio = wavfile.read(audio_file_name)
            except:
                break

            fl_name=line[0].split('_')
            if (len(fl_name)==4):
                audioStart = int(float(fl_name[-2])*sr)
                audioEnd = int(float(fl_name[-1])*sr)
            else:
                #print(float(fl_name[-1]))
                str=float(fl_name[-1])-round(1/15,2)
                #print(str)
                if str<0:
                    str=(str*-1)
                audioStart = int(float(str)*sr)
                audioEnd = int(float(fl_name[-1])*sr)
                #print(audioStart)
                #print(audioEnd)
                #break

            out_put_file=os.path.join(out_put_audio,f"{line[0]}.wav")
            audio_data=in_audio[audioStart:audioEnd]
            wavfile.write(out_put_file, sr, audio_data)
            print(f"write audio file {line[0]}.wav")
