# Virtual Analog Modelling of Parameterized Audio Systems Using Bufferized LSTM's

Fast and real-time adopted bufferized LSTM-based networks for Virtual Analog Compression Modelling trained and evaluated on [SignalTrain LA2A Dataset](https://zenodo.org/record/3824876). From 10x up to 150x faster models than current state-of-the-art (2022) are avaliable, having up to 20% better performance on STFT metric (LSTM-256-ID).

## Installation

Use the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Avaliable Models
The following models are avaliable from `/assets` folder:
- LSTM
  - LSTM-80
  - LSTM-256
    ![Alt text](img/LSTM-256.png?raw=true "Squeeze-LSTM Architecture")

- LSTM-ID (Input Dependent) 
  - LSTM-80-ID
  - LSTM-256-ID
  ![Alt text](img/LSTM-256-ID2.png?raw=true "Squeeze-LSTM Architecture")

- Squeeze-LSTM
    ![Alt text](img/Squeeze-LSTM.png?raw=true "Squeeze-LSTM Architecture")


## Inference
To use `infer.py` script simply change the following lines:

```bash
# path to folder with input audios
path_to_data = Path('path_to_audio')
# path to output folder
out_path = Path('output_path')
# 1-st param is comp/lim, 2-nd param is gain reduction
params = [0.0, 50.0]
```
If output files format is not supported inside your DAW you can use following commands to reconvert it. 
```bash
y, sr = soundfile.read('wav_path')
soundfile.write('wav_path', y, sr)
```

## Training

To properly use provided train scripts, e.g `train_256_id.py`, insert right dataset part in the config:

```bash
config = {
    # path to dataset 
    'dataset_dir': "path_to_dataset/SignalTrain_LA2A_Dataset_1.1",
    ...
}
```

Link to the datasets: [SignalTrain LA2A Dataset](https://zenodo.org/record/3824876)

## Test

Use `test.ipynb` jupyter notebook to evaluate the models. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)