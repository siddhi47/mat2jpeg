"""
    author: Siddhi
    date: 2023-02-09
    description: Convert the signlas from .mat to images
"""

import os
import glob
import scipy
import librosa
import itertools
import numpy as np
import librosa.display
import concurrent.futures
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def generate_mel_spectrogram(signal):
    """
        Generate mel spectrogram from signal

        params:
            signal: signal to generate mel spectrogram from
        return:
            mel_spectrogram: numpy array of mel spectrogram
    """
    # Generate mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=50, n_fft=2048, hop_length=5, n_mels=128)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Convert to float32
    mel_spectrogram = mel_spectrogram.astype(np.float32)
    return mel_spectrogram


def get_file_list(directory:str):
    """
    get all the file with .mat extension
    
    params:
        directory: Directory to check.

    returns:
        generator for all the files with .mat extension
    """

    glob_pattern = os.path.join(directory, '**/*.mat')
    return glob.iglob(glob_pattern, recursive=True)


def get_mel_spectogram_from_file(file_path:str)->np.ndarray:
    """
    get mel spectrogram from file

    params:
        file_path: path to the file

    returns:
        mel_spectrogram: mel spectrogram of the file
    """
    # Read the file
    signal = scipy.io.loadmat(file_path)['val'][0]
    signal = signal.astype(np.float32)
    mel_spectrogram = generate_mel_spectrogram(signal)
    return mel_spectrogram

def save_spectogram(src_file:str, dest_folder:str)->str:
    """
    Saves a jpeg file for a  given .mat file

    params:
        src_file: path to the .mat file
        dest_folder: path to the destination folder
    returns:
        path to the saved file
    """
    print(f'Processing {src_file}')
    mel_spectrogram = get_mel_spectogram_from_file(src_file)

    #mel_spectrogram = mel_spectrogram.astype(np.uint8)
    # Get the file name
    file_name = os.path.basename(src_file)
    file_name = os.path.splitext(file_name)[0]
    file_name = file_name + '.jpeg'

    # Save the file
    dest_file = os.path.join(dest_folder, file_name)
    librosa.display.specshow(mel_spectrogram)
    plt.savefig(dest_file, bbox_inches='tight', pad_inches=-.05, dpi=100)
    return dest_file

    
def main():
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to the source folder')
    parser.add_argument('--dest', type=str, required=True, help='Path to the destination folder')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    args = parser.parse_args()

    # Get the list of files
    file_list = get_file_list(args.src)

    # Save the files
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as executor:
        executor.map(save_spectogram, file_list, itertools.repeat(args.dest))

if __name__ == '__main__':
    main()

