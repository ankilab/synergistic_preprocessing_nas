import torch
import argparse
from tqdm import tqdm
import os
import json
from torch.utils.data import DataLoader

# Data preparation
from data.prepare_speech_commands import prepare_speech_commands  
from data.prepare_esc_50_dataset import prepare_esc50_dataset
from data.prepare_spoken100_dataset import prepare_spoken100_dataset
from data.prepare_vocal_sound_dataset import prepare_vocal_sound_dataset

# Dataloaders
from dataloader.dl_speech_commands import SubsetSC
from dataloader.dl_esc50 import ESC50DataLoader
from dataloader.dl_spoken100 import Spoken100DataLoader
from dataloader.dl_vocal_sound import VocalSoundDataLoader

# Data Preprocessing
from helper.get_pre_processing_transform import get_pre_processing_transform

# Models
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

torch.multiprocessing.set_sharing_strategy('file_system')

# Constants for audio processing
N_FFT = 25 # in ms
HOP_LENGTH = 10 # in ms
N_MELS = 64 # number of mel bands

def run_baseline(model_name, num_labels, dataset, orig_sr, sample_length, n_fft, hop_length, n_mels, results_path):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        'method': 'mel',
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'use_db': True,
        'stft_power': 2,
        'sample_rate': orig_sr
    }

    # Load model
    if model_name == 'mobilenetv3-small':
        model = mobilenet_v3_small(num_classes=num_labels)
    elif model_name == 'mobilenetv3-large':
        model = mobilenet_v3_large(num_classes=num_labels)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_labels)
    else:
        raise ValueError(f'Model {model_name} not supported')

    model.to(device)

    # Prepare the dataset
    preprocess_transform = get_pre_processing_transform(params, orig_sr=orig_sr, sample_length=sample_length)

    if dataset == 'speech_commands':
        train_dataset = SubsetSC(subset='training', transform=preprocess_transform)
        val_dataset = SubsetSC(subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        N_EPOCHS = 10
    elif dataset == 'esc50':
        train_dataset = ESC50DataLoader("data/ESC-50-master/audio", subset='training', transform=preprocess_transform)
        val_dataset = ESC50DataLoader("data/ESC-50-master/audio", subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        N_EPOCHS = 100
    elif dataset == 'spoken100':
        train_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='training', transform=preprocess_transform)
        val_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        N_EPOCHS = 100
    elif args.dataset == 'vocal_sound':
        train_dataset = VocalSoundDataLoader("data/VocalSound", subset='training', transform=preprocess_transform)
        val_dataset = VocalSoundDataLoader("data/VocalSound", subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        N_EPOCHS = 20
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')

    # Check if class_weights.json exists, if not calculate class weights
    if os.path.exists(f'data/{args.dataset}_class_weights.json'):
        print('Loading class weights from file')
        with open(f'data/{args.dataset}_class_weights.json', 'r') as f:
            class_weights = json.load(f)
        class_weights = torch.FloatTensor(class_weights).to(device)
    else: 
        print('Calculating class weights now')
        class_weights = train_dataset.get_class_weights()
        with open(f'data/{args.dataset}_class_weights.json', 'w') as f:
            json.dump(class_weights, f)
        class_weights = torch.FloatTensor(class_weights).to(device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    val_accs = []
    for epoch in range(N_EPOCHS):
        model.train()
        for input, target in tqdm(train_loader):
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target, weight=class_weights)
            loss.backward()
            optimizer.step()        

        scheduler.step()
    
        model.eval()
        num_correct = 0
        num_total = 0
        best_accuracy = 0
        with torch.no_grad():
            for input, target in val_loader:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                _, predicted = output.max(1)
                num_total += target.size(0)
                num_correct += predicted.eq(target).sum().item()
                
        accuracy = num_correct / num_total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'{results_path}/model.pth')

        print(f'Epoch {epoch+1}/{N_EPOCHS} - Accuracy: {accuracy:.2f}')
        val_accs.append(accuracy)

    with open(f'{results_path}/val_accs.csv', 'w') as f:
        f.write(','.join(map(str, val_accs)))

    # save best accuracy to text file
    with open(f'{results_path}/best_accuracy.txt', 'w') as f:
        f.write(f'{best_accuracy:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="speech_commands", choices=['speech_commands', 'spoken100', 'vocal_sound'])
    parser.add_argument('--model' , type=str, default='mobilenetv3-small', choices=['mobilenetv3-small', 'mobilenetv3-large', 'mobilenetv2']) # only relevant for experiment 2
    args = parser.parse_args()

    results_path = f"results/baselines/{args.dataset}/{args.model}"
    os.makedirs(results_path, exist_ok=True)

    # clean up the results folder
    for file in os.listdir(results_path):
        os.remove(os.path.join(results_path, file))

    if args.dataset == 'speech_commands':
        path_sc_silence = "data/SpeechCommands/speech_commands_v0.02/_silence_"
        if not os.path.exists(path_sc_silence) or len(os.listdir(path_sc_silence)) == 0:
            # Class was not created yet, prepare the dataset here now
            prepare_speech_commands()
        num_labels = 12
        orig_sr = 16000
        sample_length = 1 # seconds
    elif args.dataset == 'esc50':
        if not os.path.exists("data/ESC-50-master/"):
            # Class was not created yet, prepare the dataset here now
            prepare_esc50_dataset()
        num_labels = 50
        orig_sr = 44100
        sample_length = 5
    elif args.dataset == 'spoken100':
        if not os.path.exists("data/SpokeN-100/"):
            prepare_spoken100_dataset()
        num_labels = 100
        orig_sr = 44100
        sample_length = 2
    elif args.dataset == 'vocal_sound':
        if not os.path.exists("data/VocalSound/"):
            prepare_vocal_sound_dataset()
        num_labels = 6
        orig_sr = 16000
        sample_length = 5 # seconds
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')

    # Calculate n_fft, hop_length and n_mels based on the sample rate
    # n_fft = int(orig_sr * N_FFT / 1000)
    # hop_length = int(orig_sr * HOP_LENGTH / 1000)
    # n_mels = N_MELS
    
    n_fft = 512
    hop_length = 32
    n_mels = 128

    print(f'Running with n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}')

    run_baseline(args.model, num_labels, args.dataset, orig_sr, sample_length, n_fft, hop_length, n_mels, results_path)

