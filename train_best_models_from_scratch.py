import torch
import json
import numpy as np
import os
import shutil
import copy
import argparse
from tqdm import tqdm
from pathlib import Path
from nni.nas.evaluator.pytorch.lightning import DataLoader

from nni.nas.hub.pytorch import MobileNetV3Space
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

from helper.get_pre_processing_transform import get_pre_processing_transform
from dataloader.dl_speech_commands import SubsetSC
from dataloader.dl_spoken100 import Spoken100DataLoader
from dataloader.dl_vocal_sound import VocalSoundDataLoader

# determine 5 random seeds
SEEDS = [1, 2, 3, 4, 5]
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_and_evaluate_model(random_seed, model, optimizer, scheduler, save_dir, dataset_name, train_dataset, 
                             val_dataset, test_dataset, n_epochs):
    save_dir = Path(save_dir) / f'seed_{random_seed}'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)
    
    with open(f'data/{dataset_name}_class_weights.json', 'r') as f:
        class_weights = json.load(f)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    model.train()
    for epoch in range(n_epochs):
        for input, target in tqdm(train_loader):
            input = input.to(DEVICE)
            target = target.to(DEVICE)
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
                input = input.to(DEVICE)
                target = target.to(DEVICE)
                output = model(input)
                _, predicted = output.max(1)
                num_total += target.size(0)
                num_correct += predicted.eq(target).sum().item()
                
        accuracy = num_correct / num_total
        print(f'Epoch {epoch}: Validation accuracy: {accuracy}')

        # check if accuracy is better than previous best, then save model and update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_dir / 'model.pth')
            
    # load best model and evaluate on test set
    model.load_state_dict(torch.load(save_dir / 'model.pth'))
    
    X_test, y_test = [], []
    for i, (input, target) in tqdm(enumerate(test_loader)):
        X_test.append(input)
        y_test.append(target)
    X_test = torch.cat(X_test)
    y_test = torch.cat(y_test)
    
    model.eval()
    num_correct = 0
    num_total = 0
    test_accuracy = 0
    
    with torch.no_grad():
        for i in range(0, len(X_test), 16):
            input = X_test[i:i+16].to(DEVICE)
            target = y_test[i:i+16].to(DEVICE)
            output = model(input)
            _, predicted = output.max(1)
            num_total += target.size(0)
            num_correct += predicted.eq(target).sum().item()
            
    test_accuracy = num_correct / num_total
    
    return test_accuracy

def train_experiment(model_path, params_path, save_path, exp_path, dataset_name, num_labels, orig_sr, sample_length, initial_lr, n_epochs):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    with open(params_path) as json_file:
        params = json.load(json_file)

    # Remove parameters that are not needed for model architecture
    arch = copy.deepcopy(params)
    keys_to_remove = ['method', 'n_fft', 'hop_length', 'n_mels', 'use_db', 'sample_rate', 'stft_power', 'n_mfcc', 'wavelet_scaling', 'wavelet_resize']

    for key in keys_to_remove:
        if key in arch:
            del arch[key]

    all_test_accuracies = []
    for seed in SEEDS:
        # Set seed
        torch.manual_seed(seed)
        
        # Load model
        if "Experiment_2" in str(model_path):
            if "MNv2" in str(model_path):
                model = MobileNetV2(num_classes=num_labels)
            elif "MNv3small" in str(model_path):
                model = mobilenet_v3_small(num_classes=num_labels)
            elif "MNv3large" in str(model_path):
                model = mobilenet_v3_large(num_classes=num_labels)
        else:
            model = MobileNetV3Space(num_labels=num_labels).load_custom_model(arch, num_labels)
            
        # model_state_dict = torch.load(model_path)
        # model.load_state_dict(model_state_dict)
        model.to(DEVICE)

        # Load preprocessing transform
        preprocess_transform = get_pre_processing_transform(params, orig_sr, sample_length)

        # Load datasets
        if dataset_name == "vocal_sound":
            train_dataset = VocalSoundDataLoader("data/VocalSound", subset='training', transform=preprocess_transform)
            val_dataset = VocalSoundDataLoader("data/VocalSound", subset='validation', transform=preprocess_transform)
            test_dataset = VocalSoundDataLoader("data/VocalSound", subset='test', transform=preprocess_transform)
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        elif dataset_name == "spoken100":
            train_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='training', transform=preprocess_transform)
            val_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='validation', transform=preprocess_transform)
            test_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='testing', transform=preprocess_transform)
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        elif dataset_name == "speech_commands":
            train_dataset = SubsetSC("data/speech_commands", subset='training', transform=preprocess_transform)
            val_dataset = SubsetSC("data/speech_commands", subset='validation', transform=preprocess_transform)
            test_dataset = SubsetSC("data/speech_commands", subset='testing', transform=preprocess_transform)
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


        test_accuracy = train_and_evaluate_model(seed, model, optimizer, scheduler, save_path, dataset_name, train_dataset, val_dataset, test_dataset, n_epochs)
        print(f"Seed {seed}: Test accuracy: {test_accuracy}")
        
        all_test_accuracies.append(test_accuracy)
            
        with open(save_path / "test_accuracies.json", "w") as f:
            json.dump(all_test_accuracies, f)
        
        # clear gpu
        torch.cuda.empty_cache()
        
        # del model and optimizer
        del model
        del optimizer
          
    # print mean and std of test accuracies
    print(f"Mean test accuracy: {np.mean(all_test_accuracies)}")
    print(f"Std test accuracy: {np.std(all_test_accuracies)}")

if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()

    if args.dataset == "vocal_sound":
        INITIAL_LR = 0.0005
        N_EPOCHS = 20
        NUM_LABELS = 6
        ORIG_SR = 16000
        SAMPLE_LENGTH = 5 # seconds
        BASE_PATH = Path("best_models/VocalSound/")
    elif args.dataset == "spoken100":
        INITIAL_LR = 0.005
        N_EPOCHS = 100
        NUM_LABELS = 100
        ORIG_SR = 44100
        SAMPLE_LENGTH = 2 # seconds
        BASE_PATH = Path("best_models/SpokeN/")
    elif args.dataset == "speech_commands":
        INITIAL_LR = 1e-3
        N_EPOCHS = 10
        NUM_LABELS = 12
        ORIG_SR = 16000
        SAMPLE_LENGTH = 1 # seconds
        BASE_PATH = Path("best_models/SpeechCommands/")
        
    ############################################## 
    # Experiment 1
    ##############################################
    exp_path = BASE_PATH / "Experiment_1"
    model_path = exp_path / "best_model.pth"
    params_path = exp_path / "params.json"
    save_path = exp_path / "final_training/"

    # train_experiment(model_path, params_path, save_path, exp_path, args.dataset, NUM_LABELS, ORIG_SR, SAMPLE_LENGTH, INITIAL_LR, N_EPOCHS)

    ############################################## 
    # Experiment 2
    ##############################################
    exp_path = BASE_PATH / "Experiment_2"
    # model_names = ["MNv2", "MNv3small", "MNv3large"]
    model_names = ["MNv3small", "MNv3large"]

    for model_name in model_names:
        model_path = exp_path / model_name / "best_model.pth"
        params_path = exp_path / model_name / "params.json"
        save_path = exp_path / model_name / "final_training/"
        
        train_experiment(model_path, params_path, save_path, exp_path, args.dataset, NUM_LABELS, ORIG_SR, SAMPLE_LENGTH, INITIAL_LR, N_EPOCHS)

    ############################################## 
    # Experiment 3
    ##############################################
    exp_path = BASE_PATH / "Experiment_3"
    model_path = exp_path / "best_model.pth"
    params_path = exp_path / "params.json"
    save_path = exp_path / "final_training/"

    train_experiment(model_path, params_path, save_path, exp_path, args.dataset, NUM_LABELS, ORIG_SR, SAMPLE_LENGTH, INITIAL_LR, N_EPOCHS)