import nni
from nni.nas.strategy import RegularizedEvolution
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from nni.nas.experiment import NasExperimentConfig
from nni.nas.evaluator.pytorch.lightning import DataLoader
from nni.nas.experiment.config.engine import ExecutionEngineConfig
from nni.nas.experiment.config.format import ModelFormatConfig

import torch
import argparse
from tqdm import tqdm
import os
import json

from dataloader.dl_speech_commands import SubsetSC
# from dataloader.dl_esc50 import ESC50DataLoader
from dataloader.dl_spoken100 import Spoken100DataLoader
#from dataloader.dl_libri_speech import LibriSpeechDataLoader
from dataloader.dl_vocal_sound import VocalSoundDataLoader

from search_spaces.exp1_search_space import Experiment1SearchSpace
from search_spaces.exp2_search_space import Experiment2SearchSpace
from search_spaces.exp3_search_space import Experiment3SearchSpace

from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

from helper.get_pre_processing_transform import get_pre_processing_transform
from data.prepare_speech_commands import prepare_speech_commands  
# from data.prepare_esc_50_dataset import prepare_esc50_dataset
from data.prepare_spoken100_dataset import prepare_spoken100_dataset
# from data.prepare_libri_speech_dataset import prepare_libri_speech
from data.prepare_vocal_sound_dataset import prepare_vocal_sound_dataset

torch.multiprocessing.set_sharing_strategy('file_system')

global model_idx
model_idx = 0

global class_weights
class_weights = None

def evaluate_model(model):
    params = nni.get_current_parameter().sample

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create folders for saving models and results
    exp_id = exp.id
    exp_dir = f"./results/{exp_id}/"
    global model_idx
    current_model_dir = os.path.join(exp_dir, f'model_{model_idx}')
    os.mkdir(current_model_dir)

    val_accs_dir = os.path.join(exp_dir, 'val_accs.csv')
    if not os.path.exists(val_accs_dir):
        with open(val_accs_dir, 'w') as f:
            f.write('model_idx,accuracy\n')

    # save model params to results dir as json
    with open(os.path.join(current_model_dir, 'params.json'), 'w') as f:
        json.dump(params, f)

    preprocess_transform = get_pre_processing_transform(params, orig_sr=orig_sr, sample_length=sample_length)

    if args.experiment == 2:
        if args.model == 'mobilenetv3-small':
            model = mobilenet_v3_small(num_classes=num_labels)
        elif args.model == 'mobilenetv3-large':
            model = mobilenet_v3_large(num_classes=num_labels)
        elif args.model == 'mobilenetv2':
            model = MobileNetV2(num_classes=num_labels)
        else:
            raise ValueError(f'Model {args.model} not supported')
        
    model.to(device)

    if args.dataset == 'speech_commands':
        train_dataset = SubsetSC(subset='training', transform=preprocess_transform)
        val_dataset = SubsetSC(subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        N_EPOCHS = 10
    # elif args.dataset == 'esc50':
    #     train_dataset = ESC50DataLoader("data/ESC-50-master/audio", subset='training', transform=preprocess_transform)
    #     val_dataset = ESC50DataLoader("data/ESC-50-master/audio", subset='validation', transform=preprocess_transform)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    #     N_EPOCHS = 100
    elif args.dataset == 'spoken100':
        train_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='training', transform=preprocess_transform)
        val_dataset = Spoken100DataLoader("data/SpokeN-100/", subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        N_EPOCHS = 100
    # elif args.dataset == 'libri_speech':
    #     train_dataset = LibriSpeechDataLoader("data/LibriSpeech/data_windowed_2s_1s/", subset='training', transform=preprocess_transform)
    #     val_dataset = LibriSpeechDataLoader("data/LibriSpeech/data_windowed_2s_1s/", subset='validation', transform=preprocess_transform)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #     N_EPOCHS = 30
    elif args.dataset == 'vocal_sound':
        train_dataset = VocalSoundDataLoader("data/VocalSound", subset='training', transform=preprocess_transform)
        val_dataset = VocalSoundDataLoader("data/VocalSound", subset='validation', transform=preprocess_transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        N_EPOCHS = 20
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')

    global class_weights
    if class_weights is None:
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

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    print("Loading data")
    if args.dataloader == 'complete':
    # I load all the data into GPU memory to speed up training, as otherwise the transform of the dataloader is the bottleneck
        X_train, y_train = [], []
        for i, (input, target) in tqdm(enumerate(train_loader)):
            X_train.append(input)
            y_train.append(target)
        X_train = torch.cat(X_train)
        y_train = torch.cat(y_train)

        X_val, y_val = [], []
        for i, (input, target) in tqdm(enumerate(val_loader)):
            X_val.append(input)
            y_val.append(target)
        X_val = torch.cat(X_val)
        y_val = torch.cat(y_val)

        for epoch in range(N_EPOCHS):
            model.train()
            for i in tqdm(range(0, len(X_train), 16)):
                input = X_train[i:i+16].to(device)
                target = y_train[i:i+16].to(device)
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
                for i in range(0, len(X_val), 16):
                    input = X_val[i:i+16].to(device)
                    target = y_val[i:i+16].to(device)
                    output = model(input)
                    _, predicted = output.max(1)
                    num_total += target.size(0)
                    num_correct += predicted.eq(target).sum().item()
                    
            accuracy = num_correct / num_total

            # check if accuracy is better than previous best, then save model and update best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(current_model_dir, 'best_model.pth'))

            print(f'Epoch {epoch}, accuracy {accuracy}')
    elif args.dataloader == 'batch':
        model.train()
        for epoch in range(N_EPOCHS):
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

            # check if accuracy is better than previous best, then save model and update best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(current_model_dir, 'best_model.pth'))

            print(f'Epoch {epoch}, accuracy {accuracy}')
    else:
        raise ValueError(f'Dataloader {args.dataloader} not supported')

    # save final result
    with open(val_accs_dir, 'a') as f:
        f.write(f'{model_idx},{best_accuracy}\n')
    
    model_idx += 1

    nni.report_final_result(best_accuracy)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="speech_commands", choices=['speech_commands', 'spoken100', 'vocal_sound'])
    parser.add_argument('--dataloader', type=str, default="complete", choices=['complete', 'batch'])
    parser.add_argument('--model' , type=str, default='mobilenetv3-small', choices=['mobilenetv3-small', 'mobilenetv3-large', 'mobilenetv2']) # only relevant for experiment 2
    parser.add_argument('--max_trial_number', type=int, default=1000)
    parser.add_argument('--population_size', type=int, default=200)
    parser.add_argument('--sample_size', type=int, default=50)
    args = parser.parse_args()

    if args.dataset == 'speech_commands':
        path_sc_silence = "data/SpeechCommands/speech_commands_v0.02/_silence_"
        if not os.path.exists(path_sc_silence) or len(os.listdir(path_sc_silence)) == 0:
            # Class was not created yet, prepare the dataset here now
            prepare_speech_commands()
        num_labels = 12
        orig_sr = 16000
        sample_length = 1 # seconds
    # elif args.dataset == 'esc50':
    #     if not os.path.exists("data/ESC-50-master/"):
    #         # Class was not created yet, prepare the dataset here now
    #         prepare_esc50_dataset()
        num_labels = 50
        orig_sr = 44100
        sample_length = 5 # seconds
    elif args.dataset == 'spoken100':
        if not os.path.exists("data/SpokeN-100/"):
            prepare_spoken100_dataset()
        num_labels = 100
        orig_sr = 44100
        sample_length = 2 # seconds
    # elif args.dataset == 'libri_speech':
    #     if not os.path.exists("data/LibriSpeech/"):
    #         prepare_libri_speech()
    #     num_labels = 28
    #     orig_sr = 16000 
    #     sample_length = 2 # seconds
    elif args.dataset == 'vocal_sound':
        if not os.path.exists("data/VocalSound/"):
            prepare_vocal_sound_dataset()
        num_labels = 6
        orig_sr = 16000
        sample_length = 5 # seconds
    else:
        raise ValueError(f'Dataset {args.dataset} not supported')

    if args.experiment == 1:
        search_space = Experiment1SearchSpace(num_labels=num_labels, orig_sr=orig_sr)
    elif args.experiment == 2:
        search_space = Experiment2SearchSpace(orig_sr=orig_sr)
    elif args.experiment == 3:
        search_space = Experiment3SearchSpace(num_labels=num_labels, orig_sr=orig_sr)
    else:
        raise ValueError(f'Experiment {args.experiment} not supported')

    strategy = RegularizedEvolution(population_size=args.population_size, sample_size=args.sample_size, crossover=True)
    evaluator = FunctionalEvaluator(evaluate_model)

    exp_config = NasExperimentConfig.default(search_space, evaluator, strategy)
    exp_config.execution_engine = ExecutionEngineConfig('sequential') # needed to enable usage of GPU
    exp_config.model_format = ModelFormatConfig('simplified') # needed to enable usage of GPU

    exp_config.experiment_name = f'{args.dataset}_exp{args.experiment}'
    exp_config.max_trial_number = args.max_trial_number
    exp_config.max_experiment_duration  = '24h'
    exp_config.trial_concurrency = 1
    exp_config.experiment_working_directory = './results'
    if not os.path.exists(exp_config.experiment_working_directory):
        os.mkdir(exp_config.experiment_working_directory)
    exp_config.debug = True

    exp = NasExperiment(search_space, evaluator, strategy, exp_config)
    exp.run(debug=True)
