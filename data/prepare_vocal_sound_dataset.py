import os
import zipfile
import wget
import argparse
import json

def prepare_vocal_sound_dataset(path="data/"):
    # Download the dataset
    dataset_url = "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1"
    dataset_path = os.path.join(path, "vs_release_16k.zip")
    wget.download(dataset_url, dataset_path)

    # Unzip the dataset into "VocalSound" directory
    extract_path = os.path.join(path, "VocalSound")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Remove the zip file after extraction
    os.remove(dataset_path)

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

def change_path(json_file_path, target_path):
    with open(json_file_path, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['data']

    # change the path in the json file
    for i in range(len(data)):
        ori_path = data[i]["wav"]
        new_path = target_path + '/audio_16k/' + ori_path.split('/')[-1]
        data[i]["wav"] = new_path

    with open(json_file_path, 'w') as f:
        json.dump({'data': data}, f, indent=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=None, help="the dataset path, please use the absolute path")
    args = parser.parse_args()

    prepare_vocal_sound_dataset()
    
    # if no path is provided, use the default path /data/VocalSound
    if args.data_dir is None:
        data_dir = 'data/VocalSound'
    else:
        data_dir = args.data_dir

    # for train, validation, test
    json_files = get_immediate_files(data_dir + '/datafiles/')
    for json_f in json_files:
        if json_f.endswith('.json'):
            print('now processing ' + data_dir + '/datafiles/' + json_f)
            change_path(data_dir + '/datafiles/' + json_f, data_dir)

    # for subtest sets
    json_files = get_immediate_files(data_dir + '/datafiles/subtest/')
    for json_f in json_files:
        if json_f.endswith('.json'):
            print('now processing ' + data_dir + '/datafiles/subtest/' + json_f)
            change_path(data_dir + '/datafiles/subtest/' + json_f, data_dir)
