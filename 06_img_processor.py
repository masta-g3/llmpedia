import json
import os


def main():
    ## Load the mapping dictionary.
    with open("arxiv_code_map.json", "r") as f:
        staging_dict = json.load(f)
    reverse_mapping = {v: k for k, v in staging_dict.items()}
    img_dir = 'img/'

    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            name_without_ext = filename[:-4]
            if name_without_ext in reverse_mapping:
                new_name = reverse_mapping[name_without_ext] + '.png'
                old_filepath = os.path.join(img_dir, filename)
                new_filepath = os.path.join(img_dir, new_name)

                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_name}.")

if __name__ == "__main__":
    main()
