import json
import os


def main():
    ## Load the name file.
    with open("llm_papers.txt") as f:
        names = f.readlines()
    with open("arxiv_code_map.json") as f:
        title_dict = json.load(f)
        code_dict = {v: k for k, v in title_dict.items()}

    names = [name.strip() for name in names]
    staging_dict = {f"paper__{str(k+1).zfill(4)}.png": v for k, v in enumerate(names)}
    img_dir = 'img/'

    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            if filename in staging_dict:
                new_name = code_dict[staging_dict[filename]] + '.png'
                old_filepath = os.path.join(img_dir, filename)
                new_filepath = os.path.join(img_dir, new_name)

                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_name}.")

if __name__ == "__main__":
    main()
