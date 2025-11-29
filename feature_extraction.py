import requests
from pathlib import Path
import pandas as pd

root_path = Path('')
path_to_dataset =  'GTEx Portal.csv' #list of tissue samples and metadata
df = pd.read_csv(path_to_dataset)
# print(df.head())

#filtering out those wiht percentage of steatosis and fibrosis information interms of percentage
df_filtered = df[df['Pathology Notes'].astype(str).str.contains('%', na=False)]
# Reset index to avoid KeyError later
df_filtered = df_filtered.reset_index(drop=True)
print(df_filtered.shape) #checking the number of samples after filtering (114, 8)


#DOWNLOADING IMAGES USING THE SELECTED TISSUE SAMPLE INFORMATION
def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"File downloaded to {destination}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # create a folder to store whole slide images
    path_to_slides = root_path
    if not path_to_slides.is_dir():
        path_to_slides.mkdir()

    # download the slides
    for i in range(df_filtered.shape[0]): #change the first_fifty to second_fifty dataset
        slide_name = df_filtered['Tissue Sample ID'][i]
        file_url = "https://brd.nci.nih.gov/brd/imagedownload/{}".format(slide_name)
        destination_file = path_to_slides / '{}.svs'.format(slide_name)

        if not destination_file.is_file():
            print(f'downloading {file_url}...')
            download_file(file_url, destination_file)
