import os
from tqdm import tqdm

def filtering(directory):
    for filename in tqdm(os.listdir(directory)):
        name_without_extension = os.path.splitext(filename)[0]

        if name_without_extension not in filtered_samples:
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

if __name__ == "__main__":
    # Get all samples that are room interiors
    room_count = 0
    total = 0
    filtered_samples = []

    with open('data/ADEChallengeData2016/sceneCategories.txt', 'r') as file:
        for line in file:
            total += 1

            if 'room' in line.split()[1]:
                filtered_samples.append(line.split()[0])
                room_count += 1

    print(f"Total number of samples: {total} -> Samples that are rooms: {room_count}")
    
    # Filtering - sceneCategories.txt
    with open('data/ADEChallengeData2016/sceneCategories.txt', 'r') as file, open('filteredSceneCategories.txt', 'w') as outfile:
        for line in file:
            if line.split()[0] in filtered_samples:
                outfile.write(line)

    # Filtering - annotations/ and images/
    directory = [
        'data/ADEChallengeData2016/annotations/training',
        'data/ADEChallengeData2016/annotations/validation',
        'data/ADEChallengeData2016/images/training',
        'data/ADEChallengeData2016/images/validation',
    ]
    
    for dir in directory:
        filtering(dir)

    