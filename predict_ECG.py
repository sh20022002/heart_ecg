import pandas as pd

def loud_data():
    path = r'C:\Users\shmue\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Visual Studio Code\python\open_pojects\test_3.10\heart_ECG'
    return 

def main():
    data = loud_data()
    for line in data:
        print(line)
    
    # create_image_ecg(data, path_data)
if __name__ == '__main__':
    main()