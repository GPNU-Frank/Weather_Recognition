import requests
import sys

def download(url):
    size = 0
    chunk_size = 1024
    response = requests.get(url, stream=True, verify=False)
    with open('None.flv', 'wb') as file:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            size += len(data)
            file.flush()
            sys.stdout.write('  [下载进度]:%.2fMB' % float(size /1024/1024 ) + '\r')

if __name__ == '__main__':
    url = 'https://runpull.runoneapp.com/runone/hd-EC00A7602C45E384943E089169F3FAC3.flv?t=5fe93fb1&k=4a1031f793648b123052fda8c1146e55'
    download(url)