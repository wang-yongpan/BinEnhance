# coding:utf-8
import subprocess
import os
import time
from multiprocessing import Process
import tqdm

from settings import IDA_PATH
import networkx as nx


def is_exist(inputName, savePath):
    fname = inputName.split('/')[-1].split(".elf")[0]
    fileName = savePath + fname + '.pkl'
    if os.path.exists(fileName):
        return True
    else:
        return False

def getAllFiles(filePath, savePath, process_num):
    fileList = []
    tmp_extention = ['nam', 'til', 'id0', 'id1', 'id2', 'id3', 'id4', 'i64', 'idb']
    files = os.walk(filePath)
    archs = ["arm", "mips", "x86"]
    coms = ["gcc-7.3.0"]
    remove_other_files(filePath, tmp_extention)
    for path, dir_path, file_name in files:
        for file in file_name:
            if file.split(".")[-1] != "elf":
                continue
            opv = file.split("_")[1]
            ar = file.split("_")[2]
            li = file.split("_")[3]
            if opv in coms and ar in archs and li == "32":
                if file.split(".")[-1] not in tmp_extention and "Os" not in file:
                    fP = str(os.path.join(path, file))
                    if not is_exist(fP, savePath):
                        fileList.append(fP)
    p_list = []
    for i in range(process_num):
        files = fileList[int((i)/process_num*len(fileList)): int((i+1)/process_num*len(fileList))]
        p = Process(target=extract, args=(files, savePath))
        p_list.append(p)
    time_start = time.time()
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    time_cost = time.time() - time_start
    print("time_cost:" + str(time_cost))
    print("bin_nums:" + str(len(fileList)))
    remove_other_files(filePath, tmp_extention)
    pass

def remove_other_files(filePath, tmp_extention):
    files = os.walk(filePath)
    for path, dir_path, file_name in files:
        for file in file_name:
            if file.split(".")[-1] in tmp_extention:
                os.remove(os.path.join(path, file))

def extract(filePaths, savePath):
    currentPath = os.getcwd()
    script_path = os.path.join(currentPath, "eesg_extract_script.py")
    tf = tqdm.tqdm(filePaths)
    for filePath in tf:
        ida_cmd = 'TVHEADLESS=1 ' + IDA_PATH + ' -Llog.txt -c -A -B -S\'' + script_path + '\' ' + filePath
        print(ida_cmd)
        s, o = subprocess.getstatusoutput(ida_cmd)
        if s != 0:
            with open('error.txt', 'a') as file:
                file.write(filePath + '\n')
            print("error: " + filePath)
        else:
            tf.set_description("[" + filePath.split("/")[-1] + "] Extract Success")
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", "-o", type=str, required=True, help="the directory of output files")
    parser.add_argument("--process-num", "-p", type=int, default=1, help="the number of process")
    args = parser.parse_args()
    from settings import DATA_BASE
    filePath = DATA_BASE
    savePath = DATA_BASE
    process_num = args.process_num
    getAllFiles(filePath, savePath, process_num)
    print("extract completed！")
    pass

