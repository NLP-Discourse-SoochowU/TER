"""
Author: Longyin
Date: 2023-2024
Email: zhangly@i2r.a-star.edu.sg
"""
import os
import shutil


def show_models(path_):
    name_list = []
    for file_name in os.listdir(path_):
        if file_name.startswith("model_") and file_name != "model_last.pth":
            name_list.append(int(file_name[6:-4]))
    name_list = sorted(name_list)
    print_list = [str(item) for item in name_list]
    print(" ".join(print_list))

def select_model(path_):
    """ Because we are finding for a model focused on retrieving """
    print("====== leaves_ac ======")
    name2scores = dict()
    for file_name in os.listdir(path_):
        if file_name.endswith(".json") and not file_name.endswith("details.json"):
            file_ = os.path.join(path_, file_name)
            # print(file_name)
            with open(file_, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if "leaves_ac" in line:
                        line = float(line[:-1].split()[1])
                        name2scores[file_name] = line
                        break
    # sorted
    name2scores = sorted(name2scores.items(), key=lambda x: x[1])
    print(name2scores[-1])


def select_one(file_path):
    select_model(file_path + "/select_on_dev")


def select_one_llm(file_path):
    select_model(file_path + "/select_on_dev_llm")


def select_all(file_path):
    for round in range(1, 6):
        tmp_round = file_path + "/round_" + str(round)
        print(f"========== Round_{round} ==========")
        select_model(tmp_round + "/select_on_dev")


def select_all_llm(file_path):
    for round in range(1, 6):
        tmp_round = file_path + "/round_" + str(round)
        print(f"========== Round_{round} ==========")
        select_model(tmp_round + "/select_on_dev_llm")

def copy_files(file_path, fine_p):
    for round in range(1, 6):
        tmp_round = file_path + "/round_" + str(round) + fine_p
        print(f"========== Round_{round} ==========")
        for file_name in os.listdir(tmp_round):
            source_path = os.path.join(tmp_round, file_name)
            target_path = os.path.join(tmp_round, "predict.tsv")
            if file_name.endswith(".csv") and not file_name.endswith("processed.csv"):
                try:
                   shutil.copy(source_path, target_path)
                except IOError as e:
                   print("Unable to copy file. %s" % e)
                except:
                   print("Unexpected error:", sys.exc_info())


if __name__ == "__main__":
    file_path = "exp/Controller_task2/retrieve_learning_v6"
    # select_one(file_path)
    # select_one_llm(file_path)
    # select_all(file_path)
    select_all_llm(file_path)
    # copy_files(file_path, "/reproduce_task2_llm")
    # show_models(file_path)
    
