#cat dataset_100ms.json | jq .events.PAPI_TOT_INS[] -> output all the different objects with their names, labels and data.
#cat dataset_100ms.json | jq .events.PAPI_TOT_INS.mysqld_php -> output only object name, label and data for mysqld
#cat dataset_100ms.json | jq '.events.PAPI_TOT_INS[] | .label'    -> output just labels values for the different tests.
#cat dataset_100ms.json | jq  '.events.PAPI_TOT_INS| keys[]' --> print all tests labels.
import json
import csv
#from plumbum import local # Run shell shell commands from python , documentations ->https://plumbum.readthedocs.io/en/latest/
# (cat ["dataset_100ms.json"] | jq  ['.events.PAPI_TOT_INS| keys[]'])()
#print(local.cmd.ls())
fields = []
data = []
labels = []
processes =[]
processes_states = []


def read_json():

    # Python program to read
    # json file
    # Opening JSON file
    json_file = './datasets/cpu_processes/dataset_100ms.json'
    f = open(json_file,)
    
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    
    # Iterating through the json
    # list
    for i in data["events"]["PAPI_TOT_INS"]:
        #print(data["events"]["PAPI_TOT_INS"][i]["data"])
        fields.append(i)
        labels.append(data["events"]["PAPI_TOT_INS"][i]["label"])
        #print(labels.append(data["events"]["PAPI_TOT_INS"][i]["label"]))
        processes_states.append(data["events"]["PAPI_TOT_INS"][i]["data"][:555])

    # print(fields)
    # print(labels)
    #print(len(processes_states))
     
    # Closing file
    f.close()
read_json()

for i in range(555):
    processes.append("ps"+str(i))
    #processes_states = processes_states[:555]
for i in range(len(fields)):
    processes_states[i].insert(0,fields[i])
    processes_states[i].insert(1,labels[i])
    #print(labels[i])



processes[0]="test_name"
processes[1] = "class"


#print(processes_states[0][1])
def create_csv():
    with open('datasets/cpu_processes/cpu_states.csv', 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(processes)
        write.writerows(processes_states)
create_csv()
print(processes)
