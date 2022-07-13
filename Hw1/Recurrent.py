import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.model_selection import train_test_split

s = np.linspace(0, 0.5*np.pi,100)
s_input = []
s_test= []
for i in range(0,len(s)):
    if i%3==0 :

        s_test.append(s[i])

    else:

        s_input.append(s[i])

s_target = np.cos(s_input)
s_target_test=np.cos(s_test)
association_weight=35
input = set(s_input)

def generate_hashmap(s_input,generalization,association_weight):

    association_to_response_mapping=defaultdict(dict)
    input=set(s_input)

    for i in input:

        i=round(float(i*association_weight),4)

        for j in range(int(i),int(i)+generalization+1):
            # m* -> a* -> contents
            association_to_response_mapping[i][j]=0

    return association_to_response_mapping

def train(s_input,s_target,association_weight,method):
    epoch=1
    error_list=list()
    generalization = 3
    num=0
    learning_rate = 0.8
    previous_error = 0
    association_to_response_mapping = generate_hashmap(s_input,generalization,association_weight)

    while epoch <= 100:
        error_sum = 0
        count = 0

        for i in s_input:

            map_value=round(float(i*association_weight),4)

            if map_value not in association_to_response_mapping.keys():

                print("input is not already a key in hashmap")

            if map_value in association_to_response_mapping.keys():

                weight_sum = 0
                for v in association_to_response_mapping[map_value].values():
                    weight_sum = weight_sum + v
                
                error = np.cos(i) - weight_sum
                # error=s_target[count] - weight_sum
                correction=error/generalization
                error_sum += error

                if error > 0 or error < 0:

                    for index,value in enumerate(association_to_response_mapping[map_value].keys()) :

                        if method == "discrete":
                            association_to_response_mapping[map_value][value]=correction

                        elif method == "continous":
                            
                            if index==0:  

                                association_to_response_mapping[map_value][value]+=0.5*correction

                            elif index==generalization:

                                association_to_response_mapping[map_value][value]+=0.5*correction

                            else:

                                association_to_response_mapping[map_value][value]+=correction
            count+=1
        
        error_sum = error_sum/len(s_input) + learning_rate * previous_error
        error_list.append(error_sum)
        previous_error = error_sum
        epoch +=1             
        print("correction during traing %f, training error after  %d generalization is %f "%(correction,generalization,error_sum))

    return association_to_response_mapping


method = "continous"
association_response_mapping = train(s_input,s_target,association_weight,method)



def find_closest_key(map_value,association_to_response_mapping):
    a=[abs(map_value-x) for x in association_to_response_mapping.keys()]
    return a.index(min(a))

def test(s_test,association_weight,association_to_response_mapping,target_test,generalization):
    test_error=0
    predicted_list=list()

    for index,i in enumerate(s_test):
        map_value=round(float(i*association_weight),4)

        if map_value not in association_to_response_mapping.keys():
            k=find_closest_key(map_value,association_to_response_mapping)
            new_map_value=list(association_to_response_mapping.keys())[k]
            map_value=new_map_value
        
        weight_sum=0
        for j in association_to_response_mapping[map_value].values():
            weight_sum=weight_sum+j
        test_error+= np.abs(target_test[index]-weight_sum)
        predicted_list.append(weight_sum)
    
    print("Average Test error percentage for generalization value %d is---> %f"%(generalization,100*(test_error/len(s_test))))
    print('Average Test Accuracy percentage %f'%(100 - np.abs(100*(test_error/len(s_test)))))
    count = 0
    for i in range(0,len(predicted_list)):
        if(predicted_list[i] == np.cos(s_test[i])):
            count+=1
    print("accuracy",count/len(s_test))

    
    return predicted_list       


predicted_list = test(s_test,association_weight,association_response_mapping,s_target_test,3)


plt.plot(s_test,predicted_list,color="blue")
plt.plot(s_test,s_target_test,color='black')
plt.title('Original vs Approximated Function using continous CMAC')
plt.legend(['generalization_3','Original Function'])
plt.show()

