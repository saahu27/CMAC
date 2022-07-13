import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.model_selection import train_test_split

s = np.linspace(0, 2*np.pi,100)
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
    time_array = list()
    time_gen = list()
    trainingerror_list = list()

    for generalization in range(1,36):
        
        association_to_response_mapping = generate_hashmap(s_input,generalization,association_weight)
        count = 0
        training_error = 0
        start = time.time() 
        err_list = list()
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

                if error > 0 or error < 0 :

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
                        
            e_value = 0

            for j in association_to_response_mapping[map_value].values():

                weight_sum= weight_sum+j
                # training_error = training_error + (s_target[count]- weight_sum)
                e_value += np.abs(np.cos(i) - weight_sum)

            err_list.append(e_value)
            count+=1

        end = time.time()

        x = 0
        for i in err_list:
            x = x + i*i

        total_error = np.sqrt(x/len(s_input))
        time_gen.append(generalization)
        time_array = np.append(time_array, (end - start))		
        print("error delta during traing %f, training error after  %d generalization is %f "%(correction,generalization,total_error))
        trainingerror_list.append(np.abs(total_error))

        if generalization == 3:
            association_response_mapping_gen = association_to_response_mapping
    
        elif generalization == 5:
            association_response_mapping_gen2 = association_to_response_mapping
    
        elif generalization == 9:        
            association_response_mapping_gen3 = association_to_response_mapping
        
        elif generalization == 35:
            association_response_mapping_gen4 = association_to_response_mapping

    return time_gen,time_array,association_response_mapping_gen,association_response_mapping_gen2,association_response_mapping_gen3,association_response_mapping_gen4,trainingerror_list


method = "discrete"
time_gen,time_array,association_response_mapping_gen,association_response_mapping_gen2,association_response_mapping_gen3,association_response_mapping_gen4,trainingerror_list_discrete = train(s_input,s_target,association_weight,method)

method = "continous"
time_gen_continous,time_array_continous,association_response_mapping_gen_continous,association_response_mapping_gen2_continous,association_response_mapping_gen3_continous,association_response_mapping_gen4_continous,trainingerror_list_continous = train(s_input,s_target,association_weight,method)



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


predicted_list3 = test(s_test,association_weight,association_response_mapping_gen,s_target_test,3)
predicted_list5 = test(s_test,association_weight,association_response_mapping_gen2,s_target_test,5)
predicted_list9 = test(s_test,association_weight,association_response_mapping_gen3,s_target_test,9)
predicted_list35 = test(s_test,association_weight,association_response_mapping_gen4,s_target_test,35)

plt.plot(s_test,predicted_list3,color="blue")
plt.plot(s_test,predicted_list5,color="green")
plt.plot(s_test,predicted_list9,color="yellow")
plt.plot(s_test,predicted_list35,color="brown")
plt.plot(s_test,s_target_test,color='black')
plt.title('Original vs Approximated Function using Discrete CMAC')
plt.legend(['generalization_3','generalization_5','generalization_9','generalization_35','Original Function'])
plt.show()

plt.plot(time_gen,time_array)
plt.title('Generelization vs Time taken for convergence-Discrete CMAC')
plt.show()

plt.plot(time_gen,trainingerror_list_discrete)
plt.title('Generelization vs RMS error')
plt.show()

predicted_list3 = test(s_test,association_weight,association_response_mapping_gen_continous,s_target_test,3)
predicted_list5 = test(s_test,association_weight,association_response_mapping_gen2_continous,s_target_test,5)
predicted_list9 = test(s_test,association_weight,association_response_mapping_gen3_continous,s_target_test,9)
predicted_list35 = test(s_test,association_weight,association_response_mapping_gen4_continous,s_target_test,35)

plt.plot(s_test,predicted_list3,color="blue")
plt.plot(s_test,predicted_list5,color="green")
plt.plot(s_test,predicted_list9,color="yellow")
plt.plot(s_test,predicted_list35,color="blue")
plt.plot(s_test,s_target_test,color='black')
plt.title('Original vs Approximated Function using continous CMAC')
# plt.legend(['generalization_3','generalization_5','generalization_9','generalization_35','Original Function'])
plt.legend(['Generalization_3','Original Function'])
plt.show()

plt.plot(time_gen_continous,time_array_continous)
plt.title('Generelization vs Time taken for convergence-continous CMAC')
plt.show()

plt.plot(time_gen,trainingerror_list_continous)
plt.title('Generelization vs RMS error')
plt.show()

