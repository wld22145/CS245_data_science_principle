import pickle

strorage_path = 'ss_tmp/'
reg_result = []
for i in range(50):
    print(i)
    dir_reg_result = pickle.load( open(strorage_path + "reg_result"+str(i)+".pkl","rb"))
    reg_result.extend(dir_reg_result)

pickle.dump(reg_result,open("reg_result.pkl","wb"))
print(len(reg_result))
print(reg_result[0])