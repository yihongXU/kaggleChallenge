import os

dir_list = next(os.walk('/home/yxu/Downloads/Images/'))[1]
for dir_name in dir_list:
    print(dir_name)
    print(dir_name.lower())
    os.rename('/home/yxu/Downloads/Images/'+dir_name, '/home/yxu/Downloads/Images/'+dir_name.lower())
