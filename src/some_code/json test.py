import  json

# cv2list = {"cv2list":["adb","thy","car"],"dicom":["dicom"]}
# # #
# # #
# with open("jsontest.json","a") as f:
#      json.dump(cv2list,f)
#


datasetw = "adbtte"
f =  open("jsontest.json","r")
dict = json.load(f)
dict["cv2list"].append(datasetw)
f.close()


f =  open("jsontest.json","w",encoding='utf-8')
json.dump(dict,f,ensure_ascii = False)

# print(dict)
# print(dict["cv2list"])
# if dataset in dict["cv2list"]:
#     print("is")
#     f.close()