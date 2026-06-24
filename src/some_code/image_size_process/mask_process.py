import  cv2
import  os


def mask_process(folder,save_path):
    mask1 = cv2.imread(r"/home/us-ubuntu/us/Liangshubo/Breast_plus/dataset/rawdata/breast_plus_v2_train/BRETRA_breast_mask.png",0)
    mask2 = cv2.imread(r"/home/us-ubuntu/us/Liangshubo/Breast_plus/dataset/rawdata/breast_plus_v2_train/SAMSUNG_mask.png",0)

    name_ext_list = os.listdir(folder)
    for i in range(len(name_ext_list)):
        nameext = name_ext_list[i]
        image_path = os.path.join(folder,nameext)
        image_array = cv2.imread(image_path,0)

        if nameext.find("I0000")>=0:
            image_array_mask = image_array*mask2
        else:
            image_array_mask = image_array*mask1

        cv2.imwrite(os.path.join(save_path,nameext),image_array_mask,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {i}/{len(name_ext_list)}")


def mask_process2(folder,save_path):
    mask1 = cv2.imread(r"/media/ubuntu/SSD_3/Liangshubo/Thyroid_plus/dataset/rawdata/n20_thyroid_v2_pretrain/mask.png",0)

    name_ext_list = os.listdir(folder)
    for i in range(len(name_ext_list)):
        nameext = name_ext_list[i]
        image_path = os.path.join(folder,nameext)
        image_array = cv2.imread(image_path,0)


        image_array_mask = image_array*mask1

        cv2.imwrite(os.path.join(save_path,nameext),image_array_mask,[cv2.IMWRITE_PNG_COMPRESSION,0])
        print(f"finish {i}/{len(name_ext_list)}")







if __name__ == '__main__':
    name_list = ["cv0_dmapdown131_thyroid_sam4"]
    for i in range(len(name_list)):

        foler =r"/media/ubuntu/SSD_3/Liangshubo/Thyroid_plus/dataset/rawdata/n20_thyroid_v2_pretrain/process_1/"+name_list[i]
        save_path= foler+"_maskcrop"
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        mask_process2(foler,save_path)







    # name_list = ["cv0_dmapdown131_thyroid_sam2_cv1_srblur","cv0_dmapdown131_thyroid_sam2_cv2_srblur","cv0_dmapdown131_thyroid_sam3_cv2_srblur"
    #              ,"cv0_dmapdown131_thyroid_sam3_cv3_srblur","cv0_dmapdown131_thyroid_sam4_cv4_srblur"]
    # for i in range(len(name_list)):
    #
    #     foler =r"/media/ubuntu/SSD_3/Liangshubo/Thyroid_plus/dataset/rawdata/n20_thyroid_v2_pretrain/process_3/"+name_list[i]
    #     save_path= foler+"_maskcrop"
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path,exist_ok=True)
    #     mask_process2(foler,save_path)