import cv2, os, random, numpy, math

def normalizeHist(hist, lenght):
    for i in range(0, len(hist)):
        hist[i] /= lenght
        hist[i] *= 10000
    return hist

def findHistogram(img, layer, len):
    hist_array = []
    for i in range(0, len):
        hist_array.append(0)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            index = img[i][j][layer]
            hist_array[index] += 1
    hist_array = normalizeHist(hist_array, img.shape[0]*img.shape[1])
    return hist_array

def euclidianDistance(hist_train, hist_test):
    total = 0
    for i in range(0, len(hist_train)):
        total += pow(hist_train[i]-hist_test[i], 2)
    return math.sqrt(total)


######################################################
choice = input("1. Histogramlari olustur ve kaydet.\n2.Benzerlik olc")

if (choice == '1'):
    fileNames = ["028.camel\\", "056.dog\\", "057.dolphin\\", "084.giraffe\\", "089.goose\\", "105.horse\\"]
    for name in fileNames:
        img_dirs = os.listdir(name)
        hists_list = []
        for img_dir in img_dirs:
            rgbImage = cv2.imread(name + img_dir)
            hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
            hist_H = findHistogram(hsvImage, 0, 360)
            hist_B = findHistogram(rgbImage, 0, 256)
            hist_G = findHistogram(rgbImage, 1, 256)
            hist_R = findHistogram(rgbImage, 2, 256)
            hist = numpy.array([img_dir, hist_H, hist_R, hist_G, hist_B], dtype=object)
            hists_list.append(hist)

        hist_file = numpy.array(hists_list, dtype=object)
        file_dir = name[:-1]
        numpy.save(file_dir + "_hist", hist_file)

else:
    fileNames = ["028.camel_hist.npy", "056.dog_hist.npy", "057.dolphin_hist.npy", "084.giraffe_hist.npy", "089.goose_hist.npy", "105.horse_hist.npy"]
    all_train_samples = []
    all_test_samples = []
    for name in fileNames:
        hists = numpy.load(name, allow_pickle = True)
        number_of_rows = hists.shape[0]
        random_indices = numpy.random.choice(number_of_rows, size=30, replace=False)
        random_rows = hists[random_indices]
        train_samples = random_rows[:25]
        test_samples = random_rows[25:]
        all_train_samples.extend(train_samples)
        all_test_samples.extend(test_samples)

    total = 0
    right = 0
    for test_s in all_test_samples:
        hist_H_distance = []
        hist_R_distance = []
        hist_G_distance = []
        hist_B_distance = []
        hist_RGB_distance = []
        for train_s in all_train_samples:
            hist_H_distance.append([test_s[0], train_s[0], euclidianDistance(test_s[1], train_s[1])])
            hist_R_distance.append([test_s[0], train_s[0], euclidianDistance(test_s[2], train_s[2])])
            hist_G_distance.append([test_s[0], train_s[0], euclidianDistance(test_s[3], train_s[3])])
            hist_B_distance.append([test_s[0], train_s[0], euclidianDistance(test_s[4], train_s[4])])
            rgb_distance = (euclidianDistance(test_s[2], train_s[2]) + euclidianDistance(test_s[3], train_s[3]) + euclidianDistance(test_s[4], train_s[4]))/3
            hist_RGB_distance.append([test_s[0], train_s[0], rgb_distance])

        similiar_pics_H = sorted(hist_H_distance, key=lambda x: x[2])
        similiar_pics_R = sorted(hist_R_distance, key=lambda x: x[2])
        similiar_pics_G = sorted(hist_G_distance, key=lambda x: x[2])
        similiar_pics_B = sorted(hist_B_distance, key=lambda x: x[2])
        similiar_pics_RGB = sorted(hist_RGB_distance, key=lambda x: x[2])

        flag = True
        for i in range(0, 5):
            print(test_s[0], "icin ", i+1, ". ler")
            print(" Test resmi   -   Egitim resmi   -   Uzaklik    -    Domain")
            print(similiar_pics_H[i], "Domain : H")
            print(similiar_pics_RGB[i], "Domain : RGB")
            print("************************************************************")
            if(similiar_pics_H[i][1][:3] == test_s[0][:3] and flag):
                right += 1
                flag = False
            elif (similiar_pics_RGB[i][1][:3] == test_s[0][:3] and flag):
                right += 1
                flag = False
        total += 1

    print("Dogruluk = ", right/total)




