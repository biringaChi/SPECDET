import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


csv_file = '../datasets/cpu_processes/cpu_states.csv'


dataset=pd.read_csv(csv_file)

#func to clean dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#dataset = dataset[(dataset.T != 0).any()]
spectre_attack_df = dataset[dataset['class']=="M"]
no_spectre_attack_df = dataset[dataset['class']=="B"]
#help(spectre_attack_df.plot)
axes = no_spectre_attack_df.plot(kind='scatter', x='ps2', y='ps3', color='blue', label='no spectre')
spectre_attack_df.plot(kind='scatter', x='ps2', y='ps3', color='red', label='spectre attack', ax=axes)
#print(dataset.dtypes)

features_df = dataset[['ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7', 'ps8', 'ps9', 'ps10', 'ps11', 'ps12', 'ps13', 'ps14', 'ps15', 'ps16', 'ps17', 'ps18', 'ps19', 'ps20', 'ps21', 'ps22', 'ps23', 'ps24', 'ps25', 'ps26', 'ps27', 'ps28', 'ps29', 'ps30', 'ps31', 'ps32', 'ps33', 'ps34', 'ps35', 'ps36', 'ps37', 'ps38', 'ps39', 'ps40', 'ps41', 'ps42', 'ps43', 'ps44', 'ps45', 'ps46', 'ps47', 'ps48', 'ps49', 'ps50', 'ps51', 'ps52', 'ps53', 'ps54', 'ps55', 'ps56', 'ps57', 'ps58', 'ps59', 'ps60', 'ps61', 'ps62', 'ps63', 'ps64', 'ps65', 'ps66', 'ps67', 'ps68', 'ps69', 'ps70', 'ps71', 'ps72', 'ps73', 'ps74', 'ps75', 'ps76', 'ps77', 'ps78', 'ps79', 'ps80', 'ps81', 'ps82', 'ps83', 'ps84', 'ps85', 'ps86', 'ps87', 'ps88', 'ps89', 'ps90', 'ps91', 'ps92', 'ps93', 'ps94', 'ps95', 'ps96', 'ps97', 'ps98', 'ps99', 'ps100', 'ps101', 'ps102', 'ps103', 'ps104', 'ps105', 'ps106', 'ps107', 'ps108', 'ps109', 'ps110', 'ps111', 'ps112', 'ps113', 'ps114', 'ps115', 'ps116', 'ps117', 'ps118', 'ps119', 'ps120', 'ps121', 'ps122', 'ps123', 'ps124', 'ps125', 'ps126', 'ps127', 'ps128', 'ps129', 'ps130', 'ps131', 'ps132', 'ps133', 'ps134', 'ps135', 'ps136', 'ps137', 'ps138', 'ps139', 'ps140', 'ps141', 'ps142', 'ps143', 'ps144', 'ps145', 'ps146', 'ps147', 'ps148', 'ps149', 'ps150', 'ps151', 'ps152', 'ps153', 'ps154', 'ps155', 'ps156', 'ps157', 'ps158', 'ps159', 'ps160', 'ps161', 'ps162', 'ps163', 'ps164', 'ps165', 'ps166', 'ps167', 'ps168', 'ps169', 'ps170', 'ps171', 'ps172', 'ps173', 'ps174', 'ps175', 'ps176', 'ps177', 'ps178', 'ps179', 'ps180', 'ps181', 'ps182', 'ps183', 'ps184', 'ps185', 'ps186', 'ps187', 'ps188', 'ps189', 'ps190', 'ps191', 'ps192', 'ps193', 'ps194', 'ps195', 'ps196', 'ps197', 'ps198', 'ps199', 'ps200', 'ps201', 'ps202', 'ps203', 'ps204', 'ps205', 'ps206', 'ps207', 'ps208', 'ps209', 'ps210', 'ps211', 'ps212', 'ps213', 'ps214', 'ps215', 'ps216', 'ps217', 'ps218', 'ps219', 'ps220', 'ps221', 'ps222', 'ps223', 'ps224', 'ps225', 'ps226', 'ps227', 'ps228', 'ps229', 'ps230', 'ps231', 'ps232', 'ps233', 'ps234', 'ps235', 'ps236', 'ps237', 'ps238', 'ps239', 'ps240', 'ps241', 'ps242', 'ps243', 'ps244', 'ps245', 'ps246', 'ps247', 'ps248', 'ps249', 'ps250', 'ps251', 'ps252', 'ps253', 'ps254', 'ps255', 'ps256', 'ps257', 'ps258', 'ps259', 'ps260', 'ps261', 'ps262', 'ps263', 'ps264', 'ps265', 'ps266', 'ps267', 'ps268', 'ps269', 'ps270', 'ps271', 'ps272', 'ps273', 'ps274', 'ps275', 'ps276', 'ps277', 'ps278', 'ps279', 'ps280', 'ps281', 'ps282', 'ps283', 'ps284', 'ps285', 'ps286', 'ps287', 'ps288', 'ps289', 'ps290', 'ps291', 'ps292', 'ps293', 'ps294', 'ps295', 'ps296', 'ps297', 'ps298', 'ps299', 'ps300', 'ps301', 'ps302', 'ps303', 'ps304', 'ps305', 'ps306', 'ps307', 'ps308', 'ps309', 'ps310', 'ps311', 'ps312', 'ps313', 'ps314', 'ps315', 'ps316', 'ps317', 'ps318', 'ps319', 'ps320', 'ps321', 'ps322', 'ps323', 'ps324', 'ps325', 'ps326', 'ps327', 'ps328', 'ps329', 'ps330', 'ps331', 'ps332', 'ps333', 'ps334', 'ps335', 'ps336', 'ps337', 'ps338', 'ps339', 'ps340', 'ps341', 'ps342', 'ps343', 'ps344', 'ps345', 'ps346', 'ps347', 'ps348', 'ps349', 'ps350', 'ps351', 'ps352', 'ps353', 'ps354', 'ps355', 'ps356', 'ps357', 'ps358', 'ps359', 'ps360', 'ps361', 'ps362', 'ps363', 'ps364', 'ps365', 'ps366', 'ps367', 'ps368', 'ps369', 'ps370', 'ps371', 'ps372', 'ps373', 'ps374', 'ps375', 'ps376', 'ps377', 'ps378', 'ps379', 'ps380', 'ps381', 'ps382', 'ps383', 'ps384', 'ps385', 'ps386', 'ps387', 'ps388', 'ps389', 'ps390', 'ps391', 'ps392', 'ps393', 'ps394', 'ps395', 'ps396', 'ps397', 'ps398', 'ps399', 'ps400', 'ps401', 'ps402', 'ps403', 'ps404', 'ps405', 'ps406', 'ps407', 'ps408', 'ps409', 'ps410', 'ps411', 'ps412', 'ps413', 'ps414', 'ps415', 'ps416', 'ps417', 'ps418', 'ps419', 'ps420', 'ps421', 'ps422', 'ps423', 'ps424', 'ps425', 'ps426', 'ps427', 'ps428', 'ps429', 'ps430', 'ps431', 'ps432', 'ps433', 'ps434', 'ps435', 'ps436', 'ps437', 'ps438', 'ps439', 'ps440', 'ps441', 'ps442', 'ps443', 'ps444', 'ps445', 'ps446', 'ps447', 'ps448', 'ps449', 'ps450', 'ps451', 'ps452', 'ps453', 'ps454', 'ps455', 'ps456', 'ps457', 'ps458', 'ps459', 'ps460', 'ps461', 'ps462', 'ps463', 'ps464', 'ps465', 'ps466', 'ps467', 'ps468', 'ps469', 'ps470', 'ps471', 'ps472', 'ps473', 'ps474', 'ps475', 'ps476', 'ps477', 'ps478', 'ps479', 'ps480', 'ps481', 'ps482', 'ps483', 'ps484', 'ps485', 'ps486', 'ps487', 'ps488', 'ps489', 'ps490', 'ps491', 'ps492', 'ps493', 'ps494', 'ps495', 'ps496', 'ps497', 'ps498', 'ps499', 'ps500', 'ps501', 'ps502', 'ps503', 'ps504', 'ps505', 'ps506', 'ps507', 'ps508', 'ps509', 'ps510', 'ps511', 'ps512', 'ps513', 'ps514', 'ps515', 'ps516', 'ps517', 'ps518', 'ps519', 'ps520', 'ps521', 'ps522', 'ps523', 'ps524', 'ps525', 'ps526', 'ps527', 'ps528', 'ps529', 'ps530', 'ps531', 'ps532', 'ps533', 'ps534', 'ps535', 'ps536', 'ps537', 'ps538', 'ps539', 'ps540', 'ps541', 'ps542', 'ps543', 'ps544', 'ps545', 'ps546', 'ps547', 'ps548', 'ps549', 'ps550', 'ps551', 'ps552', 'ps553', 'ps554', 'ps555','ps556']]
#features_df = features_df[(features_df.T != 0).any()]
X = np.array(features_df)
y = np.asarray(dataset['class'])
#print(x[0:5])

#Divide the data into train/test, train(80%)/test(20%)
# Train has two components (X,y), X is a 2 dimensional array , y is 1D
#Test (X,y)X is a 2 dimensional array , y is 1D
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=4)
#print(X_test.shape)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)
# Modeling SVM using scikit-learn
from sklearn import svm
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)

#Results and evaluations
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict ))