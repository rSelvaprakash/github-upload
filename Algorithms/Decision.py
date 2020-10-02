import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
import time

train_url = 'dataset/NSL_KDD_Train.csv'
test_url = 'dataset/NSL_KDD_Test.csv'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


df = pd.read_csv(train_url,header=None, names = col_names)

df_test = pd.read_csv(test_url, header=None, names = col_names)

print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)

df.head(5)
print('Label distribution Training set:')
print(df['label'].value_counts())
print()
print('Label distribution Test set:')
print(df_test['label'].value_counts())


# **Step 1: Data preprocessing:**
# 
# One-Hot-Encoding

# protocol_type (column 2), service (column 3), flag (column 4).

print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())
# Test set
print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


# **LabelEncoder**
# 
# **Insert categorical features into a 2D numpy array**

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]

df_categorical_values.head()

# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
print(unique_protocol2)

# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
print(unique_service2)


# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
print(unique_flag2)


# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2


#do it for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


# **Transform categorical features into numbers using LabelEncoder()**
df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

print(df_categorical_values.head())
print('--------------------')
print(df_categorical_values_enc.head())

# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)


# **One-Hot-Encoding**

enc = OneHotEncoder(categories='auto')
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)


# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data.head()

trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference
for col in difference:
    testdf_cat_data[col] = 0

print(df_cat_data.shape)    
print(testdf_cat_data.shape)

newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)

# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)

print(newdf.shape)
print(newdf_test.shape)
labeldf=newdf['label']
labeldf_test=newdf_test['label']


# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1,
                            'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1 })
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1
                           ,'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1})


# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test

x = newdf.drop(['duration','land','wrong_fragment','urgent','num_failed_logins','logged_in','num_compromised','num_file_creations','num_root','root_shell','su_attempted','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','flag_S2','flag_S3','flag_SH','srv_rerror_rate','service_csnet_ns','service_ctf','service_daytime','service_discard','service_domain','service_domain_u','service_echo','service_eco_i','service_ecr_i','service_efs','service_exec','service_finger','service_ftp','service_ftp_data','service_gopher','service_netbios_ns','service_ldap','service_kshell','service_klogin','service_iso_tsap','service_imap4','service_http_443','service_hostnames','service_netbios_dgm','service_name','service_mtp','service_login','service_link','service_pop_3','service_pop_2','service_pm_dump','service_other','service_ntp_u','service_nntp','service_nnsp','service_netstat','service_netbios_ssn','service_ssh','service_sql_net','service_sunrpc','service_smtp','service_shell','service_rje','service_remote_job','service_private','service_printer','service_uucp_path','service_uucp','service_urp_i','service_time','service_tim_i','service_tftp_u','service_telnet','service_systat','service_supdup','dst_host_count','srv_diff_host_rate','diff_srv_rate','flag_S0','flag_S1','rerror_rate','flag_RSTR','flag_RSTOS0','flag_RSTO','flag_REJ','flag_OTH','service_whois','service_vmnet','srv_serror_rate','serror_rate','service_urh_i','service_red_i','service_harvest','service_http_2784','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','Protocol_type_tcp','Protocol_type_udp','service_IRC','service_X11','service_Z39_50','service_auth','service_bgp','service_courier','service_http_8001','service_aol'], axis=1)

x_test =newdf_test.drop(['duration','land','wrong_fragment','urgent','num_failed_logins','logged_in','num_compromised','num_file_creations','num_root','root_shell','su_attempted','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','flag_S2','flag_S3','flag_SH','srv_rerror_rate','service_csnet_ns','service_ctf','service_daytime','service_discard','service_domain','service_domain_u','service_echo','service_eco_i','service_ecr_i','service_efs','service_exec','service_finger','service_ftp','service_ftp_data','service_gopher','service_netbios_ns','service_ldap','service_kshell','service_klogin','service_iso_tsap','service_imap4','service_http_443','service_hostnames','service_netbios_dgm','service_name','service_mtp','service_login','service_link','service_pop_3','service_pop_2','service_pm_dump','service_other','service_ntp_u','service_nntp','service_nnsp','service_netstat','service_netbios_ssn','service_ssh','service_sql_net','service_sunrpc','service_smtp','service_shell','service_rje','service_remote_job','service_private','service_printer','service_uucp_path','service_uucp','service_urp_i','service_time','service_tim_i','service_tftp_u','service_telnet','service_systat','service_supdup','dst_host_count','srv_diff_host_rate','diff_srv_rate','flag_S0','flag_S1','rerror_rate','flag_RSTR','flag_RSTOS0','flag_RSTO','flag_REJ','flag_OTH','service_whois','service_vmnet','srv_serror_rate','serror_rate','service_urh_i','service_red_i','service_harvest','service_http_2784','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','Protocol_type_tcp','Protocol_type_udp','service_IRC','service_X11','service_Z39_50','service_auth','service_bgp','service_courier','service_http_8001','service_aol'], axis=1)


# **Step 2: Feature Scaling**


# Split dataframes into X & Y
# X Özellikler , Y sonuç değişkenleri
X_Df = x.drop('label',1)
Y_Df = newdf.label

# test set
X_Df_test = x_test.drop('label',1)
Y_Df_test = newdf_test.label


X_Df.shape


colNames=list(X_Df)
colNames_test=list(X_Df_test)


from sklearn import preprocessing
scaler1 = preprocessing.StandardScaler().fit(X_Df)
X_Df=scaler1.transform(X_Df) 

# test data
scaler5 = preprocessing.StandardScaler().fit(X_Df_test)
X_Df_test=scaler5.transform(X_Df_test) 


# ## Random Forest - Feature Selection

# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(n_estimators=10,n_jobs=2)
# rfe = RFE(estimator=clf, n_features_to_select=13, step=1)


# rfe.fit(X_Df, Y_Df.astype(int))
# X_rfeDoS=rfe.transform(X_Df)
# true=rfe.support_
# rfecolindex_DoS=[i for i, x in enumerate(true) if x]
# rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)


# ### **Summary of features selected by RFE**


# print('Features selected for DoS:',rfecolname_DoS)
# print()


print(X_Df.shape)

# # Decision Tree

import time
from sklearn.tree import DecisionTreeClassifier 

# Create Decision Tree classifer object
clf_Tree = DecisionTreeClassifier()
train0 = time.time()
# Train Decision Tree Classifer
clf_Tree = clf_Tree.fit(X_Df, Y_Df.astype(int))
train1 = time.time() - train0

test0 = time.time()
Y_Df_pred=clf_Tree.predict(X_Df_test)
test1 = time.time() - test0

# Create confusion matrix
pd.crosstab(Y_Df_test, Y_Df_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])

from sklearn.model_selection import cross_val_score
from sklearn import metrics
accuracy = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_Tree, X_Df_test, Y_Df_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
print("train_time:%.3fs\n" %train1)
print("test_time:%.3fs\n" %test1)

